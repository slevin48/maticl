"""Generate test fixtures and reference outputs from the PyTorch TabICL package.

Matrices are written as: int32 rows, int32 cols, then float64 values in
column-major order (see tests/read_matrix.m).

Fixtures (in tests/fixtures/<case>/):
  forward : raw model forward on a fixed, already-preprocessed table
            Xtr, ytr, Xte -> logits (+ intermediate E1, R for debugging)
  clf_*   : full TabICLClassifier.fit / predict_proba on a dataset, with the
            ensemble configuration exported so MATLAB can replay it exactly
  clf_mixed : the same on a DataFrame with categorical, string and boolean
            columns (needs pandas), with the encoded matrices Xtr_enc, Xte_enc
"""

import struct
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def write_matrix(path, a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", a.shape[0], a.shape[1]))
        f.write(np.asfortranarray(a).astype("<f8").tobytes(order="F"))


def forward_case(clf):
    out = ROOT / "forward"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_train, n_test, H, C = 60, 20, 5, 3
    X = rng.standard_normal((n_train + n_test, H))
    X[:, 1] = X[:, 0] * 0.5 + 0.3 * X[:, 1]
    y = (X[:n_train, 0] + 0.5 * X[:n_train, 2] > 0).astype(int) + (X[:n_train, 3] > 1).astype(int)
    assert len(np.unique(y)) == C

    model = clf.model_
    cfg = clf.inference_config_
    Xt = torch.from_numpy(X).float()[None]
    yt = torch.from_numpy(y).float()[None]
    with torch.no_grad():
        emb = model.col_embedder(Xt.clone(), y_train=yt, mgr_config=cfg.COL_CONFIG)  # (1, T, H+4, d)
        reps = model.row_interactor(emb.clone(), mgr_config=cfg.ROW_CONFIG)  # (1, T, 4d)
        logits = model(Xt, yt, return_logits=True, inference_config=cfg)  # (1, n_test, C)

    write_matrix(out / "Xtr.bin", X[:n_train])
    write_matrix(out / "ytr.bin", y)
    write_matrix(out / "Xte.bin", X[n_train:])
    write_matrix(out / "E1.bin", emb[0, :, 4, :].numpy())  # embedding of first feature column
    write_matrix(out / "R.bin", reps[0].numpy())
    write_matrix(out / "logits.bin", logits[0].numpy())
    print("forward:", logits.shape)


def clf_case(name, X, y, n_estimators=4, test_size=0.3):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=42, device="cpu")
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    acc = float((clf.classes_[proba.argmax(1)] == yte).mean())

    # Ensemble members in output order: methods in dict order, then members.
    eg = clf.ensemble_generator_
    lines = []
    for method, configs in eg.ensemble_configs_.items():
        for feat, cls in configs:
            lines.append(f"{method} | {' '.join(map(str, feat))} | {' '.join(map(str, cls))}")
    (out / "ensemble.txt").write_text("\n".join(lines) + "\n")

    write_matrix(out / "Xtr.bin", Xtr)
    write_matrix(out / "ytr.bin", ytr)
    write_matrix(out / "Xte.bin", Xte)
    write_matrix(out / "yte.bin", yte)
    write_matrix(out / "proba.bin", proba)
    # Preprocessed training table of the first member for preprocessing checks
    for method, pp in eg.preprocessors_.items():
        write_matrix(out / f"Xpp_{method}.bin", pp.X_transformed_)
        write_matrix(out / f"Xte_pp_{method}.bin", pp.transform(eg.unique_filter_.transform(clf.X_encoder_.transform(Xte))))
    print(f"{name}: train={len(ytr)} test={len(yte)} features={X.shape[1]} acc={acc:.4f}")
    return clf


def mixed_case(name="clf_mixed", n_train=80, n_test=30):
    """DataFrame with numeric, categorical, string and boolean columns.

    Text columns are written one value per line, an empty line meaning a
    missing value. Xtr_enc / Xte_enc hold the output of TransformToNumerical
    (categorical columns first, ordinal-encoded, then the numeric columns).
    """
    import pandas as pd

    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    n = n_train + n_test
    num = rng.standard_normal(n)
    cnt = rng.integers(0, 6, n)
    colour = rng.choice(["red", "green", "blue", "violet"], n)
    size = rng.choice(["Low", "high", "Medium", "low"], n)
    city = rng.choice(["Paris", "Lyon", "Nice"], n)
    flag = rng.random(n) > 0.4
    score = num + 0.8 * (colour == "red") - 0.8 * (colour == "blue") + 0.5 * flag + 0.3 * (size == "high")
    y = np.digitize(score + 0.3 * rng.standard_normal(n), [-0.3, 0.7])
    # missing values (NaN: sklearn would encode None in object columns as a
    # category of its own), and categories that only appear in the test rows
    num[rng.random(n) < 0.1] = np.nan
    colour = colour.astype(object)
    colour[rng.random(n) < 0.1] = np.nan
    colour[n_train + np.arange(0, n_test, 7)] = "orange"
    size = size.astype(object)
    size[rng.random(n) < 0.1] = np.nan
    city = city.astype(object)
    city[rng.random(n) < 0.1] = np.nan
    city[n_train + 3] = "Lille"

    df = pd.DataFrame({
        "num": num,
        "colour": pd.Categorical(colour, categories=["violet", "red", "green", "blue", "orange"]),
        "cnt": cnt,
        "size": pd.Series(size, dtype=object),
        "flag": pd.array(flag, dtype="boolean"),
        "city": pd.Series(city, dtype=object),
    })
    tr, te = df.iloc[:n_train], df.iloc[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    clf = TabICLClassifier(n_estimators=4, random_state=42, device="cpu")
    clf.fit(tr, ytr)
    proba = clf.predict_proba(te)
    acc = float((clf.classes_[proba.argmax(1)] == yte).mean())

    eg = clf.ensemble_generator_
    lines = []
    for method, configs in eg.ensemble_configs_.items():
        for feat, cls in configs:
            lines.append(f"{method} | {' '.join(map(str, feat))} | {' '.join(map(str, cls))}")
    (out / "ensemble.txt").write_text("\n".join(lines) + "\n")

    for part, d in (("tr", tr), ("te", te)):
        write_matrix(out / f"num_{part}.bin", d[["num", "cnt"]].to_numpy(dtype=float))
        write_matrix(out / f"flag_{part}.bin", d["flag"].to_numpy(dtype=float))
        for col in ("colour", "size", "city"):
            vals = ["" if pd.isna(v) else str(v) for v in d[col].astype(object)]
            (out / f"{col}_{part}.txt").write_text("\n".join(vals) + "\n")
        write_matrix(out / f"X{part}_enc.bin", clf.X_encoder_.transform(d))
    write_matrix(out / "ytr.bin", ytr)
    write_matrix(out / "yte.bin", yte)
    write_matrix(out / "proba.bin", proba)
    print(f"{name}: train={n_train} test={n_test} features={df.shape[1]} acc={acc:.4f}")


def main():
    torch.manual_seed(0)
    iris = load_iris()
    clf = clf_case("clf_iris", iris.data, iris.target)
    forward_case(clf)

    wine = load_wine()
    clf_case("clf_wine", wine.data, wine.target)

    bc = load_breast_cancer()
    clf_case("clf_breast_cancer", bc.data, bc.target)

    mixed_case()


if __name__ == "__main__":
    main()
