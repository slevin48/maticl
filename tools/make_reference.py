"""Generate test fixtures and reference outputs from the PyTorch TabICL package.

Matrices are written as: int32 rows, int32 cols, then float64 values in
column-major order (see tests/read_matrix.m).

Fixtures (in tests/fixtures/<case>/):
  forward : raw model forward on a fixed, already-preprocessed table
            Xtr, ytr, Xte -> logits (+ intermediate E1, R for debugging)
  clf_*   : full TabICLClassifier.fit / predict_proba on a dataset, with the
            ensemble configuration exported so MATLAB can replay it exactly
  reg_*   : full TabICLRegressor.fit / predict (mean, median, quantiles),
            with its ensemble configuration

Usage: python tools/make_reference.py [clf|reg]   (default: all fixtures)
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier, TabICLRegressor

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


def reg_case(name, X, y, n_estimators=4, test_size=0.3):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=0)
    reg = TabICLRegressor(n_estimators=n_estimators, random_state=42, device="cpu")
    reg.fit(Xtr, ytr)
    # levels inside the predicted quantiles and in both exponential tails
    alphas = [0.0005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9995]
    pred = reg.predict(Xte, output_type=["mean", "median", "quantiles"], alphas=alphas)

    eg = reg.ensemble_generator_
    lines = []
    for method, configs in eg.ensemble_configs_.items():
        for feat, _ in configs:
            lines.append(f"{method} | {' '.join(map(str, feat))}")
    (out / "ensemble.txt").write_text("\n".join(lines) + "\n")

    write_matrix(out / "Xtr.bin", Xtr)
    write_matrix(out / "ytr.bin", ytr)
    write_matrix(out / "Xte.bin", Xte)
    write_matrix(out / "yte.bin", yte)
    write_matrix(out / "alphas.bin", np.array(alphas)[None])
    write_matrix(out / "mean.bin", pred["mean"])
    write_matrix(out / "median.bin", pred["median"])
    write_matrix(out / "quantiles.bin", pred["quantiles"])
    rmse = float(np.sqrt(np.mean((pred["mean"] - yte) ** 2)))
    print(f"{name}: train={len(ytr)} test={len(yte)} features={X.shape[1]} rmse={rmse:.4f}")


def main():
    which = sys.argv[1:] or ["clf", "reg"]
    torch.manual_seed(0)
    if "reg" in which:
        diabetes = load_diabetes()
        reg_case("reg_diabetes", diabetes.data, diabetes.target)
    if "clf" not in which:
        return
    iris = load_iris()
    clf = clf_case("clf_iris", iris.data, iris.target)
    forward_case(clf)

    wine = load_wine()
    clf_case("clf_wine", wine.data, wine.target)

    bc = load_breast_cancer()
    clf_case("clf_breast_cancer", bc.data, bc.target)


if __name__ == "__main__":
    main()
