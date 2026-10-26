# maticl: TabICL in pure MATLAB

A port of [TabICL](https://github.com/soda-inria/tabicl) v2, the tabular
in-context-learning foundation model from Inria's Soda team, to plain MATLAB
with no toolboxes. Give it a labelled training table and it classifies new
rows in a single forward pass, with no training or tuning. It reproduces the
Python `TabICLClassifier`, including the preprocessing and ensembling, and is
tested on three runtimes:

* [MATLAB](https://www.mathworks.com/products/matlab.html)
* [Nelson](https://nelson-lang.github.io/nelson-website/) 2.0
* [RunMat](https://runmat.com) 0.6.2

```matlab
addpath src
M   = tabicl_load();                           % pretrained weights, once
clf = tabicl_fit(Xtrain, ytrain, 'Model', M);  % "fit" = store context
[yhat, proba] = tabicl_predict(clf, Xtest);
```

## Results

Compared with the reference Python package (`tabicl` 2.2.0, same checkpoint,
same ensemble):

| test | Nelson 2.0 | RunMat 0.6.2 |
|---|---|---|
| raw forward pass, max logit error | 1.9e-5 | 1.9e-5 |
| iris: max probability error / accuracy (Python 1.000) | 2.1e-6 / 1.000 | 2.1e-6 / 1.000 |
| wine: max probability error / accuracy (Python 1.000) | 3.8e-7 / 1.000 | 3.8e-7 / 1.000 |
| breast cancer: max probability error / accuracy (Python 0.9532) | 3.8e-6 / 0.9532 | 3.8e-6 / 0.9532 |
| 26 classes, raw forward pass: max probability error | 7.5e-6 | 7.5e-6 |
| 15 classes (synthetic): max probability error / accuracy (Python 0.8889) | 3.5e-6 / 0.8889 | 3.4e-6 / 0.8889 |
| full test suite wall time | 47 s | 21 min + 25 min (many-class tests) |

The remaining differences are float32 rounding (PyTorch computes in float32,
maticl in double). Predicted labels agree 100% on all datasets. The last two
rows use more classes than the checkpoint's 10, which goes through the
many-class strategy (see below).

## Setup

Nothing to install. On first use `tabicl_load()` downloads the official
checkpoint from Hugging Face
([jingang/TabICL](https://huggingface.co/jingang/TabICL),
`tabicl-classifier-v2-20260212.ckpt`, 110 MB) into `weights/` and reads it
directly. A PyTorch `.ckpt` is a zip of raw float32 tensor files, and the
committed index `assets/tabicl-classifier-v2.idx` says which file holds which
tensor. This works in MATLAB (including MATLAB Online), Nelson and RunMat.

Optionally, with Python you can export a single flat weight file, which loads
slightly faster:

```sh
uv venv .venv-ref -p 3.12
uv pip install --python .venv-ref/bin/python tabicl
.venv-ref/bin/python tools/export_weights.py   # -> weights/tabicl-classifier-v2.{idx,bin}
```

`tabicl_load` uses `weights/tabicl-classifier-v2.bin` if it exists.

## Usage

```matlab
addpath src
M = tabicl_load();                 % or tabicl_load('path/to/prefix')

clf = tabicl_fit(X, y, 'Model', M, 'NumEstimators', 8);
proba = tabicl_predict_proba(clf, Xtest);   % columns follow clf.classes
yhat  = tabicl_predict(clf, Xtest);
```

* `X` is a numeric matrix; `NaN` values are mean-imputed.
* `y` can be numeric, a string array or a cellstr, with any number of
  classes. Beyond 10 (the checkpoint's limit), maticl uses the same
  many-class strategy as Python: mixed-radix label ensembling in the column
  embedding and hierarchical classification in the ICL stage. Each ensemble
  member then costs one column-embedding pass per label digit and one ICL
  pass per tree node (2 and 4 for 26 classes).
* `example.m` is a two-moons demo.

`tabicl_fit` options (defaults follow `TabICLClassifier`):

| option | default | meaning |
|---|---|---|
| `'Model'` | loaded on demand | struct from `tabicl_load`; pass it to avoid reloading |
| `'NumEstimators'` | 8 | ensemble size |
| `'NormMethods'` | `{'none', 'power'}` | normalisations cycled through the ensemble |
| `'FeatShuffle'` | `'latin'` | feature permutations: `latin`, `shift`, `random`, `none` |
| `'ClassShuffle'` | `'shift'` | class permutations, same choices |
| `'OutlierThreshold'` | 4 | in standard deviations |
| `'SoftmaxTemperature'` | 0.9 | |
| `'AverageLogits'` | true | average logits (else probabilities) over members |
| `'Seed'` | 42 | RNG seed for the permutations (`[]` keeps the current state) |
| `'Ensemble'` | | explicit members to replay, see `tests/test_classifier.m` |

Lower-level entry points are `[logits, proba] = tabicl_forward(M, Xtr, ytr, Xte)`
(raw logits on an already-preprocessed table; `proba` holds the hierarchical
class probabilities when there are more than 10 classes, `[]` otherwise) and
the three stages `tabicl_col_embed`, `tabicl_row_interact` and `tabicl_icl`.

## Running the tests

The fixtures in `tests/fixtures` (inputs and PyTorch outputs) are committed,
so the tests need neither Python nor a manual weight export:

```sh
nelson-cli -f tests/run_tests.m
cd "$(mktemp -d)" && runmat /path/to/maticl/tests/run_tests.m
```

```matlab
run('tests/run_tests.m')          % MATLAB
```

* RunMat compiles every `.m` file in the working directory, so start it from
  a directory without unrelated `.m` files.
* `tools/make_reference.py` regenerates the fixtures from the Python package.
* `tools/export_weights.py` also regenerates `assets/tabicl-classifier-v2.idx`.

## Documentation

* [docs/architecture.md](docs/architecture.md): the model and the pipeline,
  stage by stage, mapped to the PyTorch sources.
* [docs/portability.md](docs/portability.md): the MATLAB subset that runs on
  all three runtimes, the RunMat 0.6.2 workarounds, and performance notes.

## Layout

```
src/        the implementation (19 functions)
tests/      run_tests.m, test_forward.m, test_classifier.m, fixtures/
tools/      export_weights.py, make_reference.py (PyTorch side)
assets/     tensor index of the Hugging Face checkpoint
docs/       architecture and portability notes
example.m   minimal demo
```

## Limitations

* Classification only (no regressor).
* Numeric inputs only: encode categorical columns yourself.
* CPU, double precision. Nelson takes about 2 s per ensemble member on a
  400-row, 30-feature table. RunMat is currently about 30× slower because
  of its matrix-multiply speed and per-statement overhead (see
  docs/portability.md).

## License and credits

BSD 3-Clause, see [LICENSE](LICENSE). maticl is a derivative of
[TabICL](https://github.com/soda-inria/tabicl), © 2025 Soda team @ Inria.
If you use it, please cite the TabICL papers:

* Qu, Holzmüller, Varoquaux, Le Morvan, *TabICL: A Tabular Foundation Model
  for In-Context Learning on Large Data*, ICML 2025.
* the TabICLv2 paper for the v2 checkpoint used here (see the
  [TabICL README](https://github.com/soda-inria/tabicl#citation)).
