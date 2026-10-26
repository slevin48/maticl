# maticl: TabICL in pure MATLAB

A port of [TabICL](https://github.com/soda-inria/tabicl) v2, the tabular
in-context-learning foundation model from Inria's Soda team, to plain MATLAB
with no toolboxes. Give it a labelled training table and it classifies new
rows, or predicts a continuous target, in a single forward pass, with no
training or tuning. It reproduces the Python `TabICLClassifier` and
`TabICLRegressor`, including the preprocessing and ensembling, and is tested
on three runtimes:

* [MATLAB](https://www.mathworks.com/products/matlab.html)
* [Nelson](https://nelson-lang.github.io/nelson-website/) 2.0
* [RunMat](https://runmat.com) 0.6.2

```matlab
addpath src
M   = tabicl_load();                           % pretrained weights, once
clf = tabicl_fit(Xtrain, ytrain, 'Model', M);  % "fit" = store context
[yhat, proba] = tabicl_predict(clf, Xtest);

R   = tabicl_load('regressor');                % regression checkpoint
reg = tabicl_fit_regressor(Xtrain, ytrain, 'Model', R);
[yhat, out] = tabicl_predict_regressor(reg, Xtest);   % mean, median, quantiles
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
| diabetes regression: max error of mean / median / quantiles, in units of std(y) | 2.9e-6 / 3.7e-6 / 2.0e-5 | 2.9e-6 / 3.7e-6 / 2.0e-5 |
| full test suite wall time | 50 s | about 29 min (21 min + 7.5 min for the regressor) |

The remaining differences are float32 rounding (PyTorch computes in float32,
maticl in double). Predicted labels agree 100% on all three datasets. The
largest regression error is in the extrapolated tail (level 0.9995); inside
the predicted quantiles it is below 4e-6.

## Setup

Nothing to install. On first use `tabicl_load()` downloads the official
checkpoint from Hugging Face
([jingang/TabICL](https://huggingface.co/jingang/TabICL),
`tabicl-classifier-v2-20260212.ckpt`, 110 MB) into `weights/` and reads it
directly. A PyTorch `.ckpt` is a zip of raw float32 tensor files, and the
committed index `assets/tabicl-classifier-v2.idx` says which file holds which
tensor. This works in MATLAB (including MATLAB Online), Nelson and RunMat.
`tabicl_load('regressor')` does the same with
`tabicl-regressor-v2-20260212.ckpt` (114 MB) and
`assets/tabicl-regressor-v2.idx`.

Optionally, with Python you can export a single flat weight file, which loads
slightly faster:

```sh
uv venv .venv-ref -p 3.12
uv pip install --python .venv-ref/bin/python tabicl
.venv-ref/bin/python tools/export_weights.py   # -> weights/tabicl-classifier-v2.{idx,bin}
.venv-ref/bin/python tools/export_weights.py --model regressor
```

`tabicl_load` uses `weights/tabicl-<classifier|regressor>-v2.bin` if it exists.

## Usage

```matlab
addpath src
M = tabicl_load();                 % or tabicl_load('path/to/prefix')

clf = tabicl_fit(X, y, 'Model', M, 'NumEstimators', 8);
proba = tabicl_predict_proba(clf, Xtest);   % columns follow clf.classes
yhat  = tabicl_predict(clf, Xtest);
```

* `X` is a numeric matrix; `NaN` values are mean-imputed.
* `y` can be numeric, a string array or a cellstr, with at most 10 classes.
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

### Regression

```matlab
R = tabicl_load('regressor');
reg = tabicl_fit_regressor(X, y, 'Model', R, 'NumEstimators', 8);
yhat = tabicl_predict_regressor(reg, Xtest);                  % predicted mean
[yhat, out] = tabicl_predict_regressor(reg, Xtest, 'Alphas', [0.05 0.5 0.95]);
% out.mean, out.median, out.quantiles (one column per level), out.alphas
```

`y` is a numeric vector; it is standardised internally and predictions are
mapped back. `tabicl_fit_regressor` takes the options of `TabICLRegressor`:
`'Model'`, `'NumEstimators'`, `'NormMethods'`, `'FeatShuffle'`,
`'OutlierThreshold'`, `'Seed'` and `'Ensemble'`, with the same defaults as
`tabicl_fit` (there is no class shuffling). The model predicts 999
quantiles per row; the median and other quantiles interpolate between them,
with exponential tails beyond levels 0.001 and 0.999, as
`QuantileDistribution` does.

Lower-level entry points are `tabicl_forward(M, Xtr, ytr, Xte)` (raw logits
on an already-preprocessed table, raw quantiles with the regressor) and the
three stages `tabicl_col_embed`, `tabicl_row_interact` and `tabicl_icl`.

## Running the tests

The fixtures in `tests/fixtures` (inputs and PyTorch outputs) are committed,
so the tests need neither Python nor a manual weight export (they load both
checkpoints):

```sh
nelson-cli -f tests/run_tests.m
cd "$(mktemp -d)" && runmat /path/to/maticl/tests/run_tests.m
```

```matlab
run('tests/run_tests.m')          % MATLAB
```

* RunMat compiles every `.m` file in the working directory, so start it from
  a directory without unrelated `.m` files.
* `tools/make_reference.py [clf|reg]` regenerates the fixtures from the
  Python package.
* `tools/export_weights.py [--model regressor]` also regenerates the index in
  `assets/`.

## Documentation

* [docs/architecture.md](docs/architecture.md): the model and the pipeline,
  stage by stage, mapped to the PyTorch sources.
* [docs/portability.md](docs/portability.md): the MATLAB subset that runs on
  all three runtimes, the RunMat 0.6.2 workarounds, and performance notes.

## Layout

```
src/        the implementation (22 functions)
tests/      run_tests.m, test_forward.m, test_classifier.m, test_regressor.m, fixtures/
tools/      export_weights.py, make_reference.py (PyTorch side)
assets/     tensor indexes of the Hugging Face checkpoints
docs/       architecture and portability notes
example.m   minimal demo
```

## Limitations

* At most 10 classes (no hierarchical many-class mode).
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
