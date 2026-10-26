% EXAMPLE Minimal TabICL classification in MATLAB / Nelson / RunMat.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'src'));

% two noisy interleaved half-moons
rng(0);
n = 300;
t = pi * rand(n, 1);
lab = rand(n, 1) > 0.5;
X = [cos(t) + 1 * lab, sin(t) .* (1 - 2 * lab) + 0.5 * lab] + 0.15 * randn(n, 2);
y = double(lab);

idx = randperm(n);
tr = idx(1:200);
te = idx(201:end);

M = tabicl_load();                         % weights/tabicl-classifier-v2.{idx,bin}
clf = tabicl_fit(X(tr, :), y(tr), 'Model', M);
[yhat, proba] = tabicl_predict(clf, X(te, :));
fprintf('test accuracy: %.3f\n', mean(yhat == y(te)));
