function ok = test_classifier(M, name)
%TEST_CLASSIFIER Compare the MATLAB classifier against TabICLClassifier.
%   Replays the exact Python ensemble (same shuffles), then also checks a
%   native MATLAB-generated ensemble for accuracy.
here = fileparts(which('read_matrix'));
fx = fullfile(here, 'fixtures', name);
Xtr = read_matrix(fullfile(fx, 'Xtr.bin'));
ytr = read_matrix(fullfile(fx, 'ytr.bin'));
Xte = read_matrix(fullfile(fx, 'Xte.bin'));
yte = read_matrix(fullfile(fx, 'yte.bin'));
ref = read_matrix(fullfile(fx, 'proba.bin'));
ens = read_ensemble(fullfile(fx, 'ensemble.txt'));

% --- exact replay of the Python ensemble ---------------------------------
t0 = tic;
clf = tabicl_fit(Xtr, ytr, 'Model', M, 'Ensemble', ens);
[yhat, proba] = tabicl_predict(clf, Xte);
elapsed = toc(t0);

pp_err = 0;
keep = clf.keep;
methods_used = fieldnames(clf.pp);
for k = 1:numel(methods_used)
    m = char(methods_used(k));
    pp = clf.pp.(m);
    pp_err = max(pp_err, max(max(abs(pp.Xtr - read_matrix(fullfile(fx, ['Xpp_' m '.bin']))))));
    Xt = tabicl_preprocess_transform(pp, Xte(:, keep));
    pp_err = max(pp_err, max(max(abs(Xt - read_matrix(fullfile(fx, ['Xte_pp_' m '.bin']))))));
end
perr = max(max(abs(proba - ref)));
[~, iref] = max(ref, [], 2);
classes = clf.classes;
agree = mean(classes(iref) == yhat);
acc = mean(yhat == yte);
[~, ipy] = max(ref, [], 2);
acc_py = mean(classes(ipy) == yte);
fprintf('%-18s replay : preproc err %.1e | proba err %.1e | label agreement %5.1f%% | acc %.4f (python %.4f) | %.1fs\n', ...
    name, pp_err, perr, 100 * agree, acc, acc_py, elapsed);
ok = perr < 1e-3 && agree == 1;

% --- native MATLAB ensemble (own random shuffles) ------------------------
t0 = tic;
n_est = numel(ens.norm_idx);
clf2 = tabicl_fit(Xtr, ytr, 'Model', M, 'NumEstimators', n_est);
yhat2 = tabicl_predict(clf2, Xte);
fprintf('%-18s native : acc %.4f (%d estimators) | %.1fs\n', name, mean(yhat2 == yte), n_est, toc(t0));
ok = ok && mean(yhat2 == yte) >= acc_py - 0.05;
end

function ens = read_ensemble(path)
% Lines look like "power | 2 0 3 1 | 2 0 1" (method | feature perm | class perm).
norms = {'none', 'power'};
fid = fopen(path, 'r');
idx = zeros(0, 1);
feat = [];
cls = [];
line = fgetl(fid);
while ischar(line)
    bars = strfind(line, '|');
    if numel(bars) == 2
        % char(codes(...)) instead of line(...): RunMat cannot index char arrays
        codes = double(line);
        idx = cat(1, idx, find(strcmp(norms, strtrim(char(codes(1:bars(1)-1))))));
        feat = cat(1, feat, reshape(sscanf(char(codes(bars(1)+1:bars(2)-1)), '%d'), 1, []));
        cls = cat(1, cls, reshape(sscanf(char(codes(bars(2)+1:end)), '%d'), 1, []));
    end
    line = fgetl(fid);
end
fclose(fid);
ens = struct();
ens.norms = norms;
ens.norm_idx = idx;
ens.feat = feat;
ens.cls = cls;
end
