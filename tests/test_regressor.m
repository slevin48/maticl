function ok = test_regressor(M, name)
%TEST_REGRESSOR Compare the MATLAB regressor against TabICLRegressor.
%   M is the regressor model (tabicl_load('regressor')). Replays the exact
%   Python ensemble, then also checks a native MATLAB-generated ensemble.
%   Errors are reported relative to the standard deviation of the targets
%   (the model works on standardised targets).
here = fileparts(which('read_matrix'));
fx = fullfile(here, 'fixtures', name);
Xtr = read_matrix(fullfile(fx, 'Xtr.bin'));
ytr = read_matrix(fullfile(fx, 'ytr.bin'));
Xte = read_matrix(fullfile(fx, 'Xte.bin'));
yte = read_matrix(fullfile(fx, 'yte.bin'));
alphas = read_matrix(fullfile(fx, 'alphas.bin'));
ref_mean = read_matrix(fullfile(fx, 'mean.bin'));
ref_median = read_matrix(fullfile(fx, 'median.bin'));
ref_q = read_matrix(fullfile(fx, 'quantiles.bin'));
ens = read_reg_ensemble(fullfile(fx, 'ensemble.txt'));

% --- exact replay of the Python ensemble ---------------------------------
t0 = tic;
reg = tabicl_fit_regressor(Xtr, ytr, 'Model', M, 'Ensemble', ens);
[yhat, out] = tabicl_predict_regressor(reg, Xte, 'Alphas', alphas);
elapsed = toc(t0);

s = reg.y_scale;
e_mean = max(abs(yhat - ref_mean)) / s;
e_median = max(abs(out.median - ref_median)) / s;
e_q = max(max(abs(out.quantiles - ref_q))) / s;
rmse = sqrt(mean((yhat - yte) .^ 2));
rmse_py = sqrt(mean((ref_mean - yte) .^ 2));
fprintf('%-18s replay : rel err mean %.1e | median %.1e | quantiles %.1e | rmse %.4f (python %.4f) | %.1fs\n', ...
    name, e_mean, e_median, e_q, rmse, rmse_py, elapsed);
ok = max(cat(1, e_mean, e_median, e_q)) < 1e-4;

% --- native MATLAB ensemble (own random shuffles) ------------------------
t0 = tic;
n_est = numel(ens.norm_idx);
reg2 = tabicl_fit_regressor(Xtr, ytr, 'Model', M, 'NumEstimators', n_est);
yhat2 = tabicl_predict_regressor(reg2, Xte);
rmse2 = sqrt(mean((yhat2 - yte) .^ 2));
fprintf('%-18s native : rmse %.4f (%d estimators) | %.1fs\n', name, rmse2, n_est, toc(t0));
ok = ok && rmse2 <= 1.05 * rmse_py;
end

function ens = read_reg_ensemble(path)
% Lines look like "power | 2 0 3 1" (method | feature permutation).
norms = {'none', 'power'};
fid = fopen(path, 'r');
idx = zeros(0, 1);
feat = [];
line = fgetl(fid);
while ischar(line)
    bars = strfind(line, '|');
    if numel(bars) == 1
        % char(codes(...)) instead of line(...): RunMat cannot index char arrays
        codes = double(line);
        idx = cat(1, idx, find(strcmp(norms, strtrim(char(codes(1:bars(1)-1))))));
        feat = cat(1, feat, reshape(sscanf(char(codes(bars(1)+1:end)), '%d'), 1, []));
    end
    line = fgetl(fid);
end
fclose(fid);
ens = struct();
ens.norms = norms;
ens.norm_idx = idx;
ens.feat = feat;
end
