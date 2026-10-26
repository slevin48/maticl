function [yhat, out] = tabicl_predict_regressor(reg, X, varargin)
%TABICL_PREDICT_REGRESSOR Predictions of a TabICL regressor for the rows of X.
%   yhat = TABICL_PREDICT_REGRESSOR(reg, X) is the predicted mean [m x 1].
%   [yhat, out] = ... also returns a struct with fields
%     mean       [m x 1] mean of the predicted distribution (= yhat)
%     median     [m x 1]
%     quantiles  [m x numel(alphas)] quantiles at the levels out.alphas
%     alphas     [1 x k] probability levels
%   Each statistic is computed per ensemble member and averaged, as in
%   TabICLRegressor.predict.
%
%   Name-value options:
%     'Alphas'   quantile levels in (0, 1), default 0.1:0.1:0.9
alphas = (1:9) / 10;
if numel(varargin) > 0
    user = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);
    names = fieldnames(user);
    for i = 1:numel(names)
        name = char(names(i));
        if ~strcmp(name, 'Alphas')
            error('tabicl:predict', 'Unknown option ''%s''.', name);
        end
        alphas = reshape(user.Alphas, 1, []);
    end
end
M = reg.model;
ens = reg.ensemble;
norms = ens.norms;
norm_idx = ens.norm_idx;
feat = ens.feat;
ytr = reg.y;
fill = reg.fill;
keep = reg.keep;

X = double(X);
for j = 1:size(X, 2)
    X(isnan(X(:, j)), j) = fill(j);
end
X = X(:, keep(:)');

% preprocess the test rows once per normalisation method
Xte = struct();
methods_used = fieldnames(reg.pp);
for k = 1:numel(methods_used)
    method = char(methods_used(k));
    Xte.(method) = tabicl_preprocess_transform(reg.pp.(method), X);
end

m = size(X, 1);
mu = zeros(m, 1);
med = zeros(m, 1);
qs = zeros(m, numel(alphas));
for e = 1:numel(norm_idx)
    method = char(norms(norm_idx(e)));
    pp = reg.pp.(method);
    Xtr_e = pp.Xtr;
    Xte_e = Xte.(method);
    f = feat(e, :) + 1;
    % raw quantiles, made monotonic by sorting (QuantileToDistribution)
    Q = sort(tabicl_forward(M, Xtr_e(:, f), ytr, Xte_e(:, f)), 2);
    mu = mu + mean(Q, 2);
    med = med + tabicl_quantile_icdf(Q, 0.5);
    qs = qs + tabicl_quantile_icdf(Q, alphas);
end
k = numel(norm_idx);
out = struct();
out.mean = mu / k * reg.y_scale + reg.y_mean;
out.median = med / k * reg.y_scale + reg.y_mean;
out.quantiles = qs / k * reg.y_scale + reg.y_mean;
out.alphas = alphas;
yhat = out.mean;
end
