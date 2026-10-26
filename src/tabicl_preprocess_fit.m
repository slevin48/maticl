function pp = tabicl_preprocess_fit(X, method, varargin)
%TABICL_PREPROCESS_FIT Fit the per-normalisation preprocessing pipeline.
%   Standard scaling (clipped to [-100, 100]), optional Yeo-Johnson power
%   transform ('power') or none ('none'), then soft outlier clipping.
%   pp.Xtr holds the transformed training data. An optional third argument
%   sets the outlier threshold (default 4).
outlier_threshold = 4;
if numel(varargin) > 0
    outlier_threshold = cell2mat(varargin(1));
end
n = size(X, 1);
pp = struct();
pp.method = method;
pp.lambdas = ones(1, size(X, 2));
pp.pt_mean = zeros(1, size(X, 2));
pp.pt_scale = ones(1, size(X, 2));

pp.mean = mean(X, 1);
pp.scale = std(X, 1, 1) + 1e-6;
Xs = min(max((X - pp.mean) ./ pp.scale, -100), 100);

if strcmp(method, 'power')
    H = size(Xs, 2);
    lambdas = ones(1, H);
    mu = mean(Xs, 1);
    va = var(Xs, 1, 1);
    for i = 1:H
        % sklearn leaves (near-)constant features untransformed
        if va(i) <= n * eps * va(i) + (n * mu(i) * eps) ^ 2
            continue;
        end
        lambdas(i) = tabicl_yeojohnson_lambda(Xs(:, i));
        Xs(:, i) = tabicl_yeojohnson(Xs(:, i), lambdas(i));
    end
    pp.lambdas = lambdas;
    pp.pt_mean = mean(Xs, 1);
    sd = std(Xs, 1, 1);
    sd(sd < 10 * eps * max(1, abs(pp.pt_mean))) = 1;
    pp.pt_scale = sd;
    Xs = (Xs - pp.pt_mean) ./ pp.pt_scale;
elseif ~strcmp(method, 'none')
    error('tabicl:preprocess', 'Unsupported normalisation method ''%s''.', method);
end

% outlier bounds from mean +- threshold * std, re-estimated without outliers
ddof = double(n > 1);
[mu, sd] = nan_mean_std(Xs, ddof);
Xc = Xs;
Xc(Xs < mu - outlier_threshold * sd | Xs > mu + outlier_threshold * sd) = NaN;
[mu, sd] = nan_mean_std(Xc, ddof);
pp.lower = mu - outlier_threshold * sd;
pp.upper = mu + outlier_threshold * sd;
pp.Xtr = clip_outliers(Xs, pp.lower, pp.upper);
end

function [mu, sd] = nan_mean_std(X, ddof)
H = size(X, 2);
mu = zeros(1, H);
sd = zeros(1, H);
for i = 1:H
    c = X(~isnan(X(:, i)), i);
    mu(i) = mean(c);
    sd(i) = sqrt(sum((c - mu(i)) .^ 2) / max(numel(c) - ddof, 1));
end
sd = max(sd, 1e-6);
end

function X = clip_outliers(X, lower, upper)
X = max(-log1p(abs(X)) + lower, X);
X = min(log1p(abs(X)) + upper, X);
end
