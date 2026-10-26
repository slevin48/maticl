function X = tabicl_preprocess_transform(pp, X)
%TABICL_PREPROCESS_TRANSFORM Apply a fitted preprocessing pipeline.
X = min(max((X - pp.mean) ./ pp.scale, -100), 100);
if strcmp(pp.method, 'power')
    lambdas = pp.lambdas;
    for i = 1:size(X, 2)
        X(:, i) = tabicl_yeojohnson(X(:, i), lambdas(i));
    end
    X = (X - pp.pt_mean) ./ pp.pt_scale;
end
X = max(-log1p(abs(X)) + pp.lower, X);
X = min(log1p(abs(X)) + pp.upper, X);
end
