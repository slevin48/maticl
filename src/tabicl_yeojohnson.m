function y = tabicl_yeojohnson(x, lmbda)
%TABICL_YEOJOHNSON Yeo-Johnson power transform (as scipy.stats.yeojohnson).
y = zeros(size(x));
pos = x >= 0;
if abs(lmbda) < eps
    y(pos) = log1p(x(pos));
else
    y(pos) = expm1(lmbda * log1p(x(pos))) / lmbda;
end
if abs(lmbda - 2) > eps
    y(~pos) = -expm1((2 - lmbda) * log1p(-x(~pos))) / (2 - lmbda);
else
    y(~pos) = -log1p(-x(~pos));
end
y(isnan(x)) = NaN;
end
