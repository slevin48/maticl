function y = tabicl_layernorm(x, w, b)
%TABICL_LAYERNORM Row-wise layer normalization (eps = 1e-5, as nn.LayerNorm).
%   x is [N x d]; w and b are [1 x d].
mu = mean(x, 2);
xc = x - mu;
v = mean(xc .^ 2, 2);
y = xc ./ sqrt(v + 1e-5) .* w + b;
end
