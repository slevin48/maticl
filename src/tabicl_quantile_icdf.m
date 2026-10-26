function v = tabicl_quantile_icdf(Q, alphas)
%TABICL_QUANTILE_ICDF Quantile function of the regressor's output distribution.
%   Q : [m x N] sorted quantiles at the levels (1:N) / (N + 1)
%   v : [m x numel(alphas)] values at the probability levels ALPHAS
%   Port of QuantileDistribution.icdf (tabicl/_model/quantile_dist.py,
%   tail_type 'exp'): linear interpolation between the predicted quantiles
%   and exponential tails, fitted on the 20 outermost quantiles, beyond them.
tol = 1e-6;
N = size(Q, 2);
lev = (1:N) / (N + 1);
k = min(20, floor(N / 4));

% left tail: Q(a) = beta_l * log(a) + b_l
la = log(max(lev(1:k), tol));
lc = la - mean(la);
Ql = Q(:, 1:k);
beta_l = mean((Ql - mean(Ql, 2)) .* lc, 2) / max(mean(lc .^ 2), tol);
beta_l = min(max(abs(beta_l), 0.01), 100);
b_l = Q(:, 1) - beta_l * log(max(lev(1), tol));
% right tail: Q(a) = -beta_r * log(1 - a) + b_r
ra = log(max(1 - lev(N-k+1:N), tol));
rc = ra - mean(ra);
Qr = Q(:, N-k+1:N);
beta_r = -mean((Qr - mean(Qr, 2)) .* rc, 2) / max(mean(rc .^ 2), tol);
beta_r = min(max(abs(beta_r), 0.01), 100);
b_r = Q(:, N) + beta_r * log(1 - min(lev(N), 1 - tol));

v = zeros(size(Q, 1), numel(alphas));
for i = 1:numel(alphas)
    a = alphas(i);
    if a < lev(1)
        v(:, i) = beta_l * log(max(a, tol)) + b_l;
    elseif a > lev(N)
        v(:, i) = -beta_r * log(max(1 - a, tol)) + b_r;
    elseif a >= lev(N)
        v(:, i) = Q(:, N);
    else
        s = min(max(sum(lev(1:N-1) <= a), 1), N - 1);
        t = (a - lev(s)) / max(lev(s+1) - lev(s), tol);
        t = min(t, 1) * (t > 0);   % clamp to [0, 1]
        v(:, i) = Q(:, s) + t * (Q(:, s+1) - Q(:, s));
    end
end
end
