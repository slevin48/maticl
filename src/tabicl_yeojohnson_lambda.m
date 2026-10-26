function lmbda = tabicl_yeojohnson_lambda(x)
%TABICL_YEOJOHNSON_LAMBDA Maximum-likelihood Yeo-Johnson parameter.
%   Port of scipy.stats.yeojohnson_normmax (brack=None): bounded Brent
%   minimisation (scipy.optimize.fminbound) of the negative log-likelihood.
x = x(~isnan(x));
x = x(:);
if all(x == 0)
    lmbda = 1;
    return;
end
log1p_max_x = log1p(20 * max(abs(x)));
log_eps = log(eps);
log_tiny = (log(realmin) - log_eps) / 2;
log_max = (log(realmax) + log_eps) / 2;
lb = log_tiny / log1p_max_x;
ub = log_max / log1p_max_x;
if all(x < 0)
    t = lb;
    lb = 2 - ub;
    ub = 2 - t;
elseif any(x < 0)
    t = lb;
    lb = max(2 - ub, lb);
    ub = min(2 - t, ub);
end
lmbda = fminbound(x, lb, ub, 1.48e-8, 500);
end

function f = neg_llf(lmbda, x)
n = numel(x);
y = tabicl_yeojohnson(x, lmbda);
v = var(y, 1);
if v >= realmin
    logvar = log(v);
else
    logvar = -Inf;
end
llf = -n / 2 * logvar + (lmbda - 1) * sum(sign(x) .* log1p(abs(x)));
if isinf(llf)
    llf = -Inf;
end
f = -llf;
end

function xf = fminbound(data, a, b, xatol, maxfun)
% scipy.optimize.fminbound (bounded Brent) applied to neg_llf(., data)
sqrt_eps = sqrt(2.2e-16);
golden_mean = 0.5 * (3 - sqrt(5));
fulc = a + golden_mean * (b - a);
nfc = fulc;
xf = fulc;
rat = 0;
e = 0;
x = xf;
fx = neg_llf(x, data);
num = 1;
ffulc = fx;
fnfc = fx;
xm = 0.5 * (a + b);
tol1 = sqrt_eps * abs(xf) + xatol / 3;
tol2 = 2 * tol1;
while abs(xf - xm) > (tol2 - 0.5 * (b - a))
    golden = true;
    if abs(e) > tol1
        golden = false;
        r = (xf - nfc) * (fx - ffulc);
        q = (xf - fulc) * (fx - fnfc);
        p = (xf - fulc) * q - (xf - nfc) * r;
        q = 2 * (q - r);
        if q > 0
            p = -p;
        end
        q = abs(q);
        r = e;
        e = rat;
        if abs(p) < abs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf)
            rat = p / q;
            x = xf + rat;
            if (x - a) < tol2 || (b - x) < tol2
                rat = tol1 * (sign(xm - xf) + ((xm - xf) == 0));
            end
        else
            golden = true;
        end
    end
    if golden
        if xf >= xm
            e = a - xf;
        else
            e = b - xf;
        end
        rat = golden_mean * e;
    end
    x = xf + (sign(rat) + (rat == 0)) * max(abs(rat), tol1);
    fu = neg_llf(x, data);
    num = num + 1;
    if fu <= fx
        if x >= xf
            a = xf;
        else
            b = xf;
        end
        fulc = nfc;
        ffulc = fnfc;
        nfc = xf;
        fnfc = fx;
        xf = x;
        fx = fu;
    else
        if x < xf
            a = x;
        else
            b = x;
        end
        if fu <= fnfc || nfc == xf
            fulc = nfc;
            ffulc = fnfc;
            nfc = x;
            fnfc = fu;
        elseif fu <= ffulc || fulc == xf || fulc == nfc
            fulc = x;
            ffulc = fu;
        end
    end
    xm = 0.5 * (a + b);
    tol1 = sqrt_eps * abs(xf) + xatol / 3;
    tol2 = 2 * tol1;
    if num >= maxfun
        break;
    end
end
end
