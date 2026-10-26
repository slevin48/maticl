function y = tabicl_gelu(x)
%TABICL_GELU Exact (erf-based) GELU activation, as torch.nn.functional.gelu.
y = 0.5 * x .* (1 + erf(x / sqrt(2)));
end
