function p = tabicl_softmax(x, dim)
%TABICL_SOFTMAX Numerically stable softmax along dimension DIM.
e = exp(x - max(x, [], dim));
p = e ./ sum(e, dim);
end
