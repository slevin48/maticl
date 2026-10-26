function c = tabicl_bmm(a, b)
%TABICL_BMM Batched matrix product over the third dimension.
%   Uses pagemtimes when the runtime provides it, otherwise loops over pages.
if exist('pagemtimes') > 0 %#ok<EXIST>
    c = pagemtimes(a, b);
    return;
end
np = size(a, 3);
c = zeros(size(a, 1), size(b, 2), np);
for i = 1:np
    c(:, :, i) = a(:, :, i) * b(:, :, i);
end
end
