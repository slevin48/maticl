function R = tabicl_row_interact(row, cfg, E, varargin)
%TABICL_ROW_INTERACT Row-wise transformer with RoPE and CLS aggregation.
%   row : row-interaction weights (M.row), cfg : model config (M.cfg)
%   E : [T x d x H] cell embeddings from TABICL_COL_EMBED
%   R : [T x (num_cls*d)] row representations
%   TABICL_ROW_INTERACT(row, cfg, E, CHUNK) processes CHUNK rows at a time (256).
chunk = 256;
if numel(varargin) > 0
    chunk = cell2mat(varargin(1));
end
[T, d, H] = size(E);
C = cfg.row_num_cls;
nh = cfg.row_nhead;
S = C + H;
nb = row.num_blocks;

R = zeros(T, C * d);
for t0 = 1:chunk:T
    rows = t0:min(T, t0 + chunk - 1);
    Tb = numel(rows);
    % tokens of a row are contiguous: [cls_1..cls_C, cell_1..cell_H]
    Z = zeros(S, Tb, d);
    Z(1:C, :, :) = repmat(reshape(row.cls, C, 1, d), [1 Tb 1]);
    Z(C+1:S, :, :) = permute(E(rows, :, :), [3 1 2]);
    x = reshape(Z, S * Tb, d);
    for b = 1:nb-1
        x = tabicl_mab(row.(sprintf('b%d', b)), x, [], Tb, nh, row.freqs);
    end
    % the last block only updates the CLS tokens
    x3 = reshape(x, S, Tb, d);
    xq = reshape(x3(1:C, :, :), C * Tb, d);
    y = tabicl_mab(row.(sprintf('b%d', nb)), xq, x, Tb, nh, row.freqs);
    y = tabicl_layernorm(y, row.lnw, row.lnb);
    R(rows, :) = reshape(permute(reshape(y, C, Tb, d), [2 3 1]), Tb, C * d);
end
end
