function E = tabicl_col_embed(col, cfg, X, ytr)
%TABICL_COL_EMBED Target-aware column-wise embedding (set transformer).
%   col : column-embedding weights (M.col), cfg : model config (M.cfg)
%   X   : [T x H] table, the first numel(ytr) rows are training rows
%   ytr : 0-based training labels (scaled targets for the regressor)
%   E   : [T x d x H] one embedding per cell
[T, H] = size(X);
n = numel(ytr);
d = cfg.embed_dim;
nh = cfg.col_nhead;
g = cfg.col_feature_group_size;
skip = -100;

if cfg.max_classes == 0
    yemb = ytr(:) * col.yW + col.yb;   % regressor: linear target encoder
else
    yemb = tabicl_onehot(ytr, cfg.max_classes) * col.yW + col.yb;
end
E = zeros(T, d, H);
for j = 1:H
    % "same" feature grouping: column j sees features j+1, j+2, j+4 (mod H)
    idx = mod((j - 1) + 2 .^ (0:g-1), H) + 1;
    F = X(:, idx);
    src = F * col.inW + col.inb;
    skipped = all(F == skip, 2);
    src(skipped, :) = skip;
    src(1:n, :) = src(1:n, :) + yemb;
    if all(src(:) == skip)
        E(:, :, j) = skip;
        continue;
    end
    for b = 1:col.num_blocks
        blk = col.(sprintf('b%d', b));
        % inducing points attend to the training rows, then all rows attend back
        hidden = tabicl_mab(blk.mab1, blk.ind, src(1:n, :), 1, nh, []);
        src = tabicl_mab(blk.mab2, src, hidden, 1, nh, []);
    end
    E(:, :, j) = src;
end
end
