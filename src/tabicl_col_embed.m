function E = tabicl_col_embed(col, cfg, X, ytr)
%TABICL_COL_EMBED Target-aware column-wise embedding (set transformer).
%   col : column-embedding weights (M.col), cfg : model config (M.cfg)
%   X   : [T x H] table, the first numel(ytr) rows are training rows
%   ytr : 0-based training labels (scaled targets for the regressor)
%   E   : [T x d x H] one embedding per cell
%
%   With more than cfg.max_classes classes, labels are split into
%   mixed-radix digits (each with at most max_classes values), the set
%   transformer runs once per digit and the embeddings are averaged.
[T, H] = size(X);
ytr = ytr(:);
n = numel(ytr);
d = cfg.embed_dim;
nh = cfg.col_nhead;
g = cfg.col_feature_group_size;
skip = -100;

if cfg.max_classes == 0
    yemb = ytr * col.yW + col.yb;   % regressor: linear target encoder
    D = 1;
else
    % one target embedding per mixed-radix digit: yemb(:, :, k)
    bases = mixed_radix_bases(max(ytr) + 1, cfg.max_classes);
    D = numel(bases);
    yemb = zeros(n, d, D);
    for k = 1:D
        divisor = prod(bases(k+1:end));
        digit = mod(floor(ytr / divisor), bases(k));
        yemb(:, :, k) = tabicl_onehot(digit, cfg.max_classes) * col.yW + col.yb;
    end
end

E = zeros(T, d, H);
for j = 1:H
    % "same" feature grouping: column j sees features j+1, j+2, j+4 (mod H)
    idx = mod((j - 1) + 2 .^ (0:g-1), H) + 1;
    F = X(:, idx);
    src0 = F * col.inW + col.inb;
    skipped = all(F == skip, 2);
    src0(skipped, :) = skip;
    acc = zeros(T, d);
    for k = 1:D
        src = src0;
        src(1:n, :) = src(1:n, :) + yemb(:, :, k);
        if all(src(:) == skip)
            acc = acc + src;
            continue;
        end
        for b = 1:col.num_blocks
            blk = col.(sprintf('b%d', b));
            % inducing points attend to the training rows, then all rows attend back
            hidden = tabicl_mab(blk.mab1, blk.ind, src(1:n, :), 1, nh, []);
            src = tabicl_mab(blk.mab2, src, hidden, 1, nh, []);
        end
        acc = acc + src;
    end
    E(:, :, j) = acc / D;
end
end

function bases = mixed_radix_bases(C, max_classes)
% Balanced bases [k_1 .. k_D], each <= max_classes, with prod >= C
% (ColEmbedding._compute_mixed_radix_bases).
if C <= max_classes
    bases = C;
    return;
end
D = ceil(log(C) / log(max_classes));
k = min(ceil(C ^ (1 / D)), max_classes);
bases = k * ones(1, D);
p = k ^ D;
i = 1;
while p < C && i <= D
    if bases(i) < max_classes
        p = p / bases(i) * (bases(i) + 1);
        bases(i) = bases(i) + 1;
    end
    i = i + 1;
end
end
