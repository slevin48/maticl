function logits = tabicl_icl(icl, cfg, R, ytr, num_classes)
%TABICL_ICL Dataset-wise in-context learning transformer.
%   icl : in-context learning weights (M.icl), cfg : model config (M.cfg)
%   R   : [T x D] row representations, first numel(ytr) rows are training rows
%   ytr : 0-based training labels (scaled targets for the regressor)
%   logits : [(T - numel(ytr)) x num_classes] raw logits for the test rows
%            (raw quantiles for the regressor, num_classes = num_quantiles)
n = numel(ytr);
nh = cfg.icl_nhead;
if cfg.max_classes == 0
    R(1:n, :) = R(1:n, :) + ytr(:) * icl.yW + icl.yb;   % regressor: linear target encoder
else
    R(1:n, :) = R(1:n, :) + tabicl_onehot(ytr, cfg.max_classes) * icl.yW + icl.yb;
end
for b = 1:icl.num_blocks
    % every row attends to the training rows only
    R = tabicl_mab(icl.(sprintf('b%d', b)), R, R(1:n, :), 1, nh, []);
end
h = tabicl_layernorm(R(n+1:end, :), icl.lnw, icl.lnb);
out = tabicl_gelu(h * icl.dW1 + icl.db1) * icl.dW2 + icl.db2;
logits = out(:, 1:num_classes);
end
