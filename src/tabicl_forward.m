function logits = tabicl_forward(M, Xtr, ytr, Xte)
%TABICL_FORWARD Raw TabICL forward pass on one preprocessed table.
%   Xtr : [n x H] training features, ytr : n 0-based labels (< max_classes)
%   Xte : [m x H] test features
%   logits : [m x C] raw logits, C = number of distinct training labels
%
%   With the regressor (M.cfg.max_classes == 0), ytr holds standardised
%   targets and the output is [m x num_quantiles] raw (unsorted) quantiles.
ytr = ytr(:);
num_classes = M.cfg.num_quantiles;
if M.cfg.max_classes > 0
    num_classes = numel(unique(ytr));
end
if max(ytr) + 1 > M.cfg.max_classes && M.cfg.max_classes > 0
    error('tabicl:classes', 'At most %d classes are supported natively.', M.cfg.max_classes);
end
X = cat(1, Xtr, Xte);
E = tabicl_col_embed(M.col, M.cfg, X, ytr);
R = tabicl_row_interact(M.row, M.cfg, E);
logits = tabicl_icl(M.icl, M.cfg, R, ytr, num_classes);
end
