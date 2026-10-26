function [logits, proba] = tabicl_forward(M, Xtr, ytr, Xte)
%TABICL_FORWARD Raw TabICL forward pass on one preprocessed table.
%   Xtr : [n x H] training features, ytr : n 0-based labels
%   Xte : [m x H] test features
%   logits : [m x C] raw logits, C = number of distinct training labels
%   proba  : [] if C <= M.cfg.max_classes. Otherwise the hierarchical
%            class probabilities, and logits = 0.9 * log(proba + 1e-6)
%            (see TABICL_ICL), as returned by the PyTorch model.
ytr = ytr(:);
num_classes = numel(unique(ytr));
X = cat(1, Xtr, Xte);
E = tabicl_col_embed(M.col, M.cfg, X, ytr);
R = tabicl_row_interact(M.row, M.cfg, E);
[logits, proba] = tabicl_icl(M.icl, M.cfg, R, ytr, num_classes);
end
