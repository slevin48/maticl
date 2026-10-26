function [logits, proba] = tabicl_icl(icl, cfg, R, ytr, num_classes)
%TABICL_ICL Dataset-wise in-context learning transformer.
%   icl : in-context learning weights (M.icl), cfg : model config (M.cfg)
%   R   : [T x D] row representations, first numel(ytr) rows are training rows
%   ytr : 0-based training labels (scaled targets for the regressor)
%   logits : [(T - numel(ytr)) x num_classes] raw logits for the test rows
%            (raw quantiles for the regressor, num_classes = num_quantiles)
%   proba  : [] when num_classes <= cfg.max_classes, and for the regressor
%
%   With more classes, hierarchical classification is used (as in
%   ICLearning._predict_hierarchical): classes are split into a tree of
%   groups with at most max_classes children, the transformer runs once per
%   node and the probabilities are multiplied down the tree. proba then
%   holds these probabilities and logits = 0.9 * log(proba + 1e-6), which is
%   what the PyTorch model returns with its default return_logits=True.
ytr = ytr(:);
n = numel(ytr);
proba = [];
if cfg.max_classes == 0 || num_classes <= cfg.max_classes
    logits = icl_logits(icl, cfg, R, ytr, num_classes);
    return;
end
proba = hierarchical(icl, cfg, R(1:n, :), ytr, R(n+1:end, :), num_classes);
logits = 0.9 * log(proba + 1e-6);
end

function logits = icl_logits(icl, cfg, R, ytr, num_classes)
% One run of the ICL transformer: logits of the test rows, first num_classes
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

function P = hierarchical(icl, cfg, Rtr, ytr, Rte, num_classes)
% Probabilities [m x num_classes] of the test rows. The tree is walked
% breadth-first with an explicit queue (no recursion, which RunMat's checker
% cannot handle): column q of 'members' marks the training rows of node q
% and column q of 'weight' holds the product of the group probabilities on
% the path from the root. Python calls the node softmaxes with its default
% temperature, 0.9, whatever the classifier's softmax_temperature.
T = 0.9;
maxc = cfg.max_classes;
m = size(Rte, 1);
P = zeros(m, num_classes);
members = ones(numel(ytr), 1);
weight = ones(m, 1);
q = 1;
while q <= size(members, 2)
    rows = find(members(:, q) > 0);
    rows = reshape(rows, 1, numel(rows));
    y = ytr(rows);
    w = weight(:, q);
    [cls, ~, pos] = unique(y);
    cls = reshape(cls, 1, numel(cls));
    pos = pos(:);
    k = numel(cls);
    Rn = cat(1, Rtr(rows, :), Rte);
    if k <= maxc
        % leaf: classify directly among the node's classes (label-encoded)
        p = tabicl_softmax(icl_logits(icl, cfg, Rn, pos - 1, k) / T, 2);
        % (a loop: P(:, cls + 1) = ... overflows RunMat's checker)
        for i = 1:k
            P(:, cls(i) + 1) = p(:, i) .* w;
        end
    else
        % internal node: balanced groups of consecutive classes (ICLearning._grouping)
        ng = min(ceil(k / maxc), maxc);
        assign = zeros(1, k);
        first = 1;
        left = k;
        for g = 1:ng
            sz = ceil(left / (ng - g + 1));
            assign(first:first+sz-1) = g - 1;
            first = first + sz;
            left = left - sz;
        end
        gi = assign(pos);
        gi = gi(:);
        gp = tabicl_softmax(icl_logits(icl, cfg, Rn, gi, ng) / T, 2);
        for g = 1:ng
            child = zeros(numel(ytr), 1);
            child(rows) = double(gi == g - 1);
            members = cat(2, members, child);
            weight = cat(2, weight, w .* gp(:, g));
        end
    end
    q = q + 1;
end
end
