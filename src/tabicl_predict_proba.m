function proba = tabicl_predict_proba(clf, X)
%TABICL_PREDICT_PROBA Class probabilities [m x C] for the rows of X.
%   Columns follow clf.classes.
M = clf.model;
ens = clf.ensemble;
norms = ens.norms;
norm_idx = ens.norm_idx;
feat = ens.feat;
cls = ens.cls;
ytr = clf.y;
fill = clf.fill;
keep = clf.keep;
T = clf.opts.SoftmaxTemperature;
average_logits = clf.opts.AverageLogits;

X = double(X);
for j = 1:size(X, 2)
    X(isnan(X(:, j)), j) = fill(j);
end
X = X(:, keep);

% preprocess the test rows once per normalisation method
Xte = struct();
methods_used = fieldnames(clf.pp);
for k = 1:numel(methods_used)
    method = char(methods_used(k));
    Xte.(method) = tabicl_preprocess_transform(clf.pp.(method), X);
end

avg = 0;
for m = 1:numel(norm_idx)
    method = char(norms(norm_idx(m)));
    pp = clf.pp.(method);
    Xtr_m = pp.Xtr;
    Xte_m = Xte.(method);
    f = feat(m, :) + 1;
    c = cls(m, :);
    logits = tabicl_forward(M, Xtr_m(:, f), c(ytr + 1), Xte_m(:, f));
    % undo the class permutation: column i is the logit of original class i
    out = logits(:, c + 1);
    if ~average_logits
        out = tabicl_softmax(out / T, 2);
    end
    avg = avg + out;
end
avg = avg / numel(norm_idx);
if average_logits
    avg = tabicl_softmax(avg / T, 2);
end
proba = avg ./ sum(avg, 2);
end
