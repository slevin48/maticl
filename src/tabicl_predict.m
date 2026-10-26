function [labels, proba] = tabicl_predict(clf, X)
%TABICL_PREDICT Predicted class labels (and probabilities) for the rows of X.
proba = tabicl_predict_proba(clf, X);
[~, i] = max(proba, [], 2);
classes = clf.classes;
labels = classes(i);
labels = labels(:);
end
