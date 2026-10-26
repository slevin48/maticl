function y = tabicl_onehot(labels, num_classes)
%TABICL_ONEHOT One-hot encode 0-based integer labels into [N x num_classes].
labels = labels(:);
n = numel(labels);
y = zeros(n, num_classes);
y((1:n)' + n * labels) = 1;
end
