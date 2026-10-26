function clf = tabicl_fit(X, y, varargin)
%TABICL_FIT Fit a TabICL classifier (in-context: stores the preprocessed data).
%   clf = TABICL_FIT(X, y) with X an [n x p] numeric matrix (NaN allowed)
%   and y a vector of numeric labels, a string array or a cellstr.
%   X may also be a table with numeric, categorical, string, cellstr and
%   logical variables (see TABICL_ENCODE_FIT), and y the name of one of them.
%
%   Name-value options (defaults follow tabicl.TabICLClassifier):
%     'Model'              struct from TABICL_LOAD (loaded on demand if empty)
%     'NumEstimators'      8
%     'NormMethods'        {'none', 'power'}
%     'FeatShuffle'        'latin'  ('latin' | 'shift' | 'random' | 'none')
%     'ClassShuffle'       'shift'  ('latin' | 'shift' | 'random' | 'none')
%     'OutlierThreshold'   4
%     'SoftmaxTemperature' 0.9
%     'AverageLogits'      true
%     'Seed'               42      ([] keeps the current RNG state)
%     'Ensemble'           explicit ensemble to replay, a struct with fields
%                          norms (cellstr), norm_idx (k x 1), feat (k x p)
%                          and cls (k x C), permutations being 0-based
opts = struct();
opts.Model = [];
opts.NumEstimators = 8;
opts.NormMethods = {'none', 'power'};
opts.FeatShuffle = 'latin';
opts.ClassShuffle = 'shift';
opts.OutlierThreshold = 4;
opts.SoftmaxTemperature = 0.9;
opts.AverageLogits = true;
opts.Seed = 42;
opts.Ensemble = [];
if numel(varargin) > 0
    user = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);
    names = fieldnames(user);
    for i = 1:numel(names)
        name = char(names(i));
        if ~isfield(opts, name)
            error('tabicl:fit', 'Unknown option ''%s''.', name);
        end
        opts.(name) = user.(name);
    end
end
if ischar(opts.NormMethods)
    opts.NormMethods = {opts.NormMethods};
end

M = opts.Model;
if isempty(M)
    M = tabicl_load();
end

enc = [];
if isa(X, 'table')
    [X, y, enc] = tabicl_encode_fit(X, y);
end
X = double(X);
if size(X, 1) ~= numel(y)
    error('tabicl:fit', 'X and y must have the same number of rows.');
end

% mean imputation (empty columns become 0)
[X, fill] = impute(X);

% label encoding
if iscell(y)
    y = string(y);
end
[classes, ~, yi] = unique(y(:));
yi = yi(:) - 1;
C = numel(classes);
if C > M.cfg.max_classes
    error('tabicl:fit', ['%d classes found but only %d are supported (the hierarchical ' ...
        'many-class strategy of the Python package is not implemented).'], C, M.cfg.max_classes);
end

% drop constant features
keep = non_constant(X);
Xk = X(:, keep(:)');
H = size(Xk, 2);

if ~isempty(opts.Seed)
    rng(opts.Seed);
end
if isempty(opts.Ensemble)
    ens = make_ensemble(H, C, opts);
else
    ens = opts.Ensemble;
end

% one preprocessing pipeline per normalisation method in use
norms = ens.norms;
used = unique(ens.norm_idx);
pp = struct();
for k = 1:numel(used)
    method = char(norms(used(k)));
    pp.(method) = tabicl_preprocess_fit(Xk, method, opts.OutlierThreshold);
end

clf = struct();
clf.model = M;
clf.classes = classes;
clf.y = yi;
clf.encoder = enc;
clf.fill = fill;
clf.keep = keep;
clf.pp = pp;
clf.ensemble = ens;
clf.opts = opts;
end

function [X, fill] = impute(X)
fill = zeros(1, size(X, 2));
for j = 1:size(X, 2)
    c = X(~isnan(X(:, j)), j);
    if ~isempty(c)
        fill(j) = mean(c);
    end
    X(isnan(X(:, j)), j) = fill(j);
end
end

function keep = non_constant(X)
% indices of the columns with more than one distinct value (at least one)
n = size(X, 1);
nuniq = zeros(1, size(X, 2));
for j = 1:size(X, 2)
    nuniq(j) = numel(unique(X(:, j)));
end
keep = find(nuniq > 1 | n <= 1);
if isempty(keep)
    keep = 1;
end
keep = reshape(keep, 1, numel(keep));
end

function ens = make_ensemble(H, C, opts)
n_est = opts.NumEstimators;
feat = shuffles(H, opts.FeatShuffle, n_est);
cls = shuffles(C, opts.ClassShuffle, n_est);
% all (feature, class) pattern pairs in random order, each with every norm method
[fi, ci] = meshgrid(1:size(feat, 1), 1:size(cls, 1));
pairs = cat(2, fi(:), ci(:));
pairs = pairs(randperm(size(pairs, 1)), :);
nn = numel(opts.NormMethods);
total = min(n_est, size(pairs, 1) * nn);
m = (1:total)';
p = pairs(floor((m - 1) / nn) + 1, :);
ens = struct();
ens.norms = opts.NormMethods;
ens.norm_idx = mod(m - 1, nn) + 1;
ens.feat = feat(p(:, 1), :);
ens.cls = cls(p(:, 2), :);
end

function P = shuffles(n, method, n_est)
% rows of P are 0-based permutations of 0:n-1
P = 0:n-1;
if strcmp(method, 'latin') && n > 4000
    method = 'random';
end
if strcmp(method, 'none') || n_est == 1
    return;
elseif strcmp(method, 'shift')
    P = mod((0:n-1) - (0:n-1)', n);
elseif strcmp(method, 'random')
    if n <= 5
        P = perms(0:n-1);
        P = P(randperm(size(P, 1), min(n_est, size(P, 1))), :);
    else
        P = zeros(n_est, n);
        for i = 1:n_est
            P(i, :) = randperm(n) - 1;
        end
    end
elseif strcmp(method, 'latin')
    % random Latin square: permute rows, columns and symbols of a cyclic one
    L = mod((0:n-1)' + (0:n-1), n);
    L = L(randperm(n), randperm(n));
    sym = randperm(n) - 1;
    P = reshape(sym(L + 1), n, n);
else
    error('tabicl:fit', 'Unknown shuffle method ''%s''.', method);
end
end
