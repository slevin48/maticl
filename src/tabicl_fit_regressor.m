function reg = tabicl_fit_regressor(X, y, varargin)
%TABICL_FIT_REGRESSOR Fit a TabICL regressor (in-context: stores the data).
%   reg = TABICL_FIT_REGRESSOR(X, y) with X an [n x p] numeric matrix (NaN
%   allowed) and y a numeric vector of continuous targets.
%
%   Name-value options (defaults follow tabicl.TabICLRegressor):
%     'Model'            struct from TABICL_LOAD('regressor') (loaded if empty)
%     'NumEstimators'    8
%     'NormMethods'      {'none', 'power'}
%     'FeatShuffle'      'latin'  ('latin' | 'shift' | 'random' | 'none')
%     'OutlierThreshold' 4
%     'Seed'             42       ([] keeps the current RNG state)
%     'Ensemble'         explicit ensemble to replay, a struct with fields
%                        norms (cellstr), norm_idx (k x 1) and feat (k x p),
%                        permutations being 0-based
%
%   The targets are standardised (as sklearn's StandardScaler); predictions
%   are mapped back by TABICL_PREDICT_REGRESSOR.
opts = struct();
opts.Model = [];
opts.NumEstimators = 8;
opts.NormMethods = {'none', 'power'};
opts.FeatShuffle = 'latin';
opts.OutlierThreshold = 4;
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

M = opts.Model;
if isempty(M)
    M = tabicl_load('regressor');
end
if M.cfg.max_classes ~= 0
    error('tabicl:fit', 'The model is a classifier: load it with tabicl_load(''regressor'').');
end

% Python casts y to float32 before scaling
y = double(single(double(y(:))));
if any(isnan(y))
    error('tabicl:fit', 'y must not contain NaN.');
end
y_mean = mean(y);
y_scale = std(y, 1);
if y_scale < 10 * eps
    y_scale = 1;
end

ens = opts.Ensemble;
if ~isempty(ens) && ~isfield(ens, 'cls')
    ens.cls = zeros(numel(ens.norm_idx), 1);
end

% Imputation, constant-column filtering, ensemble generation and the
% per-method preprocessing are those of the classifier, without class
% shuffling (TabICLRegressor shuffles features only). tabicl_fit gets dummy
% labels, two classes rather than one because RunMat cannot slice a 1 x 1
% class-permutation matrix, and a stub model: it only reads cfg.max_classes.
cfg = M.cfg;
cfg.max_classes = 2;
stub = struct();
stub.cfg = cfg;
clf = tabicl_fit(X, mod((0:numel(y)-1)', 2), 'Model', stub, ...
    'NumEstimators', opts.NumEstimators, 'NormMethods', opts.NormMethods, ...
    'FeatShuffle', opts.FeatShuffle, 'ClassShuffle', 'none', ...
    'OutlierThreshold', opts.OutlierThreshold, 'Seed', opts.Seed, 'Ensemble', ens);

reg = struct();
reg.model = M;
reg.y = (y - y_mean) / y_scale;
reg.y_mean = y_mean;
reg.y_scale = y_scale;
reg.fill = clf.fill;
reg.keep = clf.keep;
reg.pp = clf.pp;
reg.ensemble = clf.ensemble;
opts.Model = [];
reg.opts = opts;
end
