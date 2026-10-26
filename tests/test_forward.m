function ok = test_forward(M, varargin)
%TEST_FORWARD Compare the raw MATLAB forward pass against PyTorch outputs.
%   TEST_FORWARD(M, NAME) uses fixture NAME (default 'forward'). For
%   'forward_many' (more classes than max_classes) the hierarchical
%   probabilities are compared too.
name = 'forward';
if numel(varargin) > 0
    name = char(varargin(1));
end
here = fileparts(which('read_matrix'));
fx = fullfile(here, 'fixtures', name);
Xtr = read_matrix(fullfile(fx, 'Xtr.bin'));
ytr = read_matrix(fullfile(fx, 'ytr.bin'));
Xte = read_matrix(fullfile(fx, 'Xte.bin'));

X = cat(1, Xtr, Xte);
t0 = tic;
E = tabicl_col_embed(M.col, M.cfg, X, ytr);
R = tabicl_row_interact(M.row, M.cfg, E);
[logits, proba] = tabicl_icl(M.icl, M.cfg, R, ytr, numel(unique(ytr)));
elapsed = toc(t0);

e1 = max(max(abs(E(:, :, 1) - read_matrix(fullfile(fx, 'E1.bin')))));
er = max(max(abs(R - read_matrix(fullfile(fx, 'R.bin')))));
ref = read_matrix(fullfile(fx, 'logits.bin'));
el = max(max(abs(logits - ref)));
fprintf('%-12s: col-embed err %.2e | row-repr err %.2e | logit err %.2e (max |logit| %.2f) | %.2fs\n', ...
    name, e1, er, el, max(abs(ref(:))), elapsed);
ok = el < 1e-3;
if exist(fullfile(fx, 'proba.bin'), 'file') > 0
    pref = read_matrix(fullfile(fx, 'proba.bin'));
    ep = max(max(abs(proba - pref)));
    [~, i1] = max(proba, [], 2);
    [~, i2] = max(pref, [], 2);
    fprintf('%-12s: %d classes, hierarchical proba err %.2e | label agreement %5.1f%%\n', ...
        name, size(pref, 2), ep, 100 * mean(i1 == i2));
    ok = ok && ep < 1e-4 && all(i1 == i2);
end
end
