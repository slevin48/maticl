function ok = test_forward(M)
%TEST_FORWARD Compare the raw MATLAB forward pass against PyTorch outputs.
here = fileparts(which('read_matrix'));
fx = fullfile(here, 'fixtures', 'forward');
Xtr = read_matrix(fullfile(fx, 'Xtr.bin'));
ytr = read_matrix(fullfile(fx, 'ytr.bin'));
Xte = read_matrix(fullfile(fx, 'Xte.bin'));

X = cat(1, Xtr, Xte);
t0 = tic;
E = tabicl_col_embed(M.col, M.cfg, X, ytr);
R = tabicl_row_interact(M.row, M.cfg, E);
logits = tabicl_icl(M.icl, M.cfg, R, ytr, numel(unique(ytr)));
elapsed = toc(t0);

e1 = max(max(abs(E(:, :, 1) - read_matrix(fullfile(fx, 'E1.bin')))));
er = max(max(abs(R - read_matrix(fullfile(fx, 'R.bin')))));
ref = read_matrix(fullfile(fx, 'logits.bin'));
el = max(max(abs(logits - ref)));
fprintf('forward: col-embed err %.2e | row-repr err %.2e | logit err %.2e (max |logit| %.2f) | %.2fs\n', ...
    e1, er, el, max(abs(ref(:))), elapsed);
ok = el < 1e-3;
end
