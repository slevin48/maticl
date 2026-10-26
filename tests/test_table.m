function ok = test_table(M)
%TEST_TABLE Table input with categorical, string, cellstr and logical columns.
%   Compares the encoding and the probabilities with TabICLClassifier on a
%   pandas DataFrame (fixture clf_mixed). The training table carries the
%   target as variable 'label'; the test table lists its variables in
%   reverse order. On a runtime without tables, the matrix encoded by
%   Python is fed instead.
here = fileparts(which('read_matrix'));
fx = fullfile(here, 'fixtures', 'clf_mixed');
ytr = read_matrix(fullfile(fx, 'ytr.bin'));
yte = read_matrix(fullfile(fx, 'yte.bin'));
ref = read_matrix(fullfile(fx, 'proba.bin'));
Xtr_enc = read_matrix(fullfile(fx, 'Xtr_enc.bin'));
Xte_enc = read_matrix(fullfile(fx, 'Xte_enc.bin'));
ens = read_ensemble(fullfile(fx, 'ensemble.txt'));

t0 = tic;
enc_err = 0;
if exist('table') > 0 && exist('categorical') > 0
    mode = 'table';
    Ttr = make_table(fx, 'tr', ytr);
    Tte = make_table(fx, 'te', []);
    clf = tabicl_fit(Ttr, 'label', 'Model', M, 'Ensemble', ens);
    enc_err = max(enc_diff(tabicl_encode_transform(clf.encoder, Ttr), Xtr_enc), ...
        enc_diff(tabicl_encode_transform(clf.encoder, Tte), Xte_enc));
    [yhat, proba] = tabicl_predict(clf, Tte);
else
    mode = 'matrix';
    clf = tabicl_fit(Xtr_enc, ytr, 'Model', M, 'Ensemble', ens);
    [yhat, proba] = tabicl_predict(clf, Xte_enc);
end
elapsed = toc(t0);

perr = max(max(abs(proba - ref)));
classes = clf.classes;
[~, iref] = max(ref, [], 2);
agree = mean(classes(iref) == yhat);
fprintf('%-18s %-6s : encoding err %.1e | proba err %.1e | label agreement %5.1f%% | acc %.4f (python %.4f) | %.1fs\n', ...
    'clf_mixed', mode, enc_err, perr, 100 * agree, mean(yhat == yte), mean(classes(iref) == yte), elapsed);
ok = enc_err == 0 && perr < 1e-3 && agree == 1;
end

function e = enc_diff(X, Xref)
% Python imputes numeric NaNs inside its encoder, maticl just after it
idx = find(~isnan(X));
e = max(abs(X(idx) - Xref(idx)));
if size(X, 1) ~= size(Xref, 1) || size(X, 2) ~= size(Xref, 2)
    e = Inf;
end
end

function T = make_table(fx, part, y)
% Same variables as the pandas DataFrame: num, colour (categorical),
% cnt, size (cellstr), flag (logical), city (string)
num = read_matrix(fullfile(fx, ['num_' part '.bin']));
flag = read_matrix(fullfile(fx, ['flag_' part '.bin'])) > 0;
colour = categorical(read_lines(fullfile(fx, ['colour_' part '.txt'])));
sz = read_lines(fullfile(fx, ['size_' part '.txt']));
city = to_string(read_lines(fullfile(fx, ['city_' part '.txt'])));
x1 = num(:, 1);
x2 = num(:, 2);
if isempty(y)
    T = table(city, flag, sz, x2, colour, x1, ...
        'VariableNames', {'city', 'flag', 'size', 'cnt', 'colour', 'num'});
else
    T = table(x1, colour, x2, sz, flag, city, y, ...
        'VariableNames', {'num', 'colour', 'cnt', 'size', 'flag', 'city', 'label'});
end
end

function c = read_lines(path)
% one value per line, empty = missing
fid = fopen(path, 'r');
c = cell(0, 1);
line = fgetl(fid);
while ischar(line)
    c = cat(1, c, {line});
    line = fgetl(fid);
end
fclose(fid);
end

function s = to_string(c)
% cellstr -> string array with <missing> for empty entries
s = string(zeros(0, 1));
for i = 1:numel(c)
    w = char(c(i));
    if isempty(w)
        s = cat(1, s, string(missing));
    else
        s = cat(1, s, string(w));
    end
end
end
