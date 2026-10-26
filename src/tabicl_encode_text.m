function [s, miss] = tabicl_encode_text(v)
%TABICL_ENCODE_TEXT Values and missing mask of a categorical-type column.
%   [s, miss] = TABICL_ENCODE_TEXT(v) with v an [n x 1] categorical, string,
%   cellstr or logical column. s is an [n x 1] string array and miss an
%   [n x 1] double mask (1 = missing): <undefined> categories, <missing>
%   strings and empty cellstr entries. Logical values become "false"/"true".
if iscategorical(v)
    [s, miss] = categorical_text(v);
    return;
end
n = size(v, 1);
if islogical(v)
    words = {'false', 'true'};
    s = reshape(string(words(double(v) + 1)), n, 1);
    miss = zeros(n, 1);
elseif isstring(v)
    s = reshape(v, n, 1);
    miss = reshape(double(ismissing(s)), n, 1);
else
    s = reshape(string(v), n, 1);
    miss = reshape(double(strlength(s) == 0), n, 1);
end
end

function [s, miss] = categorical_text(v)
ok = 0;
s = string(zeros(0, 1));
miss = zeros(0, 1);
try
    % MATLAB, Nelson: <undefined> becomes <missing>
    s = string(v);
    s = reshape(s, numel(s), 1);
    miss = reshape(double(ismissing(s)), numel(s), 1);
    ok = 1;
catch
end
if ok
    return;
end
% RunMat 0.6.2: categorical is a 1x1 object with no string conversion, but
% its Categories and Codes (NaN = undefined) properties are readable
cats = v.Categories;
codes = double(v.Codes);
n = numel(codes);
codes = reshape(codes, n, 1);
miss = double(isnan(codes));
s = string(zeros(n, 1));
if ~isempty(cats)
    codes(find(miss)) = 1;
    s = reshape(cats(codes), n, 1);
end
end
