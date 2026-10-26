function [X, y, enc] = tabicl_encode_fit(T, y)
%TABICL_ENCODE_FIT Fit the column encoder of a table and encode it.
%   [X, y, enc] = TABICL_ENCODE_FIT(T, y) ports TransformToNumerical of the
%   Python package. Categorical, string, cellstr and logical variables are
%   ordinal-encoded (sorted distinct values -> 0, 1, ...; missing -> -1),
%   numeric variables pass through (NaN kept, imputed later by TABICL_FIT).
%   As in Python, the encoded categorical columns come first, then the
%   numeric ones, each group in table order. Other variable types are
%   dropped with a warning.
%
%   y may name a variable of T, which is then used as the target and left
%   out of X. A categorical y is turned into strings, so that its classes
%   are sorted like Python's LabelEncoder.
names = T.Properties.VariableNames;
nv = numel(names);
ycol = 0;
if ischar(y) || (isstring(y) && numel(y) == 1)
    hit = find(strcmp(names, char(y)));
    if isempty(hit)
        error('tabicl:encode', 'y names no variable of the table: ''%s''.', char(y));
    end
    ycol = hit(1);
    y = T.(char(y));
end
if iscategorical(y)
    [y, ymiss] = tabicl_encode_text(y);
    if any(ymiss)
        error('tabicl:encode', 'y has undefined values.');
    end
end

kind = zeros(1, nv);   % 1 = categorical, 2 = numeric, 0 = not used
cats = struct();
for k = 1:nv
    % (no `continue`: RunMat 0.6.2 mishandles it here)
    if k ~= ycol
        [kk, cats] = fit_variable(T, char(names(k)), k, cats);
        kind(k) = kk;
    end
end
if ~any(kind)
    error('tabicl:encode', 'The table has no usable variable.');
end

enc = struct();
enc.names = names;
enc.kind = kind;
enc.cats = cats;
enc.order = cat(2, find(kind == 1), find(kind == 2));
X = tabicl_encode_transform(enc, T);
end

function [kind, cats] = fit_variable(T, name, k, cats)
v = T.(name);
kind = 0;
if isnumeric(v)
    kind = 2;
elseif islogical(v) || isstring(v) || iscellstr(v) || iscategorical(v)
    if size(v, 2) ~= 1
        error('tabicl:encode', 'Variable ''%s'' must have a single column.', name);
    end
    kind = 1;
    [s, miss] = tabicl_encode_text(v);
    present = find(miss == 0);
    u = unique(s(present(:)'));
    u = reshape(u, numel(u), 1);
    cats.(sprintf('v%d', k)) = u;
    if numel(u) > 40
        warning('tabicl:encode', ['Variable ''%s'' has %d categories; ordinal encoding ' ...
            'may not suit high-cardinality columns.'], name, numel(u));
    end
else
    warning('tabicl:encode', 'Variable ''%s'' (%s) is not used.', name, class(v));
end
end
