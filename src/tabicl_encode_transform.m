function X = tabicl_encode_transform(enc, T)
%TABICL_ENCODE_TRANSFORM Encode a table with an encoder from TABICL_ENCODE_FIT.
%   Variables are looked up by name, so T may hold them in any order and
%   may contain extra ones. Categories not seen in training and missing
%   values both become -1.
if isempty(enc)
    error('tabicl:encode', 'The classifier was fitted on a matrix: pass a matrix.');
end
names = enc.names;
kind = enc.kind;
order = enc.order;
have = T.Properties.VariableNames;
n = height(T);   % size(T, 1) is 1 on RunMat 0.6.2
X = zeros(n, 0);
for i = 1:numel(order)
    k = order(i);
    name = char(names(k));
    if ~any(strcmp(have, name))
        error('tabicl:encode', 'The table has no variable ''%s''.', name);
    end
    v = T.(name);
    if kind(k) == 1
        if isnumeric(v)
            error('tabicl:encode', 'Variable ''%s'' was categorical in training.', name);
        end
        [s, miss] = tabicl_encode_text(v);
        u = enc.cats.(sprintf('v%d', k));
        code = -ones(n, 1);
        if ~isempty(u)
            [tf, loc] = ismember(s, u);
            hit = find(reshape(double(tf), n, 1) .* (1 - miss));
            loc = reshape(loc, n, 1);
            code(hit(:)') = loc(hit(:)') - 1;
        end
        X = cat(2, X, code);
    else
        X = cat(2, X, reshape(double(v), n, []));
    end
end
end
