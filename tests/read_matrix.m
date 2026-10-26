function a = read_matrix(path)
%READ_MATRIX Read a matrix written by tools/make_reference.py.
fid = fopen(path, 'r');
if fid < 0
    error('read_matrix:open', 'Cannot open %s', path);
end
sz = fread(fid, 2, 'int32');
a = fread(fid, sz(1) * sz(2), 'double');
fclose(fid);
a = reshape(a, sz(1), sz(2));
end
