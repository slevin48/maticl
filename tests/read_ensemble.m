function ens = read_ensemble(path)
%READ_ENSEMBLE Read an ensemble.txt written by tools/make_reference.py.
% Lines look like "power | 2 0 3 1 | 2 0 1" (method | feature perm | class perm).
norms = {'none', 'power'};
fid = fopen(path, 'r');
idx = zeros(0, 1);
feat = [];
cls = [];
line = fgetl(fid);
while ischar(line)
    bars = strfind(line, '|');
    if numel(bars) == 2
        % char(codes(...)) instead of line(...): RunMat cannot index char arrays
        codes = double(line);
        idx = cat(1, idx, find(strcmp(norms, strtrim(char(codes(1:bars(1)-1))))));
        feat = cat(1, feat, reshape(sscanf(char(codes(bars(1)+1:bars(2)-1)), '%d'), 1, []));
        cls = cat(1, cls, reshape(sscanf(char(codes(bars(2)+1:end)), '%d'), 1, []));
    end
    line = fgetl(fid);
end
fclose(fid);
ens = struct();
ens.norms = norms;
ens.norm_idx = idx;
ens.feat = feat;
ens.cls = cls;
end
