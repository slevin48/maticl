function M = tabicl_load(varargin)
%TABICL_LOAD Load exported TabICL weights into a MATLAB struct.
%   M = TABICL_LOAD(PREFIX) reads PREFIX.idx and PREFIX.bin as produced by
%   tools/export_weights.py. Without arguments, loads
%   weights/tabicl-classifier-v2 relative to the repository root.
%
%   Transformer blocks are stored as fields b1, b2, ... (e.g. M.icl.b12).
%   1-D tensors become row vectors and 2-D [out x in] PyTorch weights become
%   [in x out], so that y = x * W + b.
if numel(varargin) > 0
    prefix = char(varargin(1));
else
    here = fileparts(which('tabicl_load'));
    prefix = fullfile(fileparts(here), 'weights', 'tabicl-classifier-v2');
end

fidx = fopen([prefix '.idx'], 'r');
if fidx < 0
    error('tabicl:load', 'Cannot open %s.idx (run tools/export_weights.py first)', prefix);
end
fbin = fopen([prefix '.bin'], 'r');
if fbin < 0
    fclose(fidx);
    error('tabicl:load', 'Cannot open %s.bin', prefix);
end

% "config <key> <value>" lines come first
cfg = struct();
line = fgetl(fidx);
while ischar(line) && strncmp(line, 'config ', 7)
    [~, rest] = strtok(line);
    [name, rest] = strtok(rest);
    cfg.(name) = parse_value(strtrim(rest));
    line = fgetl(fidx);
end
if ~strcmp(cfg.col_feature_group, 'same') || cfg.col_affine || ~cfg.col_target_aware ...
        || cfg.row_rope_interleaved || ~cfg.norm_first || cfg.max_classes == 0
    fclose(fidx);
    fclose(fbin);
    error('tabicl:load', 'Only the TabICLv2 classifier configuration is supported.');
end

% Tensors follow in state_dict order and are read group by group (one
% transformer block at a time). Groups are read in small helper functions:
% RunMat's per-statement cost grows with the data held by the current frame.
col = struct();
[g, line] = read_group(fidx, fbin, line, 'col_embedder.in_linear.');
col.inW = g.weight;
col.inb = g.bias;
col.num_blocks = cfg.col_num_blocks;
for b = 1:cfg.col_num_blocks
    [g, line] = read_group(fidx, fbin, line, sprintf('col_embedder.tf_col.blocks.%d.', b - 1));
    col.(sprintf('b%d', b)) = make_isab(g, ~strcmp(cfg.col_ssmax, 'none'));
end
[g, line] = read_group(fidx, fbin, line, 'col_embedder.y_encoder.');
col.yW = g.weight;
col.yb = g.bias;

row = struct();
[g, line] = read_group(fidx, fbin, line, 'row_interactor.cls_tokens');
row.cls = transpose(g.value);
row.num_blocks = cfg.row_num_blocks;
for b = 1:cfg.row_num_blocks
    [g, line] = read_group(fidx, fbin, line, sprintf('row_interactor.tf_row.blocks.%d.', b - 1));
    row.(sprintf('b%d', b)) = make_mab(g, false);
end
[g, line] = read_group(fidx, fbin, line, 'row_interactor.tf_row.rope.');
row.freqs = g.freqs;
[g, line] = read_group(fidx, fbin, line, 'row_interactor.out_ln.');
row.lnw = g.weight;
row.lnb = g.bias;

icl = struct();
icl.num_blocks = cfg.icl_num_blocks;
for b = 1:cfg.icl_num_blocks
    [g, line] = read_group(fidx, fbin, line, sprintf('icl_predictor.tf_icl.blocks.%d.', b - 1));
    icl.(sprintf('b%d', b)) = make_mab(g, ~strcmp(cfg.icl_ssmax, 'none'));
end
[g, line] = read_group(fidx, fbin, line, 'icl_predictor.ln.');
icl.lnw = g.weight;
icl.lnb = g.bias;
[g, line] = read_group(fidx, fbin, line, 'icl_predictor.y_encoder.');
icl.yW = g.weight;
icl.yb = g.bias;
[g, line] = read_group(fidx, fbin, line, 'icl_predictor.decoder.');
icl.dW1 = g.x0_weight;
icl.db1 = g.x0_bias;
icl.dW2 = g.x2_weight;
icl.db2 = g.x2_bias;
fclose(fidx);
fclose(fbin);
if ischar(line)
    error('tabicl:load', 'Unexpected tensor in index: %s', line);
end

M = struct();
M.cfg = cfg;
M.col = col;
M.row = row;
M.icl = icl;
end

function [g, line] = read_group(fidx, fbin, line, prefix)
% Read the consecutive tensors whose name starts with PREFIX into a struct
% keyed by the rest of the name ('.' -> '_', leading digits prefixed by 'x',
% 'value' for an exact match). LINE is the next unread index line.
g = struct();
count = 0;
np = numel(prefix);
while ischar(line)
    [~, rest] = strtok(line);
    [name, rest] = strtok(rest);
    if ~strncmp(name, prefix, np)
        break;
    end
    v = sscanf(rest, '%d');
    dims = reshape(v(3:end), 1, numel(v) - 2);
    g.(field_name(regexprep(name, ['^' strrep(prefix, '.', '\.')], ''))) = read_tensor(fbin, dims, name);
    count = count + 1;
    line = fgetl(fidx);
end
if count == 0
    error('tabicl:load', 'Expected tensors starting with %s', prefix);
end
end

function key = field_name(rest)
if isempty(rest)
    key = 'value';
    return;
end
key = strrep(strrep(rest, 'multihead_attn', 'mab'), 'ssmax_layer', 'ss');
key = strrep(key, '.', '_');
key = regexprep(key, '^(\d)', 'x$1');
end

function t = read_tensor(fbin, dims, name)
n = prod(dims);
t = double(fread(fbin, n, 'single'));
if numel(t) ~= n
    error('tabicl:load', 'Weight file truncated at %s', name);
end
if numel(dims) == 1
    t = reshape(t, 1, n);
else
    t = reshape(t, fliplr(dims));
end
end

function blk = make_isab(g, has_ssmax)
% Induced self-attention block: inducing points + two attention blocks.
blk = struct();
blk.ind = transpose(g.ind_vectors);
blk.mab1 = make_mab(sub_group(g, 'mab1_'), has_ssmax);
blk.mab2 = make_mab(sub_group(g, 'mab2_'), false);
end

function s = sub_group(g, prefix)
s = struct();
names = fieldnames(g);
for i = 1:numel(names)
    name = char(names(i));
    if strncmp(name, prefix, numel(prefix))
        s.(regexprep(name, ['^' prefix], '')) = g.(name);
    end
end
end

function P = make_mab(g, has_ssmax)
P = struct();
P.n1w = g.norm1_weight;
P.n1b = g.norm1_bias;
P.n2w = g.norm2_weight;
P.n2b = g.norm2_bias;
P.W1 = g.linear1_weight;
P.b1 = g.linear1_bias;
P.W2 = g.linear2_weight;
P.b2 = g.linear2_bias;
W = g.attn_in_proj_weight;
bias = g.attn_in_proj_bias;
d = size(W, 1);
A = struct();
A.Wq = W(:, 1:d);
A.Wk = W(:, d+1:2*d);
A.Wv = W(:, 2*d+1:3*d);
A.bq = bias(1:d);
A.bk = bias(d+1:2*d);
A.bv = bias(2*d+1:3*d);
A.Wo = g.attn_out_proj_weight;
A.bo = g.attn_out_proj_bias;
A.has_ssmax = has_ssmax;
S = struct();
if has_ssmax
    S.bW1 = g.attn_ss_base_mlp_0_weight;
    S.bb1 = g.attn_ss_base_mlp_0_bias;
    S.bW2 = g.attn_ss_base_mlp_2_weight;
    S.bb2 = g.attn_ss_base_mlp_2_bias;
    S.qW1 = g.attn_ss_query_mlp_0_weight;
    S.qb1 = g.attn_ss_query_mlp_0_bias;
    S.qW2 = g.attn_ss_query_mlp_2_weight;
    S.qb2 = g.attn_ss_query_mlp_2_bias;
end
A.ssmax = S;
P.attn = A;
end

function v = parse_value(s)
if strcmp(s, 'True')
    v = true;
elseif strcmp(s, 'False')
    v = false;
else
    v = str2double(s);
    if isnan(v)
        v = s;
    end
end
end
