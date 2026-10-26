function out = tabicl_mha(P, xq, xkv, B, nh, rope_freqs)
%TABICL_MHA Multi-head scaled dot-product attention.
%   xq  : [Sq*B x d] queries, B groups of Sq consecutive tokens
%   xkv : [Sk*B x d] keys/values, B groups of Sk consecutive tokens
%   P   : projection weights (Wq, Wk, Wv, bq, bk, bv, Wo, bo) and optional
%         query-aware scalable softmax parameters (P.ssmax, if P.has_ssmax)
%   rope_freqs : [1 x hd/2] rotary frequencies, or [] for no RoPE
d = size(xq, 2);
hd = d / nh;
Sq = size(xq, 1) / B;
Sk = size(xkv, 1) / B;

q = xq * P.Wq + P.bq;
k = xkv * P.Wk + P.bk;
v = xkv * P.Wv + P.bv;

if ~isempty(rope_freqs)
    q = apply_rope(q, Sq, B, nh, rope_freqs);
    k = apply_rope(k, Sk, B, nh, rope_freqs);
end
if P.has_ssmax
    q = apply_qassmax(q, P.ssmax, Sk, nh);
end

% [S*B x hd*nh] -> [S x hd x B*nh] pages, one page per (group, head)
q3 = reshape(permute(reshape(q, Sq, B, hd, nh), [1 3 2 4]), Sq, hd, B * nh);
k3 = reshape(permute(reshape(k, Sk, B, hd, nh), [3 1 2 4]), hd, Sk, B * nh);
v3 = reshape(permute(reshape(v, Sk, B, hd, nh), [1 3 2 4]), Sk, hd, B * nh);

s = tabicl_bmm(q3, k3) / sqrt(hd);
p = tabicl_softmax(s, 2);
o = tabicl_bmm(p, v3);

o = reshape(permute(reshape(o, Sq, hd, B, nh), [1 3 2 4]), Sq * B, d);
out = o * P.Wo + P.bo;
end

function x = apply_rope(x, S, B, nh, freqs)
% Non-interleaved rotary embedding: each head is split into halves [x1, x2]
% and rotated as [x1*cos - x2*sin, x2*cos + x1*sin], position = token index.
half = numel(freqs);
hd = 2 * half;
pos = repmat((0:S-1)', B, 1);
ang = pos * freqs;
c = cos(ang);
s = sin(ang);
for h = 1:nh
    i1 = (h - 1) * hd + (1:half);
    i2 = i1 + half;
    x1 = x(:, i1);
    x2 = x(:, i2);
    x(:, i1) = x1 .* c - x2 .* s;
    x(:, i2) = x2 .* c + x1 .* s;
end
end

function q = apply_qassmax(q, S, n, nh)
% Query-aware scalable softmax (elementwise variant):
%   q <- q .* base_mlp(log n) .* (1 + tanh(query_mlp(q)))
hd = size(q, 2) / nh;
logn = log(max(n, 1));
base = tabicl_gelu(logn * S.bW1 + S.bb1) * S.bW2 + S.bb2;
for h = 1:nh
    idx = (h - 1) * hd + (1:hd);
    qh = q(:, idx);
    modulation = 1 + tanh(tabicl_gelu(qh * S.qW1 + S.qb1) * S.qW2 + S.qb2);
    q(:, idx) = qh .* base(idx) .* modulation;
end
end
