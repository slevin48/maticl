function y = tabicl_mab(P, x, kv, B, nh, rope_freqs)
%TABICL_MAB Pre-norm transformer block (attention + GELU feed-forward).
%   x  : [Sq*B x d] queries (also the residual stream)
%   kv : [Sk*B x d] keys/values, or [] for self-attention on x
qn = tabicl_layernorm(x, P.n1w, P.n1b);
if isempty(kv)
    kvn = qn;
else
    kvn = tabicl_layernorm(kv, P.n1w, P.n1b);
end
y = x + tabicl_mha(P.attn, qn, kvn, B, nh, rope_freqs);
h = tabicl_layernorm(y, P.n2w, P.n2b);
y = y + tabicl_gelu(h * P.W1 + P.b1) * P.W2 + P.b2;
end
