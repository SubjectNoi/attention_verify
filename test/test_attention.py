import torch
import numpy as np
from MyAttn import my_attention

dev = "cuda:0"

# Assuming 32 heads, 128 dims / head
HEAD = 32
HEAD_DIM = 128
BATCH = 8
TOKENS = 1024
HIDDEN = HEAD * HEAD_DIM

q = torch.randn((BATCH, 1, HIDDEN)).type(torch.float16).to(dev)
k = torch.randn((BATCH, 1, HIDDEN)).type(torch.float16).to(dev)
v = torch.randn((BATCH, 1, HIDDEN)).type(torch.float16).to(dev)
k_cache = torch.randn((BATCH, TOKENS, HIDDEN)).type(torch.float16).to(dev)
v_cache = torch.randn((BATCH, TOKENS, HIDDEN)).type(torch.float16).to(dev)

o = my_attention(q, k, v, k_cache, v_cache)

# Referece attention:
print(k_cache.shape, k.shape)
_k = torch.cat((k_cache, k), dim=1)
_k = torch.transpose(_k, 1, 2)
attn_weight = torch.bmm(q, _k) / np.sqrt(HEAD_DIM)
attn_weight = torch.softmax(attn_weight, dim=1)
_v = torch.cat((v_cache, v), dim=1)
o_reference = torch.bmm(attn_weight, _v)

print(o - o_reference) # Should be small.
