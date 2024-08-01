#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <torch/extension.h>

__global__ void my_attention_kernel(
    half* _o,
    half* _q,
    half* _k,
    half* _v,
    half* _k_cache,
    half* _v_cache,
    int _batch,
    int _tokens
)
{

}

torch::Tensor my_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_cache,
    torch::Tensor v_cache
)
{
    auto batch = q.size(0);
    auto hidden_dim = q.size(2);
    auto tokens = k_cache.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({batch, hidden_dim}, 0, options);

    half* q_ptr = reinterpret_cast<half*>(q.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(k.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(v.data_ptr<at::Half>());
    half* k_cache_ptr = reinterpret_cast<half*>(k_cache.data_ptr<at::Half>());
    half* v_cache_ptr = reinterpret_cast<half*>(v_cache.data_ptr<at::Half>());

    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    // Invoke your kernel:
    // my_attention_kernel <<<?, ?>>> (o_ptr, q_ptr, k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, batch, tokens, ...);

    return o;

}