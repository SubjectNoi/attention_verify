#pragma once
#include <torch/extension.h>

torch::Tensor my_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_cache,
    torch::Tensor v_cache
);