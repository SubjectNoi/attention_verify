#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "my_attention/my_attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_attention", &my_attention, "[Description]");
}
