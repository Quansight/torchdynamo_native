#include <tdnat/llvm_function_builder.h>
#include <tdnat/ops.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace {

PYBIND11_MODULE(_C, m) {
  // Initialize global registry + LLVM components.
  tdnat::initialize();

  m.def("operation_in_registry", [](const std::string &opname) {
    return tdnat::get_aten_op(opname).has_value();
  });

  py::class_<tdnat::Value>(m, "Value")
      .def(py::init());

  py::class_<tdnat::LLVMFunctionBuilder>(m, "LLVMFunctionBuilder")
      .def(py::init<const std::string&, size_t, size_t>());
}

} // namespace
