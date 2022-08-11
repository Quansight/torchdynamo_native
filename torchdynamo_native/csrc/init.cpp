#include <tdnat/function.h>
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

  py::class_<tdnat::FunctionData>(m, "FunctionData")
      .def(py::init<const std::string&, size_t, size_t>());

  py::class_<tdnat::Function>(m, "Function")
      .def(py::init<const tdnat::FunctionData&>());
}

} // namespace
