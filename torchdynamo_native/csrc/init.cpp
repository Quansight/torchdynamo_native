#include <tdnat/function.h>
#include <tdnat/ops.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

using namespace tdnat;

static Function function_init(const std::string id, size_t in_tensors,
                              size_t out_tensors) {
  return {{id, in_tensors, out_tensors}};
}

static std::vector<at::Tensor>
jitfunction_run(JITFunction &jitfn, const std::vector<at::Tensor> &tensors) {
  return jitfn.run(tensors);
}

namespace {

PYBIND11_MODULE(_C, m) {
  // Initialize global registry + LLVM components.
  initialize();

  m.def("operation_in_registry", [](const std::string &opname) {
    return get_aten_op(opname).has_value();
  });

  py::class_<Value>(m, "Value").def(py::init());

  py::class_<Function>(m, "Function")
      .def(py::init(&function_init))

      .def("set_placeholder", &Function::set_placeholder)
      .def("set_output", &Function::set_output)
      .def("set_outputs", &Function::set_outputs)

      .def("add_call", &Function::add_call)

      .def("dump", &Function::dump)
      .def("finalize", &Function::finalize)

      .def("build_bool", &Function::build_bool)
      .def("build_scalar_type", &Function::build_scalar_type)
      .def("build_optional_tensorlist", &Function::build_optional_tensorlist)
      .def("build_scalar_int", &Function::build_scalar)
      .def("build_integer", &Function::build_integer<int64_t>)

      .def("build_arrayref_int", &Function::build_arrayref<int64_t>)
      .def("build_arrayref_tensor", &Function::build_arrayref<at::Tensor>)
      .def("build_arrayref_lit_int", &Function::build_arrayref_lit<int64_t>)
      .def("build_arrayref_lit_tensor",
           &Function::build_arrayref_lit<at::Tensor>)

      .def("build_nullopt_tensor", &Function::build_nullopt<at::Tensor>)

      .def("build_optional_tensor", &Function::build_optional<at::Tensor>)
      .def("build_optional_lit_int", &Function::build_optional_lit<int64_t>)
      .def("build_optional_lit_scalar_type",
           &Function::build_optional_lit<at::ScalarType>)

      .def("into_jit", &Function::into_jit);

  py::class_<JITFunction>(m, "JITFunction").def("__call__", &jitfunction_run);
}

} // namespace
