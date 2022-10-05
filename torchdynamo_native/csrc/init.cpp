#include <tdnat/function.h>
#include <tdnat/ops.h>

#include <c10/util/Exception.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/pybind11.h>

using namespace tdnat;

namespace pybind11
{
namespace detail
{

template <>
struct type_caster<at::ScalarType> {
public:
  // NOLINTNEXTLINE
  PYBIND11_TYPE_CASTER(at::ScalarType, _("at::ScalarType"));

  bool load(handle src, bool)
  {
    PyObject *source = src.ptr();

    if (!THPDtype_Check(source) && !THPPythonScalarType_Check(source)) {
      return false;
    }

    if (source == reinterpret_cast<PyObject *>(&PyFloat_Type)) {
      value = at::ScalarType::Double;
    } else if (source == reinterpret_cast<PyObject *>(&PyBool_Type)) {
      value = at::ScalarType::Bool;
    } else if (source == reinterpret_cast<PyObject *>(&PyLong_Type)) {
      value = at::ScalarType::Long;
    } else {
      auto dtype = reinterpret_cast<THPDtype *>(source);
      value = dtype->scalar_type;
    }

    return true;
  }

  static handle cast(at::ScalarType src, return_value_policy /* policy */, handle /* parent */)
  {
    TORCH_CHECK(false, "pybind11 casting not implemented for at::ScalarType.");
  }
};

} // namespace detail
} // namespace pybind11

static Function function_init(const std::string &id, size_t in_tensors, size_t out_tensors)
{
  return {{id, in_tensors, out_tensors}};
}

static std::vector<at::Tensor>
jitfunction_run(JITFunction &jitfn, const std::vector<at::Tensor> &tensors)
{
  return jitfn.run(tensors);
}

namespace
{

// NOLINTNEXTLINE
PYBIND11_MODULE(_C, m)
{
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
      .def("build_optional_tensorlist", &Function::build_optional_tensorlist)
      .def("build_scalar_type", &Function::build_scalar_type)
      .def("build_scalar_int", &Function::build_scalar)
      .def("build_vector_at_tensor", &Function::build_vector_at_tensor)

      .def("build_int", &Function::build_int<int64_t>)

      .def("build_arrayref_int", &Function::build_arrayref<int64_t>)
      .def("build_arrayref_tensor", &Function::build_arrayref<at::Tensor>)
      .def("build_arrayref_lit_int", &Function::build_arrayref_lit<int64_t>)
      .def("build_arrayref_lit_tensor", &Function::build_arrayref_lit<at::Tensor>)

      .def("build_nullopt_tensor", &Function::build_nullopt<at::Tensor>)

      .def("build_optional_tensor", &Function::build_optional<at::Tensor>)
      .def("build_optional_lit_int", &Function::build_optional_lit<int64_t>)
      .def("build_optional_lit_scalar_type", &Function::build_optional_lit<at::ScalarType>)

      .def("into_jit", &Function::into_jit);

  py::class_<JITFunction>(m, "JITFunction")
      .def("run", &jitfunction_run)
      .def("__call__", &jitfunction_run);
}

} // namespace
