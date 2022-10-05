#include <tdnat/function.h>
#include <tdnat/ops.h>

#include <c10/util/Exception.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/MemoryFormat.h>
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

template <>
struct type_caster<at::MemoryFormat> {
public:
  // NOLINTNEXTLINE
  PYBIND11_TYPE_CASTER(at::MemoryFormat, _("at::MemoryFormat"));

  bool load(handle src, bool)
  {
    PyObject *source = src.ptr();

    if (THPMemoryFormat_Check(source)) {
      auto mf = reinterpret_cast<THPMemoryFormat *>(source);
      value = mf->memory_format;
      return true;
    } else {
      return false;
    }
  }

  static handle cast(at::ScalarType src, return_value_policy /* policy */, handle /* parent */)
  {
    TORCH_CHECK(false, "pybind11 casting not implemented for at::MemoryFormat.");
  }
};

} // namespace detail
} // namespace pybind11

// Wrapper around nat::Function constructor.
static Function function_init(const std::string &id, size_t in_tensors, size_t out_tensors)
{
  return {{id, in_tensors, out_tensors}};
}

// Wrapper around nat::JITFunction::run.
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

      // Input & Output.
      .def("set_placeholder", &Function::set_placeholder)
      .def("set_output", &Function::set_output)
      .def("set_outputs", &Function::set_outputs)

      // Operator call.
      .def("add_call", &Function::add_call)

      // Post-construction.
      .def("dump", &Function::dump)
      .def("finalize", &Function::finalize)

      .def("build_bool", &Function::build_bool)
      .def("build_optional_tensorlist", &Function::build_optional_tensorlist)
      .def("build_scalar_type", &Function::build_scalar_type)
      .def("build_memory_format", &Function::build_memory_format)
      .def("build_vector_at_tensor", &Function::build_vector_at_tensor)

      .def("build_scalar_int", &Function::build_scalar_int)
      .def("build_scalar_float", &Function::build_scalar_float)

      // Build: scalar types.
      .def("build_int", &Function::build_int<int64_t>)
      .def("build_float", &Function::build_float<double>)

      // Build: ArrayRef<T> for in-memory T.
      .def("build_arrayref_int", &Function::build_arrayref<int64_t>)
      .def("build_arrayref_tensor", &Function::build_arrayref<at::Tensor>)

      // Build: ArrayRef<T> for literal T.
      .def("build_arrayref_lit_int", &Function::build_arrayref_lit<int64_t>)
      .def("build_arrayref_lit_tensor", &Function::build_arrayref_lit<at::Tensor>)

      // Build: c10::nullopt.
      .def("build_nullopt_bool", &Function::build_nullopt<bool>)
      .def("build_nullopt_int", &Function::build_nullopt<int64_t>)
      .def("build_nullopt_float", &Function::build_nullopt<double>)
      .def("build_nullopt_str", &Function::build_nullopt<c10::string_view>)
      .def("build_nullopt_scalar_type", &Function::build_nullopt<at::ScalarType>)
      .def("build_nullopt_memory_format", &Function::build_nullopt<at::MemoryFormat>)
      .def("build_nullopt_layout", &Function::build_nullopt<at::Layout>)
      .def("build_nullopt_layout", &Function::build_nullopt<at::Device>)
      .def("build_nullopt_generator", &Function::build_nullopt<at::Generator>)
      .def("build_nullopt_tensor", &Function::build_nullopt<at::Tensor>)

      // Build: c10::optional<T> for in-memory T.
      .def("build_optional_tensor", &Function::build_optional<at::Tensor>)
      .def("build_optional_arrayref_int", &Function::build_optional<at::IntArrayRef>)

      // Build: c10::optional<T> for literal T.
      .def("build_optional_lit_int", &Function::build_optional_lit<int64_t>)
      .def("build_optional_lit_float", &Function::build_optional_lit<double>)
      .def("build_optional_lit_scalar_type", &Function::build_optional_lit<at::ScalarType>)
      .def("build_optional_lit_memory_format", &Function::build_optional_lit<at::MemoryFormat>)

      // Transform Function into JITFunction.
      .def("into_jit", &Function::into_jit);

  py::class_<JITFunction>(m, "JITFunction")
      .def("run", &jitfunction_run)
      .def("__call__", &jitfunction_run);
}

} // namespace
