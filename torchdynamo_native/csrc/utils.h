#pragma once

#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_dimname.h>

#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <string>

namespace tdnat {

enum class ArgType {
  List,
  Optional,

  // Copied from 'torchgen/model.py'.
  Base_Generator,
  Base_ScalarType,
  Base_Tensor,
  Base_int,
  Base_Dimname,
  Base_float,
  Base_str,
  Base_bool,
  Base_Layout,
  Base_Device,
  Base_Scalar,
  Base_MemoryFormat,
  Base_QScheme,
  Base_Storage,
  Base_Stream,
  Base_SymInt,
  Base_ConstQuantizerPtr,
};

inline bool check_base(const py::object &obj, ArgType type) {
  switch (type) {
  default:
    TORCH_CHECK(false, "bad ArgType: ", type);
  }
}

struct ArgTypeChecker {
  std::vector<ArgType> types_;

  ArgTypeChecker(std::initializer_list<ArgType> types) : types_(types) {}

  bool check(const py::object &obj, size_t i) {
    return check(obj, types_[i], i);
  }

  bool check(const py::object &obj, ArgType type, size_t i) {
    switch (type) {
    case ArgType::List:
      if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        return py::len(obj) == 0 ||
               std::all_of(obj.begin(), obj.end(), [this, i](auto &o) {
                 return check(o.template cast<py::object>(), i + 1);
               });
      } else {
        return false;
      }

    case ArgType::Optional:
      return obj.is_none() || check(obj, i + 1);

    case ArgType::Base_Generator:
      return THPGenerator_Check(obj.ptr());

    case ArgType::Base_ScalarType:
      return THPDtype_Check(obj.ptr());

    case ArgType::Base_Tensor:
      return THPVariable_Check(obj.ptr());

    case ArgType::Base_int:
      if (THPVariable_Check(obj.ptr())) {
        const auto &var = THPVariable_Unpack(obj.ptr());
        return at::isIntegralType(var.scalar_type(), /*includeBool=*/false) &&
               !var.requires_grad() && var.dim() == 0;

      } else {
        return THPUtils_checkLong(obj.ptr());
      }

    case ArgType::Base_Dimname:
      return THPUtils_checkDimname(obj.ptr());

    case ArgType::Base_float:
      if (THPVariable_Check(obj.ptr())) {
        const auto &var = THPVariable_Unpack(obj.ptr());
        return !var.requires_grad() && var.dim() == 0;
      } else {
        return THPUtils_checkDouble(obj.ptr());
      }

    case ArgType::Base_str:
      return py::isinstance<py::str>(obj);

    case ArgType::Base_bool:
      return py::isinstance<py::bool_>(obj);

    case ArgType::Base_Layout:
      return THPLayout_Check(obj.ptr());

    case ArgType::Base_Device:
      return THPDevice_Check(obj.ptr());

    case ArgType::Base_Scalar:
      return PyComplex_Check(obj.ptr()) || check(obj, ArgType::Base_float, i);

    case ArgType::Base_MemoryFormat:
      return THPMemoryFormat_Check(obj.ptr());

    case ArgType::Base_QScheme:
      return THPQScheme_Check(obj.ptr());

    case ArgType::Base_Stream:
      return THPStream_Check(obj.ptr());

    case ArgType::Base_SymInt:
      return check(obj, ArgType::Base_int, i);

    case ArgType::Base_Storage:
      // Need 'isStorage' function which is not linkable.
    case ArgType::Base_ConstQuantizerPtr:
      // TODO: check for this case.
      return true;
    }
  }
};

struct ATenOpArg {
  std::string name_;
  c10::optional<c10::IValue> default_;
  ArgTypeChecker checker_;
  bool is_kwarg_;
  size_t position_;

  ATenOpArg(std::string name, ArgTypeChecker checker, size_t position,
            c10::optional<c10::IValue> def)
      : name_(name), default_(def), checker_(checker), is_kwarg_(false),
        position_(position) {}

  ATenOpArg(std::string name, ArgTypeChecker checker,
            c10::optional<c10::IValue> def)
      : name_(name), default_(def), checker_(checker), is_kwarg_(true),
        position_(0) {}

  bool in_kwargs(const py::kwargs &kwargs) {
    return kwargs.contains(name_) && checker_.check(kwargs[name_.data()], 0);
  }

  bool in_args(const py::args &args) {
    return args.size() > position_ && checker_.check(args[position_], 0);
  }

  bool check(const py::args &args, const py::kwargs &kwargs) {
    if (is_kwarg_) {
      return default_.has_value() || in_kwargs(kwargs);
    } else {
      return default_.has_value() || in_args(args) || in_kwargs(kwargs);
    }
  }
};

struct ATenOp {
  virtual std::vector<ATenOpArg> &arguments() = 0;

  bool check(const py::args &args, const py::kwargs &kwargs) {
    auto &args_info = arguments();
    return std::all_of(args_info.begin(), args_info.end(),
                       [&](auto &info) { return info.check(args, kwargs); });
  }
};

} // namespace tdnat
