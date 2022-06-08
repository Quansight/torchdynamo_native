#include <pybind11/pybind11.h>

namespace {

PYBIND11_MODULE(_C, m) {
  m.def("hello", []() { return "Hello World"; });
}

} // namespace
