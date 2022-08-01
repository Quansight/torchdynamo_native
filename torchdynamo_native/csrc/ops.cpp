#include <torchdynamo_native/csrc/ops.h>

namespace tdnat {

extern ATenOpRegistry GlobalRegistry;

c10::optional<ATenOpRef> get_aten_op(const std::string &opname) {
  if (GlobalRegistry.find(opname) == GlobalRegistry.end()) {
    return c10::nullopt;
  }
  return GlobalRegistry[opname];
}

} // namespace tdnat
