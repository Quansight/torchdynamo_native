#include <llvm/Support/TargetSelect.h>

#include <tdnat/ops.h>

static tdnat::ATenOpRegistry &get_registry() {
  static tdnat::ATenOpRegistry registry;
  return registry;
}

namespace tdnat {

void global_register_aten_operations(ATenOpRegistry &registry);

void initialize_registry(ATenOpRegistry &registry) {
  global_register_aten_operations(registry);
}

void initialize_llvm() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();
}

void initialize() { initialize(get_registry()); }

void initialize(ATenOpRegistry &registry) {
  initialize_llvm();
  initialize_registry(registry);
}

} // namespace tdnat
