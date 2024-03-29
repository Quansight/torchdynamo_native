#include <tdnat/function.h>

#include <c10/util/Exception.h>

#include <llvm/Support/Error.h>

using namespace tdnat;

JITFunction::JITFunction(llvm::orc::LLJIT *jit, FunctionData data) :
    jit_(jit),
    data_(std::move(data))
{
}

std::vector<at::Tensor *> JITFunction::run(at::ArrayRef<at::Tensor> in_tensors)
{
  std::vector<at::Tensor *> out_tensors(data_.out_tensors_);
  run_out(in_tensors, out_tensors);
  return out_tensors;
}

void JITFunction::run_out(
    at::ArrayRef<at::Tensor> in_tensors,
    at::ArrayRef<at::Tensor *> out_tensors
)
{

  TORCH_CHECK(
      in_tensors.size() == data_.in_tensors_,
      "Expected number of inputs mismatch actual number of inputs: ",
      data_.in_tensors_,
      " != ",
      in_tensors.size()
  );
  TORCH_CHECK(
      out_tensors.size() == data_.out_tensors_,
      "Expected number of outputs mismatch actual number of outputs: ",
      data_.out_tensors_,
      " != ",
      out_tensors.size()
  );

  if (cache_ == nullptr) {
    auto symbol = llvm::cantFail(jit_->lookup(data_.id_));

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    cache_ = reinterpret_cast<RunFnType>(symbol.getAddress());
  }

  cache_(in_tensors.data(), out_tensors.data());
}
