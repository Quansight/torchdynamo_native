// generated from: ${generator_file}

#include <torchdynamo_native/csrc/ops.h>
#include <ATen/core/TensorBody.h>

${aten_ops_include}

namespace tdnat {

ATenOpRegistry GlobalRegistry {
    ${aten_ops_entry}
};

} // namespace tdnat
