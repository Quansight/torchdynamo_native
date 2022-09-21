// generated from: ${generator_file}

#include <ATen/core/TensorBody.h>
#include <tdnat/ops_util.h>

${ops_include}

namespace tdnat
{

void ${register_function_prefix}${shard_id}(${register_function_parameters})
{
  ${ops_entry}
}

} // namespace tdnat
