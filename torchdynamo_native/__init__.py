import logging

logger = logging.getLogger(__name__)

try:
    import torch

    from torchdynamo_native._C import (
        Function,
        Value,
        operation_in_registry,
    )

    from torchdynamo_native.convert import (
        str_to_py,
        py_to_value,
    )

    from torchdynamo_native.schema import (
        AlignedArg,
        align_arguments,
        find_native_function,
    )

    from torchdynamo_native.compiler import (
        aot_llvmjit,
        llvmjit,
    )

    import torchdynamo_native.testing as testing

except Exception as e:
    import warnings
    import traceback

    warnings.warn(
        "failed when trying to import the full 'torchdynamo_native'. "
        "To show the error, set the logging level to DEBUG."
    )

    logger.debug("\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))

import torchdynamo_native.utils as utils
