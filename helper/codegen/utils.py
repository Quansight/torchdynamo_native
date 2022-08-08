import os
import torchgen

def get_packaged_yaml_path() -> str:
    torchgen_init = torchgen.__file__
    torchgen_dir = os.path.dirname(torchgen_init)
    return os.path.join(torchgen_dir, "packaged", "ATen", "native")

def get_native_functions_yaml_path() -> str:
    return os.path.join(get_packaged_yaml_path(), "native_functions.yaml")

def get_tags_yaml_path() -> str:
    return os.path.join(get_packaged_yaml_path(), "tags.yaml")
