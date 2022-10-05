import os
import torchgen

from torchgen.gen import ParsedYaml, parse_native_yaml


def parse_native_functions_yaml() -> ParsedYaml:
    # Torchgen base file.
    torchgen_init = torchgen.__file__
    torchgen_dir = os.path.dirname(torchgen_init)

    # Packaged files directory.
    packaged_dir = os.path.join(torchgen_dir, "packaged", "ATen", "native")

    # Path to YAML files.
    native_functions_yaml_path = os.path.join(packaged_dir, "native_functions.yaml")
    tags_yaml_path = os.path.join(packaged_dir, "tags.yaml")

    return parse_native_yaml(native_functions_yaml_path, tags_yaml_path)
