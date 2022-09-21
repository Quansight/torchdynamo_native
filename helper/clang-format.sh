#! /bin/bash

THISDIR=$(dirname "$0")
PROJDIR=$(dirname "${THISDIR}")

SOURCE_DIRS=(
    "lib"
    "lib/tests"
    "torchdynamo_native/csrc"
)

format() {
    FILE="${1}"

    set -x
    clang-format -i "${FILE}"
    set +x
}

for dir in "${SOURCE_DIRS[@]}"; do
    DIRPATH="${PROJDIR}/${dir}"

    for f in "${DIRPATH}"/*.cpp; do
        format "${f}"
    done
done

INCLUDE_DIRPATH="${PROJDIR}/include/tdnat"
for f in "${INCLUDE_DIRPATH}"/*.h; do
    format "${f}"
done
