#! /bin/bash

THISDIR=$(dirname "$0")
PROJDIR=$(dirname "${THISDIR}")
BUILDDIR="${PROJDIR}/build"

tidy() {
    COMPCMDDIR="${1}"

    # Throw away the first parameter
    shift 1

    set -x
    clang-tidy -p "${COMPCMDDIR}" "$@"
    set +x
}


tidy "${BUILDDIR}" "${PROJDIR}/lib"/*.cpp
tidy "${BUILDDIR}" "${PROJDIR}/lib/tests"/*.cpp
tidy "${PROJDIR}" "${PROJDIR}/torchdynamo_native/csrc"/*.cpp
