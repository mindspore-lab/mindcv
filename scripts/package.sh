#!/bin/bash

set -e

BASE_PATH=$(cd "$(dirname $0)/.."; pwd)
OUTPUT_PATH="${BASE_PATH}/output"


if [[ -d "${OUTPUT_PATH}" ]]; then
    rm -rf "${OUTPUT_PATH}"
fi
mkdir -pv "${OUTPUT_PATH}"

python ${BASE_PATH}/setup.py bdist_wheel

mv ${BASE_PATH}/dist/*whl ${OUTPUT_PATH}

cd "${OUTPUT_PATH}" || exit
PACKAGE_LIST=$(ls mindcv-*.whl) || exit
for PACKAGE_NAME in ${PACKAGE_LIST}; do
    echo "writing sha256sum of ${PACKAGE_NAME}"
    sha256sum -b "${PACKAGE_NAME}" > "${PACKAGE_NAME}.sha256"
done
