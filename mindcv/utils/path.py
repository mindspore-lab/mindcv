#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pathlib

FILE_TYPE_ALIASES = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz")
}

ARCHIVE_TYPE_SUFFIX = [".tar", ".zip"]

COMPRESS_TYPE_SUFFIX = [".bz2", ".gz"]


def detect_file_type(filename: str):  # pylint: disable=inconsistent-return-statements
    """Detect file type by suffixes and return tuple(suffix, archive_type, compression)."""
    suffixes = pathlib.Path(filename).suffixes
    if not suffixes:
        raise RuntimeError(f"File `{filename}` has no suffixes that could be used to detect.")
    suffix = suffixes[-1]

    # Check if the suffix is a known alias.
    if suffix in FILE_TYPE_ALIASES:
        return suffix, FILE_TYPE_ALIASES[suffix][0], FILE_TYPE_ALIASES[suffix][1]

    # Check if the suffix is an archive type.
    if suffix in ARCHIVE_TYPE_SUFFIX:
        return suffix, suffix, None

    # Check if the suffix is a compression.
    if suffix in COMPRESS_TYPE_SUFFIX:
        # Check for suffix hierarchy.
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]
            # Check if the suffix2 is an archive type.
            if suffix2 in ARCHIVE_TYPE_SUFFIX:
                return suffix2 + suffix, suffix2, suffix
        return suffix, None, suffix
