import errno
import ntpath
import string
from typing import Union
import os
import random

file_mode: int = 511

default_dir: str = "/tmp/folders/wp"
default_prefix: str = "tmp"
default_suffix: str = ""

# Max tries for creating tmpdir
OS_TMP_MAX = os.TMP_MAX


def _get_name(seed: int = None, length: int = 8) -> str:
    """
    Retrieves a directory name str. To be extended with seeded random name.
    """

    # Get all letters and digits
    letters_and_digits = string.ascii_letters + string.digits

    if seed is None:
        print("No seed used.")
        return "".join(random.choices(letters_and_digits, k=length))
    else:
        if type(seed) is not int:
            raise ValueError("Seed must be of type: " + str(int))
        raise NotImplementedError("Seeded tempfile naming is not yet implemented")


def mkdtemp(suffix: str = None, prefix: str = None, dir: str = None) -> Union[bytes, str]:
    """
    Depending on which directory and which OS you create the temporary directory, Permission Errors might come up.
    Default is set to `default_dir`.

    Attributes
    __________
    suffix: str
            suffix of the tmp directory name. `default_suffix` is an empty string.
    prefix: str
        prefix of the tmp directory name. `default_prefix` is 'tmp'.
    dir: str
        Directory path where to create the temporary directory.

    Example:
        `mkdtemp(suffix="my_suffix", prefix="my_prefix", dir="/var/folders")` results to:
        "/var/folders/my_prefix{some_random_string}my_suffix"
    """
    if dir is None:
        dir = f"{default_dir}/{_get_name(length=16)}/T"
    if suffix is None:
        suffix = default_suffix
    if prefix is None:
        prefix = default_prefix

    for i in range(OS_TMP_MAX):
        file_name: str = _get_name()
        try:
            full_path: ntpath = os.path.join(dir, prefix + file_name + suffix)
            os.makedirs(name=full_path, mode=file_mode, exist_ok=False)

            return full_path
        except FileExistsError:
            # Continue with another random name which might not have been used yet.
            continue
        except PermissionError:
            raise TmpFilePermissionError(f"Permission Error while trying to recursively create directory in path: "
                                         f"{full_path}")


class TmpFilePermissionError(Exception):
    def __init__(self, message: str):
        super.__init__(message)
        self.error_code: int = errno.EACCES
