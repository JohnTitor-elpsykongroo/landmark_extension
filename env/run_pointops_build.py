"""Build the `pointops` CUDA extension with a Windows-locale workaround.

torch's ``cpp_extension`` queries ``cl`` for its version and decodes the output
with ``subprocess.SUBPROCESS_DECODE_ARGS`` (the OEM code page, cp936 here).
This machine's MSVC installation is language-pack'd and writes its banner as
UTF-8 bytes, so that decode raises ``UnicodeDecodeError`` before compilation
even starts. Patching the decode args to UTF-8 for this one build fixes it
without touching the repository and without changing the system locale.
"""

import subprocess
import sys
from pathlib import Path


subprocess.SUBPROCESS_DECODE_ARGS = ('utf-8',)

# torch.utils.cpp_extension imports the constant at module import time, so
# rebind it there as well.
import torch.utils.cpp_extension as _cpp_extension

_cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8',)

repo_root = Path(__file__).resolve().parents[2]

sys.argv = [str(repo_root / 'setup.py'), 'build_ext', '--inplace']
sys.path.insert(0, str(repo_root))

exec(compile((repo_root / 'setup.py').read_text(), str(repo_root / 'setup.py'), 'exec'))
