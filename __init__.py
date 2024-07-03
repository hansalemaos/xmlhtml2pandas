from __future__ import annotations
import re
import numpy as np
import platform
import subprocess
import os
import sys
import time
from typing import Union

try:
    from .xmlhtml import *

except Exception as e:
    import Cython, setuptools, flatten_any_dict_iterable_or_whatsoever, lxml, pandas, numpy, exceptdrucker

    iswindows = "win" in platform.platform().lower()
    if iswindows:
        addtolist = []
    else:
        addtolist = ["&"]

    olddict = os.getcwd()
    dirname = os.path.dirname(__file__)
    os.chdir(dirname)
    compile_file = os.path.join(dirname, "xmlhtml_compile.py")
    subprocess.run(
        " ".join([sys.executable, compile_file, "build_ext", "--inplace"] + addtolist),
        shell=True,
        env=os.environ,
        preexec_fn=None if iswindows else os.setpgrp,
    )
    if not iswindows:
        time.sleep(30)
    from .xmlhtml import *

    os.chdir(olddict)
