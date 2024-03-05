from setuptools import setup
from Cython.Build import cythonize
import os
os.environ['CC'] = 'clang-15'
os.environ['CFLAGS'] = '-O3 -march=native'
os.environ['LDSHARED'] = 'clang-15 -shared'


setup(
    name = "cython_vsh",
    ext_modules = cythonize(["*.py"],
                            compiler_directives={'language_level': '3'}),
    extra_compile_args=["-O3 -march=native -mtune=native"]
)
