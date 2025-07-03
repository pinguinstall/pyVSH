from setuptools import setup
from Cython.Build import cythonize
import os
os.environ['CC'] = 'clang'
os.environ['CFLAGS'] = '-O3 -march=native'
os.environ['LDSHARED'] = 'clang -shared'


setup(
    name = "cython_vsh",
    ext_modules = cythonize(["*.py"],
                            compiler_directives={'language_level': '3'}),
    extra_compile_args=["-O3 -march=native -mtune=native"]
)
