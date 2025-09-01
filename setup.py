from setuptools import setup, find_packages
# TODO: add Cython version setup
# from Cython.Build import cythonize
# import os
# os.environ['CC'] = 'clang'
# os.environ['CFLAGS'] = '-O3 -march=native'
# os.environ['LDSHARED'] = 'clang -shared'

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyVSH",
    version="0.1.0",
    author="Robin Geyer",
    # author_email="your.email@example.com",
    description="Spherical and Vector Spherial Harmonics (for Astronomy)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pinguinstall/pyVSH",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)

# setup(
#     name = "cython_vsh",
#     ext_modules = cythonize(["*.py"],
#                             compiler_directives={'language_level': '3'}),
#     extra_compile_args=["-O3 -march=native -mtune=native"]
# )
