from setuptools import setup, find_packages
import sys

platform_specific_packages = {
    "darwin": ["tensorflow-macos", "tensorflow-metal"],
    "linux": ["tensorflow>=2.4"],
    "cywin": ["tensorflow>=2.4"],
    "win3D": ["tensorflow>=2.4"],
}

setup(
    name='tf_layers',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/tf_layers",
    licence="LGPL",
    python_requires='>=3.6',
    description="Some tensorflow layers",
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        *platform_specific_packages[sys.platform],
        "numpy"
    ],
)
