from setuptools import setup, find_packages

setup(
    name='tf_layers',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/tf_layers",
    licence="LGPL",
    python_requires='>=3.6',
    description="Some tensorflow layers",
    version='1',
    packages=find_packages(),
    install_requires=[
        "tensorflow"
    ],
)
