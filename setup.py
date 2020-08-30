"""setup.py"""
from setuptools import setup, find_packages

with open("README.md") as f:
    README = f.read()

with open("LICENSE") as f:
    LICENSE = f.read()

with open("requirements.txt") as f:
    REQUIRES = f.read().splitlines()

setup(
    name="qa",
    version="0.1.0",
    description="qa library",
    long_description=README,
    author="sourcepirate",
    author_email="sathyanarrayanan@yandex.com",
    url="https://github.com/sourcepirate/qa.git",
    license=LICENSE,
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=REQUIRES,
    include_package_data=True,
    test_suite="tests",
)