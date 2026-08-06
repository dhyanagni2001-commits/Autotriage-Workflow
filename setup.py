from setuptools import find_packages, setup

setup(
    name="autotriage",
    version="0.1.0",
    packages=find_packages(include=["agent", "agent.*", "classifier", "classifier.*",
                                     "retriever", "retriever.*", "eval", "eval.*",
                                     "data", "data.*"]),
    python_requires=">=3.10",
)
