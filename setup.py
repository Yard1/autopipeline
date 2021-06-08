from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        README = f.read()
    return README


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="autopipeline",
    version="0.1.0",
    description="AutoPipeline",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Yard1/autopipeline",
    author="Antoni Baum",
    author_email="antoni.baum@protonmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    install_requires=required,
)
