import codecs, os, sys, re, setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)

    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="cmacpy",
    version=find_version("cmacpy/__init__.py"),
    author="John Armstrong",
    author_email="j.armstrong@strath.ac.uk",
    description="A Python package for general CMAC usage.",
    long_description=long_description,
    long_description_content_type="text/rst",
    url="https://github.com/bionictoucan/cmacpy",
    packages=["cmacpy"],
    install_requires = [
        "matplotlib >= 3.5.1",
        "numpy >= 1.22.2",
        "pandas >= 1.4.1",
        "scikit-image >= 0.18.3",
        "scipy >= 1.8.0",
        "sphinx >= 5.0.1",
        "sphinx-gallery >= 0.10.1",
        "sphinx-rtd-theme >= 1.0.0",
        "tqdm >= 4.62.3",
        "torch >= 1.12.0",
        "torchaudio >=0.10.2",
        "torchvision >= 0.11.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">3.8"
)