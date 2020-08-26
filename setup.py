from setuptools import setup

# version from https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
import re
VERSIONFILE="common/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")

setup(
    name="ensemble-anomaly-prediction",
    version=verstr,
    packages=['common', 'outliers', 'priors', 'explorer'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[],
    include_package_data=True,

    package_data={},

    # metadata to display on PyPI
    author="Nick Dowmon",
    author_email="ndowmon@gmail.com",
    description="Package used to detect anomalies in prior ",
    keywords="ensemble-prediction anomaly-detection",
    url="",   # project home page, if any
    project_urls={},
    classifiers=[]

)