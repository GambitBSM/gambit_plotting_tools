from setuptools import setup, find_packages

setup(
    name="gambit_plotting_tools",
    version="1.0.0",
    description="A collection of plotting tools developed for use with GAMBIT and GAMBIT-light, see gambitbsm.org.",
    author="Anders Kvellestad",
    author_email="anders.kvellestad@fys.uio.no",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GambitBSM/gambit_plotting_tools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy >= 1.23.0",
        "matplotlib >= 3.5.2",
        "scipy >= 1.10.1",
        "h5py >= 3.9.0",
        "urllib",
        "pillow",
    ],
    include_package_data=True,
    license="BSD-3-Clause",
)
