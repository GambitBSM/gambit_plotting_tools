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
    entry_points = {
        'console_scripts': ['print_confidence_level_table=gambit_plotting_tools.print_confidence_level_table:main',
                            'print_dataset_names=gambit_plotting_tools.print_dataset_names:main',
                            'print_high_loglike_points=gambit_plotting_tools.print_high_loglike_points:main'
                            ]
    },
    include_package_data=True,
    package_data={
        'gambit_plotting_tools': ['gambit_logo_small.png'],
    },    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy >= 1.23.0",
        "matplotlib >= 3.8.0",
        "scipy >= 1.10.1",
        "h5py >= 3.9.0",
        "pillow",
        "tabulate",
    ],
    license="BSD-3-Clause",
)
