import setuptools


setuptools.setup(
    name="neurockt",
    version="0.1.0",
    author="Tatsuya Masuko",
    description="Tiny fastai wrappers for deep learning everywhere.",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0",
    install_requires=[
        "torch >= 1.6.0",
        "pyyaml",
        "einops"
    ]
)