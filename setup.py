from setuptools import setup, find_packages

setup(
    name="flexml",
    version="0.1.0",
    author="Ozgur Aslan",
    author_email="ozguraslank@gmail.com",
    description="Easy-to-use and flexible AutoML library for Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ozguraslank/flexml",
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    keywords=["AutoML", "Machine Learning", "Data Science", "Flexibility"],
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.3,<=1.26.4",
        "pandas>=2.0.1",
        "scikit-learn>=1.5.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=0.24.4",
        "tqdm>=4.60.0",
        "optuna>=3.0.0",
        "ipython>=7.11.0",
        "jinja2>=3.1.0"
    ],
    python_requires='>=3.9',
    include_package_data=False,  # Set to True if you want to include non-code files
)