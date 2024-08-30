from setuptools import setup, find_packages

def parse_requirements(filename: str):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="flexml",
    version="1.0.2",
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
    install_requires=parse_requirements("requirements.txt"),
    python_requires='>=3.9',
    include_package_data=False,  # Set to True if you want to include non-code files
)