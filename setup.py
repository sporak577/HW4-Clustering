from setuptools import setup, find_packages

setup(
    name="k_means clustering algorithm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy"
    ],
    author="Sophie-Christine Porak",
    author_email="sophie.porak@ucsf.edu",
    description="A K-Means clustering package with silhouette scoring",
    long_description=open("README.md").read()
    long_description_content_type="test/markdown",
    url="https://github.com/sporak577/HW4-Clustering.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.1",
    )