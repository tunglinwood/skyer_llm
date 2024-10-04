from setuptools import setup, find_packages

setup(
    name="skyer_llm",
    version="0.1.0",
    author="tunglinwood",
    author_email="tomwu.tunglin@gmail.com",
    description="A simple package called skyer for building a Large Language Model from scratch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tunglinwood/skyer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
