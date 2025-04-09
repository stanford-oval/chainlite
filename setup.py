from setuptools import find_packages, setup

setup(
    name="chainlite",
    version="0.4.3",
    author="Sina Semnani",
    author_email="sinaj@cs.stanford.edu",
    description="A Python package that uses LangChain and LiteLLM to call large language model APIs easily",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/stanford-oval/chainlite",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "langchain>=0.3",
        "langchain-community>=0.3",
        "langgraph>=0.2",
        "litellm==1.65.4.post1",  # the unified interface to LLM APIs
        "numpydoc",  # needed for function calling with LiteLLM
        "grandalf",  # to visualize LangGraph graphs
        "pydantic>=2.5",
        "redis[hiredis]",
    ],
    extras_require={
        "dev": [
            "invoke",  # for running tasks and scripts
            "pytest",  # for testing
            "pytest-asyncio",  # for testing async code
            "setuptools",  # for building wheels
            "wheel",  # for building wheels
            "twine",  # for uploading to PyPI
            "isort",  # for code formatting
            "black",  # for code formatting
            "tuna",  # for measuring import time
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    license="Apache License 2.0",
)
