from setuptools import find_packages, setup

setup(
    name="chainlite",
    version="0.3.6",
    author="Sina Semnani",
    author_email="sinaj@cs.stanford.edu",
    description="A Python package that uses LangChain and LiteLLM to call large language model APIs easily",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/stanford-oval/chainlite",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "langchain==0.3.4",
        "langchain-community==0.3.3",
        "langchain-core==0.3.12",
        "langchain-text-splitters==0.3.0",
        "langgraph==0.2.39",
        "langsmith==0.1.137",
        "langgraph-checkpoint==2.0.1",
        "langgraph-sdk==0.1.33",
        "litellm==1.57.1",
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
