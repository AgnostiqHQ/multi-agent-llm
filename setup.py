from setuptools import find_packages, setup

setup(
    name="multi_agent_llm",
    version="0.1",
    description="LLM Multi-Agent implementation.",
    author="Agnostiq Inc",
    author_email="contact@agnostiq.ai",
    packages=find_packages(),
    install_requires=["openai>=1.46.0", "pydantic"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
