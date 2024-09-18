from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent

# Read version from VERSION file
version = (this_directory / "VERSION").read_text().strip()

# Read text from README.md
long_description = (this_directory / "README.md").read_text()

# Read list of requirements
install_requires = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="multi_agent_llm",
    version=version,
    description="An LLM Multi-Agent framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Agnostiq Inc.",
    author_email="contact@agnostiq.ai",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
