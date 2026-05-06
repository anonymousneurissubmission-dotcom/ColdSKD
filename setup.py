from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQ = ROOT / "requirements.txt"
README = ROOT / "README.md"

setup(
    name="coldskd",
    version="0.1.0",
    description="Cold Self-Distillation for OOD Detection in Label-Smoothed Models",
    long_description=README.read_text() if README.is_file() else "",
    long_description_content_type="text/markdown",
    author="Anonymous",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(),
    py_modules=["config"],
    install_requires=[
        line.strip() for line in REQ.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ] if REQ.is_file() else [],
    include_package_data=True,
)
