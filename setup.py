from setuptools import find_packages, setup


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="Benchmarks",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
