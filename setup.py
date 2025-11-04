from setuptools import setup, find_packages

setup(
    name="real_estate_valuation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "scikit-learn",
        "mlflow",
        "pytest",
        "httpx"
    ]
)