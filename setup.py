"""
@Project  : dichotomous-score
@File     : setup.py
@Author   : Shaobo Cui
@Date     : 17.10.2024 14:16
"""

# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

def get_version():
    with open("oppositescore/__init__.py", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Version not found")

# Read the README file for the long description
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements file
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    include_package_data=True,
    name='opposite-score',
    version=get_version(),
    description='Optimized Text Embeddings Designed for Measuring Opposite/Contrasting Relationships',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Shaobo Cui',
    author_email='shaobo.cui@epfl.ch',
    url='https://github.com/cui-shaobo/conditional-dichotomy-quantification',
    packages=find_packages(include=['oppositescore', 'oppositescore.*']),
    install_requires=requirements,
    zip_safe=False,
    keywords='text embedding, NLP, opposite relationships, dichotomy',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7, <3.12',
)