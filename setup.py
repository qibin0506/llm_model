from setuptools import setup, find_packages

# 1. python3 setup.py sdist bdist_wheel
# 2. pip3 install dist/project_llama-0.1.tar.gz
setup(name='project_llama',
    version='0.1',
    description='llama model',
    author='qibin',
    author_email='qibin0506@gmail.com',
    packages=find_packages(),)
