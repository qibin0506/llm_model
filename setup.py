from setuptools import setup, find_packages

setup(
    name='project_llm_model',
    version='0.8.1',
    description='LLM and VLM model',
    author='qibin',
    author_email='qibin0506@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
    ],
)
