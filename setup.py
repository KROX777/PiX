from setuptools import setup, find_packages

setup(
    name='pix',
    version='0.1.0',
    description='Physics discovery and PDE explorer',
    packages=find_packages(include=['pix', 'pix.*']),
    include_package_data=True,
    install_requires=[
        # dependencies can be copied from requirements.txt
    ],
    python_requires='>=3.8',
)
