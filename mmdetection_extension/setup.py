import os
from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="mmdet_extension",
        version=0.1,
        description=("mmdetection_extension"),
        author="windzu",
        author_email="windzu1@gmail.com",
        url="https://github.com/windzu/mmlab_extension",
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
        ],
        install_requires=[
            "rospkg",
            "future",
            "tensorboard",
        ],
        license="Apache License 2.0",
    )
