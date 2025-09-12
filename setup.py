from setuptools import setup, find_packages

setup(
    name="xlib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyrealsense2==2.54.1.5216",
        "numpy",
        "opencv-python",
        "matplotlib",
        "scipy",
        "ur-rtde",
    ],
    entry_points={
        "console_scripts": [],
    },
    url="https://github.com/MickyFlowers/xlib.git",
    author="Yuxi Chen",
    author_email="cyx010402@gmail.com",
    description="lib of x",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
