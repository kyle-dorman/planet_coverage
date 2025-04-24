from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Analyze planet coverage near the coastline.",
    author="kyle",
    license="LICENSE.txt",
)

# # Pinned bc of a jupyter notebook issue
# # https://github.com/microsoft/azuredatastudio/issues/24436
# traitlets==5.9.0
# # Install directly b/c latest version doesn't build on Mac
# triangle @ git+https://github.com/drufat/triangle.git
# gdal[numpy] == 3.10.0.*