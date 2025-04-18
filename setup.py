from setuptools import find_packages, setup

package_name = "parking_spot"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "rclpy",
        "sensor_msgs",
        "cv_bridge",
        "scikit-learn",
        "numpy<2",
        "more-itertools",
    ],
    zip_safe=True,
    maintainer="smartrollerz",
    maintainer_email="info@dhbw-smartrollerz.org",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "parking_spot = parking_spot.detect:main",
        ],
    },
)
