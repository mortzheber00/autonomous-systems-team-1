from setuptools import setup

package_name = "state_machine_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/state_machine.yaml"]),
        ("share/" + package_name + "/launch", ["launch/state_machine.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tom",
    maintainer_email="tom@example.com",
    description="High-level mission state machine for cave exploration",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "state_machine_node = state_machine_pkg.state_machine_node:main",
        ],
    },
)
