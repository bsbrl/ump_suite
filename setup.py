from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ump_suite'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Required: ament index + package manifest
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Install launch files so `ros2 launch ump_suite app.launch.py` works
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raianbsbrl',
    maintainer_email='raianbsbrl@todo.todo',
    description='Sensapex UMP + ODrive + Blackfly (PySpin) ROS2 nodes + GUI + logging',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'gui_node = ump_suite.gui_node:main',
            'ump_driver_node = ump_suite.ump_driver_node:main',
            'odrive_driver_node = ump_suite.odrive_driver_node:main',
            'camera_node = ump_suite.camera_node:main',
            'logger_node = ump_suite.logger_node:main',
        ],
    },
)