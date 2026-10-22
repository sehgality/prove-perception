from setuptools import setup
from glob import glob
import os

package_name = 'prove_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aidan Sehgal',
    maintainer_email='aidan.sehgal21@gmail.com',
    description='PROVE camera + LiDAR fusion for Isaac Sim on Jetson Orin Nano.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = prove_fusion.detector_node:main',
            'fusion_node = prove_fusion.fusion_node:main',
            'lane_node = prove_fusion.lane_node:main',
            'viz_node = prove_fusion.viz_node:main',
        ],
    },
)
