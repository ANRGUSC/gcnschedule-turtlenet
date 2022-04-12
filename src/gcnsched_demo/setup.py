from setuptools import setup
import os
from glob import glob

package_name = 'gcnsched_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*_launch.py')),
    ],
    install_requires=['setuptools', 'numpy', 'torch'], #, 'matplotlib', 'networkx'],
    zip_safe=True,
    maintainer='lilly',
    maintainer_email='lilliamc@usc.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'executor = gcnsched_demo.executor_node:main',
            'scheduler = gcnsched_demo.scheduler:main',
            'bandwidth = gcnsched_demo.bandwidth_node:main',
            'visualizer = gcnsched_demo.visualizer:main'
        ],
    },
)
