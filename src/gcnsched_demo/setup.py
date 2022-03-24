from setuptools import setup

package_name = 'gcnsched_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lilly',
    maintainer_email='lilliamc@usc.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scheduler = gcnsched_demo.scheduler:main',
            'sample_node = gcnsched_demo.sample_node:main',
            'service = gcnsched_demo.bandwidth_server:main',
            'client = gcnsched_demo.bandwidth_client_timer:main',
        ],
    },
)
