from setuptools import find_packages, setup

package_name = 'environment'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='TODO',
    author_email='todo@todo.com',
    maintainer='TODO',
    maintainer_email='todo@todo.com',
    description='Environment package for ROS2 simulation',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wall_visualizer = environment.wall_visualizer:main',
        ],
    },
)
