from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'environment'

package_dir = os.path.abspath(os.path.dirname(__file__))


def rel(path):
    return os.path.relpath(path, package_dir)


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), [rel(p) for p in glob(os.path.join(package_dir, 'urdf', '*.urdf'))]),
        (os.path.join('share', package_name, 'meshes'), [rel(p) for p in glob(os.path.join(package_dir, 'meshes', '*.stl'))]),
        (os.path.join('share', package_name, 'launch'), [rel(p) for p in glob(os.path.join(package_dir, 'launch', '*.py'))]),
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
