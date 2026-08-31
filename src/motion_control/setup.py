from setuptools import find_packages, setup

package_name = 'motion_control'

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
    maintainer='malek',
    maintainer_email='malekhmadi280@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'paintaction_ap1 = motion_control.paintaction_ap1:main',
            'paintaction_ap2 = motion_control.paintaction_ap2:main',
            'paintaction = motion_control.paintaction:main',
            'paintaction_ap3 = motion_control.paintaction_ap3:main',
            'paintaction_ap4 = motion_control.paintaction_ap4:main',
            'cal_patten_ap1 = motion_control.cal_patten_ap1:main',
            'cal_patten_ap2 = motion_control.cal_patten_ap2:main',
            'cal_patten_ap3 = motion_control.cal_patten_ap3:main',
            'cal_patten_ap4 = motion_control.cal_patten_ap4:main',
            'paint_marker = motion_control.paint_marker:main',
            'cal_patten_ap5 = motion_control.cal_patten_ap5:main',
            'paintaction_ap5 = motion_control.paintaction_ap5:main',
            'dataset_generator_ap1 = motion_control.dataset_generator_ap1:main',
            'paint_visualizer = motion_control.paint_visualizer:main',
            'cal_patten_ap6 = motion_control.cal_patten_ap6:main',  
            'paintaction_ap6 = motion_control.paintaction_ap6:main',
        ],
    },
)
