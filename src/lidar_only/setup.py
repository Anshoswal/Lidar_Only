from setuptools import find_packages, setup

package_name = 'lidar_only'

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
    maintainer='ansh',
    maintainer_email='anshoswal0006@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cones_detector = lidar_only.cones_detector:main',
            'cone_new = lidar_only.cone_new:main',
            'actual_detector = lidar_only.actual_detector:main',
            'intensity_checker = lidar_only.intensity_checker:main',
            'db_and_ran = lidar_only.db_and_ran:main',
            'final = lidar_only.final:main',
        ],
    },
)
