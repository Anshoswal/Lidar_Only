# # from setuptools import find_packages, setup

# # package_name = 'lidar_only'

# # setup(
# #     name=package_name,
# #     version='0.0.0',
# #     packages=find_packages(exclude=['test']),
# #     data_files=[
# #         ('share/ament_index/resource_index/packages',
# #             ['resource/' + package_name]),
# #         ('share/' + package_name, ['package.xml']),
# #     ],
# #     install_requires=['setuptools'],
# #     zip_safe=True,
# #     maintainer='ansh',
# #     maintainer_email='anshoswal0006@gmail.com',
# #     description='TODO: Package description',
# #     license='TODO: License declaration',
# #     tests_require=['pytest'],
# #     entry_points={
# #         'console_scripts': [
# #             'cones_detector = lidar_only.cones_detector:main',
# #             'cone_new = lidar_only.cone_new:main',
# #             'actual = lidar_only.actual:main',
# #             'intensity_checker = lidar_only.intensity_checker:main',
# #             'db_and_ran = lidar_only.db_and_ran:main',
# #             'pre_final = lidar_only.pre_final:main',
# #             '1_D = lidar_only.1_D:main',
# #             'vis = lidar_only.vis:main',
# #             'end_game = lidar_only.end_game:main',
# #         ],
# #     },
# # )

# import os
# from glob import glob
# from setuptools import find_packages, setup

# package_name = 'lidar_only'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(exclude=['test']),
#     # This section ensures all necessary files are installed
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#         # CRITICAL: This line finds and installs your 'models' folder
#         (os.path.join('share', package_name, 'models'), glob(os.path.join(package_name, 'models', '*'))),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='ansh',
#     maintainer_email='anshoswal0006@gmail.com',
#     description='A ROS 2 node for real-time cone classification using a neural network.',
#     license='Apache-2.0',
#     tests_require=['pytest'],
#     # This lists the executable scripts you can run with `ros2 run`
#     entry_points={
#         'console_scripts': [
#             'end_game = lidar_only.end_game:main',
#         ],
#     },
# )

import os
from glob import glob
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
        # This line finds and installs your 'models' folder
        (os.path.join('share', package_name, 'models'), glob(os.path.join(package_name, 'models', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ansh',
    maintainer_email='anshoswal0006@gmail.com',
    description='A ROS 2 node for real-time cone classification using a neural network.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'end_game = lidar_only.end_game:main',
        ],
    },
)