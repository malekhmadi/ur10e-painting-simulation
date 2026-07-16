from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    urdf_path = Path(__file__).resolve().parent.parent / 'urdf' / 'obj.urdf'
    robot_description = urdf_path.read_text(encoding='utf-8')

    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='-0.4'),
        DeclareLaunchArgument('y', default_value='0.55'),
        DeclareLaunchArgument('z', default_value='1.0'),
        DeclareLaunchArgument('roll', default_value='0.0'),
        DeclareLaunchArgument('pitch', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('parent_frame', default_value='world'),
        DeclareLaunchArgument('child_frame', default_value='object_link'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='object_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
            remappings=[('robot_description', 'object_description')],
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('z'),
                LaunchConfiguration('roll'), LaunchConfiguration('pitch'), LaunchConfiguration('yaw'),
                LaunchConfiguration('parent_frame'), LaunchConfiguration('child_frame'),
            ],
        ),
    ])
