from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    urdf_path = Path(__file__).resolve().parent.parent / 'urdf' / 'obj.urdf'
    robot_description = urdf_path.read_text(encoding='utf-8')

    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='-0.2'),
        DeclareLaunchArgument('y', default_value='0.5'),
        DeclareLaunchArgument('z', default_value='1.2'),
        DeclareLaunchArgument('roll', default_value='1.57'),
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
                '--x', LaunchConfiguration('x'),
                '--y', LaunchConfiguration('y'),
                '--z', LaunchConfiguration('z'),
                '--roll', LaunchConfiguration('roll'),
                '--pitch', LaunchConfiguration('pitch'),
                '--yaw', LaunchConfiguration('yaw'),
                '--frame-id', LaunchConfiguration('parent_frame'),
                '--child-frame-id', LaunchConfiguration('child_frame'),
            ],
        ),
    ])
