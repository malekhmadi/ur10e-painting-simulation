#!/usr/bin/env python3
"""Move the robot along a surface-segmented painting path derived from the object mesh."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

from motion_control.cal_patten_ap4 import generate_surface_based_path


def euler_to_quaternion(roll, pitch, yaw):
    """Convert roll/pitch/yaw (radians) to quaternion (x, y, z, w)."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


def quaternion_to_matrix4(quat_xyzw):
    """Build a 4x4 homogeneous rotation matrix from a quaternion (x, y, z, w),
    suitable for trimesh.apply_transform()."""
    x, y, z, w = quat_xyzw
    rot3 = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)],
    ])
    matrix4 = np.eye(4)
    matrix4[:3, :3] = rot3
    return matrix4


def normal_to_quaternion(normal, up_hint=np.array([0.0, 0.0, 1.0])):
    """Orient the tool's approach axis (-Z) opposite the given surface normal."""
    normal = normal / np.linalg.norm(normal)
    z_axis = -normal

    if abs(np.dot(z_axis, up_hint)) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up_hint, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    trace = np.trace(rot_matrix)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
        y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
        z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
    else:
        i = np.argmax(np.diag(rot_matrix))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            z = 0.25 * s

    return [x, y, z, w]


def read_mesh_scale_from_urdf(package_name='environment', urdf_filename='obj.urdf'):
    """Read the <mesh scale="..."/> value directly from the object's URDF."""
    urdf_path = Path(get_package_share_directory(package_name)) / 'urdf' / urdf_filename
    if not urdf_path.exists():
        raise FileNotFoundError(f'URDF not found: {urdf_path}')

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mesh_elem = next(root.iter('mesh'), None)
    if mesh_elem is None:
        raise ValueError(f'No <mesh> tag found in URDF: {urdf_path}')

    scale_attr = mesh_elem.get('scale', '1 1 1')
    scale = [float(v) for v in scale_attr.split()]
    if len(scale) != 3:
        raise ValueError(f'Malformed scale attribute: {scale_attr}')
    return scale


def read_object_pose_from_launch(package_name='environment', launch_filename='object_tf.launch.py'):
    """Read x/y/z/roll/pitch/yaw default values directly out of the installed
    object_tf.launch.py file."""
    launch_path = Path(get_package_share_directory(package_name)) / 'launch' / launch_filename
    if not launch_path.exists():
        raise FileNotFoundError(f'Launch file not found: {launch_path}')

    text = launch_path.read_text(encoding='utf-8')

    def extract(name, default):
        match = re.search(
            rf"DeclareLaunchArgument\(\s*['\"]{name}['\"]\s*,\s*default_value\s*=\s*['\"]([^'\"]+)['\"]",
            text,
        )
        return float(match.group(1)) if match else default

    x = extract('x', 0.0)
    y = extract('y', 0.0)
    z = extract('z', 0.0)
    roll = extract('roll', 0.0)
    pitch = extract('pitch', 0.0)
    yaw = extract('yaw', 0.0)

    return np.array([x, y, z]), (roll, pitch, yaw)


class ActionSurfaceSegmentedPattern(Node):
    def __init__(self):
        super().__init__('paintaction_ap4')
        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )

        translation, euler = read_object_pose_from_launch()
        quaternion = euler_to_quaternion(*euler)
        self.get_logger().info(
            f'Object pose from launch file: pos={translation}, rpy={euler}'
        )

        self.path_points, self.path_normals = self._build_surface_path(translation, quaternion)

    def _build_surface_path(self, translation, quaternion):
        scale = read_mesh_scale_from_urdf()

        mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        mesh.apply_scale(scale)
        # rotate the MESH itself before generating the path, so facet normals
        # and tangent frames are computed in the object's final orientation
        mesh.apply_transform(quaternion_to_matrix4(quaternion))

        points, normals = generate_surface_based_path(
            mesh,
            row_spacing=0.05,
            min_facet_area=1e-6,
            robot_position=np.array([0.0, 0.0, -1.0]),
            angle_ranges=((90.0, 110.0),),
            lift_distance=0.03,
        )

        # translation only now — rotation already baked into the mesh above
        points = points + translation 

        self.get_logger().info(f'Built surface-segmented path with {len(points)} points')
        return points, normals

    def move_to(self, position, quat_xyzw):
        self.moveit2.move_to_pose(
            position=position.tolist(),
            quat_xyzw=quat_xyzw,
            cartesian=True,
            cartesian_max_step=0.002,
            cartesian_fraction_threshold=0.3,
        )
        success = self.moveit2.wait_until_executed()
        if success:
            self.get_logger().info('Move succeeded')
        else:
            self.get_logger().warn('Move failed')


def main(args=None):
    rclpy.init(args=args)
    node = ActionSurfaceSegmentedPattern()

    node.get_logger().info('Moving along the surface-segmented painting path...')
    for point, normal in zip(node.path_points, node.path_normals):
        quat = normal_to_quaternion(normal) 
        node.move_to(point, quat)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()