#!/usr/bin/env python3
"""Move the robot along a face-based painting path derived from the object mesh."""

from pathlib import Path

import numpy as np
import trimesh
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.duration import Duration

from ament_index_python.packages import get_package_share_directory
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

from motion_control.cal_patten_ap2 import generate_face_based_path


def normal_to_quaternion(normal, up_hint=np.array([0.0, 0.0, 1.0])):
    """
    Build a quaternion (x, y, z, w) that orients the tool's approach axis
    (assumed -Z in the tool frame) opposite to the given surface normal,
    so the tool faces INTO the surface along -normal.
    """
    normal = normal / np.linalg.norm(normal)
    approach = -normal  # tool points opposite the outward surface normal

    # tool Z axis = approach direction
    z_axis = approach

    # pick a reference "up" not parallel to z_axis
    if abs(np.dot(z_axis, up_hint)) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up_hint, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # rotation matrix -> quaternion (x, y, z, w)
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


class ActionFacePattern(Node):
    def __init__(self):
        super().__init__('paintaction_ap2')

        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        placement = self._lookup_object_transform()
        self.path_points, self.path_normals = self._build_face_path(placement)

    def _lookup_object_transform(self, parent_frame='world', child_frame='object_link', timeout_sec=10.0):
        deadline = self.get_clock().now() + Duration(seconds=timeout_sec)
        while rclpy.ok() and self.get_clock().now() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    parent_frame, child_frame, rclpy.time.Time()
                )
                t = transform.transform.translation
                self.get_logger().info(
                    f'Found live transform {parent_frame} -> {child_frame}: '
                    f'x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}'
                )
                return np.array([t.x, t.y, t.z])
            except (LookupException, ConnectivityException, ExtrapolationException):
                rclpy.spin_once(self, timeout_sec=0.2)

        raise RuntimeError(
            f'Could not find transform {parent_frame} -> {child_frame} after {timeout_sec}s. '
            f'Is object_tf.launch.py running?'
        )

    def _build_face_path(self, placement):
        scale = 0.01

        mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        mesh.apply_scale(scale)

        points, normals = generate_face_based_path(mesh, points_per_unit_area=50, band_count=6)

        # shift from the mesh's local frame into world coordinates using the live TF placement
        points = points + placement

        self.get_logger().info(f'Built face-based path with {len(points)} points')
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
    node = ActionFacePattern()

    node.get_logger().info('Moving along the face-based painting path...')
    for point, normal in zip(node.path_points, node.path_normals):
        quat = [-0.5, 0.5, 0.5, 0.5]
        node.move_to(point, quat)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()