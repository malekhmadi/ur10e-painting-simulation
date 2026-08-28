#!/usr/bin/env python3
"""Generate labeled training data: for every painting point from the
existing path-generation pipeline, ask MoveIt whether it's reachable and
collision-free, and save the result. No robot motion happens here."""

import csv
from pathlib import Path
from unittest import result

import numpy as np
from sklearn.preprocessing import scale
import trimesh
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

from motion_control.cal_patten_ap4 import (
    read_mesh_scale_from_urdf,
    read_object_pose_from_launch,
    euler_to_quaternion,
    quaternion_to_matrix4,
    normal_to_quaternion,
    generate_surface_based_path,
    ROW_SPACING,
    ROBOT_POSITION,
    ANGLE_RANGES,
    LIFT_DISTANCE,
)


class DatasetGenerator(Node):
    def __init__(self):
        super().__init__('dataset_generator_ap1')

        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )

        # give rclpy a chance to actually discover and connect to MoveIt's
        # services before we start calling compute_ik/plan
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.5)

        self.scale = read_mesh_scale_from_urdf()
        translation, quaternion = self._read_object_pose()
        self.path_points, self.path_normals = self._build_path(translation, quaternion)

    def _read_object_pose(self):
        translation, (roll, pitch, yaw) = read_object_pose_from_launch()
        quaternion = euler_to_quaternion(roll, pitch, yaw)
        self.get_logger().info(f'Object pose: pos={translation}, rpy=({roll}, {pitch}, {yaw})')
        return translation, quaternion

    def _build_path(self, translation, quaternion):

        mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        mesh.apply_scale(self.scale)
        mesh.apply_transform(quaternion_to_matrix4(quaternion))

        points, normals = generate_surface_based_path(
            mesh,
            row_spacing=ROW_SPACING,
            robot_position=ROBOT_POSITION,
            angle_ranges=ANGLE_RANGES,
            lift_distance=LIFT_DISTANCE,
        )
        points = points + translation

        self.get_logger().info(f'Loaded {len(points)} candidate painting points')
        return points, normals

    def check_reachable_and_collision_free(self, position, quat_xyzw):
        """
        Two-stage check matching MoveIt's real pipeline:
        1. Does an IK solution exist for this pose?
        2. If yes, can a collision-free trajectory actually be planned to it?

        Returns a dict with both results — no robot motion happens (plan only,
        never executed).
        """
        result = {'ik_found': False, 'plan_found': False}

        joint_state = self.moveit2.compute_ik(
            position=position.tolist(),
            quat_xyzw=quat_xyzw,
        )
        if joint_state is None:
            return result  # fails at stage 1, stage 2 is meaningless — skip it

        result['ik_found'] = True

        # stage 2: actually plan a trajectory to this pose (collision-checked by MoveIt)
        trajectory = self.moveit2.plan(
            position=position.tolist(),
            quat_xyzw=quat_xyzw,
        )
        result['plan_found'] = trajectory is not None
        return result

    def generate_dataset(self, output_path, object_name):
        rows = []
        total = len(self.path_points)
        scale_x, scale_y, scale_z = self.scale

        for i, (point, normal) in enumerate(zip(self.path_points, self.path_normals)):
            quat = normal_to_quaternion(normal)
            result = self.check_reachable_and_collision_free(point, quat)

            rows.append({
                'object_name': object_name,
                'x': point[0], 'y': point[1], 'z': point[2],
                'nx': normal[0], 'ny': normal[1], 'nz': normal[2],
                'qx': quat[0], 'qy': quat[1], 'qz': quat[2], 'qw': quat[3],
                'scale_x': scale_x, 'scale_y': scale_y, 'scale_z': scale_z,
                'ik_found': int(result['ik_found']),
                'plan_found': int(result['plan_found']),
                'reachable': int(result['ik_found'] and result['plan_found']),
            })

            if (i + 1) % 25 == 0 or (i + 1) == total:
                self.get_logger().info(f'Checked {i + 1}/{total} points')

        file_exists = output_path.exists()
        with open(output_path, 'a', newline='') as f:  # 'a' = append, not overwrite
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not file_exists:
                writer.writeheader()  # only write header once, on first run
            writer.writerows(rows)

        ik_ok = sum(r['ik_found'] for r in rows)
        plan_ok = sum(r['plan_found'] for r in rows)
        self.get_logger().info(
            f'Appended {len(rows)} labeled points for "{object_name}" to {output_path}\n'
            f'  Scale: {self.scale}\n'
            f'  IK found: {ik_ok}/{len(rows)}\n'
            f'  Collision-free plan found: {plan_ok}/{len(rows)}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = DatasetGenerator()

    import sys
    object_name = sys.argv[1] if len(sys.argv) > 1 else 'unknown_object'

    output_path = Path.home() / 'ros2_project' / 'ml' / 'data' / 'generated' / 'combined_dataset.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node.generate_dataset(output_path, object_name)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()