#!/usr/bin/env python3
"""Execute the surface-grouped painting strokes on the robot. Adds the
object mesh as a MoveIt collision object, plans each stroke-to-stroke
transition live from the robot's current state, and tries alternative
candidates (next stroke, or the same stroke reversed) if a transition
plan fails before giving up on it."""

from pathlib import Path

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

from motion_control.cal_patten_ap4 import (
    normal_to_quaternion,
    read_mesh_scale_from_urdf,
    read_object_pose_from_launch,
    euler_to_quaternion,
)
from motion_control.cal_patten_ap6 import build_reachable_strokes

from motion_control.cal_patten_ap6 import build_reachable_strokes, add_object_to_planning_scene




class ActionStrokeSequence(Node):
    def __init__(self):
        super().__init__('paintaction_ap6')

        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.5)

        # Step 6: register the object as a real collision object in the
        # planning scene, so MoveIt refuses to plan any transition that
        # would pass through it.
        add_object_to_planning_scene(self.moveit2, logger=self.get_logger())

        self.get_logger().info('Building strokes and checking reachability...')
        self.strokes, dropped_points, _ = build_reachable_strokes(self.moveit2, logger=self.get_logger())
        self.get_logger().info(
            f'{len(self.strokes)} strokes survived filtering '
            f'({len(dropped_points)} individual points dropped as unreachable)'
        )


    def _plan_and_execute(self, position, quat_xyzw):
        """Plan (from current robot state) and execute a move to the given
        pose. Returns True on success, False if planning/execution failed —
        never raises, so callers can try alternatives."""
        trajectory = self.moveit2.plan(position=position.tolist(), quat_xyzw=quat_xyzw)
        if trajectory is None:
            return False

        self.moveit2.execute(trajectory)
        return self.moveit2.wait_until_executed()

    def _try_stroke(self, points, normals, reverse=False):
        """
        Step 5: plan a transition from wherever the robot currently is to
        this stroke's starting pose, then execute the full stroke if the
        transition succeeded. Optionally tries the stroke in reverse
        (step 7's "alternative candidate").
        """
        if reverse:
            points = points[::-1]
            normals = normals[::-1]

        start_point, start_normal = points[0], normals[0]
        start_quat = normal_to_quaternion(start_normal)

        transitioned = self._plan_and_execute(start_point, start_quat)
        if not transitioned:
            return False

        for point, normal in zip(points[1:], normals[1:]):
            quat = normal_to_quaternion(normal)
            success = self._plan_and_execute(point, quat)
            if not success:
                self.get_logger().warn('Mid-stroke move failed unexpectedly — continuing to next point')

        return True

    def run(self):
        remaining = list(self.strokes)

        while remaining:
            stroke_points, stroke_normals = remaining.pop(0)

            # Step 7: attempt forward direction, then reversed direction,
            # before giving up on this stroke and moving to the next candidate.
            if self._try_stroke(stroke_points, stroke_normals, reverse=False):
                self.get_logger().info(f'Stroke completed ({len(stroke_points)} points)')
                continue

            self.get_logger().warn('Transition failed forward — trying stroke reversed')
            if self._try_stroke(stroke_points, stroke_normals, reverse=True):
                self.get_logger().info(f'Stroke completed reversed ({len(stroke_points)} points)')
                continue

            self.get_logger().warn(
                f'Could not find a valid transition to this stroke '
                f'(forward or reversed) — skipping it, {len(remaining)} strokes left'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ActionStrokeSequence()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()