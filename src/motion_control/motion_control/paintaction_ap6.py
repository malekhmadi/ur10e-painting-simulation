#!/usr/bin/env python3
"""Execute the phased painting pipeline (Phases 1-5 from cal_pattern_ap6.py)
on the robot. Segments are only dropped if truly unreachable (no IK).
Transition failures trigger retry (reversed segment, then requeue for a
later attempt) before a segment is ever skipped."""

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

from motion_control.cal_patten_ap4 import normal_to_quaternion
from motion_control.cal_patten_ap6 import (
    load_and_prepare_mesh,
    add_object_to_planning_scene,
    generate_all_surfaces,
    check_reachability_preserve_structure,
    split_into_valid_segments,
)


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

        logger = self.get_logger()

        # Phase 6 (scene setup, same as before)
        add_object_to_planning_scene(self.moveit2, logger=logger)

        # Phases 1-2: generate every surface's raw zigzag strokes, untouched
        mesh, translation = load_and_prepare_mesh()
        all_surfaces = generate_all_surfaces(mesh, logger=logger)

        # Phase 3: label every point IK-reachable/unreachable, and
        # separately track direct-transition feasibility — nothing removed yet
        all_surfaces = check_reachability_preserve_structure(self.moveit2, all_surfaces, logger=logger)

        # Phase 4: split into valid_paint_segments — only IK-unreachable
        # points create gaps; a "failed_transition" point is NOT dropped here
        self.valid_paint_segments = split_into_valid_segments(all_surfaces, min_segment_length=2, logger=logger)

        logger.info(f'Ready to execute {len(self.valid_paint_segments)} valid paint segments')

    def _plan_and_execute(self, position, quat_xyzw):
        """Plan (from current robot state) and execute a move to the given
        pose. Returns True on success, False otherwise — never raises."""
        trajectory = self.moveit2.plan(position=position.tolist(), quat_xyzw=quat_xyzw)
        if trajectory is None:
            return False
        self.moveit2.execute(trajectory)
        return self.moveit2.wait_until_executed()

    def _paint_segment(self, segment, reverse=False):
        """
        Phase 5: attempt the transition to this segment's start pose, then
        (if the transition succeeded) paint every point in the segment.
        Returns True only if the transition itself succeeded — mid-segment
        move failures are logged but don't cancel the rest of the segment,
        since every point in it was already confirmed IK-reachable.
        """
        points = segment['points']
        normals = segment['normals']
        if reverse:
            points = points[::-1]
            normals = normals[::-1]

        start_quat = normal_to_quaternion(normals[0])
        transitioned = self._plan_and_execute(points[0], start_quat)
        if not transitioned:
            return False

        for point, normal in zip(points[1:], normals[1:]):
            quat = normal_to_quaternion(normal)
            success = self._plan_and_execute(point, quat)
            if not success:
                self.get_logger().warn(
                    f"Segment {segment['segment_id']}: mid-segment move failed unexpectedly "
                    f"(point was pre-checked IK-reachable) — continuing"
                )

        return True

    def run(self):
        logger = self.get_logger()

        # queue of (segment, attempt_count) — requeue on failure instead of
        # discarding, per Phase 5's "try alternative ordering before rejecting"
        queue = [(seg, 0) for seg in self.valid_paint_segments]
        max_attempts = len(self.valid_paint_segments) + 1  # bound to avoid infinite loop

        painted = 0
        permanently_skipped = []

        while queue:
            segment, attempts = queue.pop(0)
            seg_id = segment['segment_id']

            if self._paint_segment(segment, reverse=False):
                logger.info(f'Segment {seg_id}: transition OK (forward), painted {len(segment["points"])} points')
                painted += 1
                continue

            logger.warn(f'Segment {seg_id}: forward transition failed — trying reversed')
            if self._paint_segment(segment, reverse=True):
                logger.info(f'Segment {seg_id}: transition OK (reversed), painted {len(segment["points"])} points')
                painted += 1
                continue

            attempts += 1
            if attempts < max_attempts and queue:
                logger.warn(
                    f'Segment {seg_id}: both directions failed (attempt {attempts}) — '
                    f're-queued to retry after other segments'
                )
                queue.append((segment, attempts))
            else:
                logger.warn(f'Segment {seg_id}: no valid transition found after {attempts} attempts — skipping')
                permanently_skipped.append(seg_id)

        logger.info(
            f'Execution complete: {painted}/{len(self.valid_paint_segments)} segments painted, '
            f'{len(permanently_skipped)} permanently skipped (ids: {permanently_skipped})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ActionStrokeSequence()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()