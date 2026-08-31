#!/usr/bin/env python3
"""Execute the face-entry-point painting pipeline (cal_pattern_ap5.py) on
the robot. Only face-to-face transitions are collision-checked live; once
a face's entry point is found reachable, every remaining point in that
face's zigzag is trusted and painted with no further per-point checking."""

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

from motion_control.cal_patten_ap4 import normal_to_quaternion
from motion_control.cal_patten_ap5 import (
    load_and_prepare_mesh,
    add_object_to_planning_scene,
    generate_all_surfaces,
    find_face_entry_points,
    build_face_segments,
)


class ActionFaceEntrySequence(Node):
    def __init__(self):
        super().__init__('paintaction_ap5')

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

        # scene setup — same collision object registration as cal_pattern_ap5/ap6
        add_object_to_planning_scene(self.moveit2, logger=logger)

        # Phases 1-2: raw zigzag strokes per face, untouched
        mesh, translation = load_and_prepare_mesh()
        all_surfaces = generate_all_surfaces(mesh, logger=logger)

        # Phase 3: search each face for its first reachable entry point —
        # this is the ONLY per-point IK/plan checking that happens
        all_surfaces = find_face_entry_points(self.moveit2, all_surfaces, max_entry_attempts=10, logger=logger)

        # Phase 4: build one trusted segment per reachable face, from its
        # entry point to the end — everything in it is trusted, unchecked
        self.valid_face_segments = build_face_segments(all_surfaces, logger=logger)

        logger.info(f'Ready to execute {len(self.valid_face_segments)} face segments')

    def _plan_and_execute(self, position, quat_xyzw):
        """Plan (from current robot state) and execute a move. Returns
        True/False, never raises."""
        trajectory = self.moveit2.plan(position=position.tolist(), quat_xyzw=quat_xyzw)
        if trajectory is None:
            return False
        self.moveit2.execute(trajectory)
        return self.moveit2.wait_until_executed()

    def _paint_face_segment(self, segment, reverse=False):
        """
        Phase 5: attempt the transition to this face segment's start pose
        (its found entry point, or the LAST point if reversed — since a
        face has no separately-verified "other end", reversing simply
        starts painting from the opposite tip of the trusted run). If the
        transition succeeds, paint every remaining point in the segment
        with NO further per-point reachability checking, matching
        cal_pattern_ap5's "trust the rest of the face" principle.
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
                    f"Segment {segment['segment_id']} (surface {segment['surface_id']}): "
                    f"mid-segment move failed unexpectedly (untested point, trusted by design) — continuing"
                )

        return True

    def run(self):
        logger = self.get_logger()

        # queue of (segment, attempt_count) — requeue on failure instead of
        # discarding, same "try alternative ordering before rejecting" logic
        queue = [(seg, 0) for seg in self.valid_face_segments]
        max_attempts = len(self.valid_face_segments) + 1

        painted = 0
        permanently_skipped = []

        while queue:
            segment, attempts = queue.pop(0)
            seg_id = segment['segment_id']
            surface_id = segment['surface_id']

            if self._paint_face_segment(segment, reverse=False):
                logger.info(
                    f'Segment {seg_id} (surface {surface_id}): transition OK (forward), '
                    f'painted {len(segment["points"])} points'
                )
                painted += 1
                continue

            logger.warn(f'Segment {seg_id} (surface {surface_id}): forward transition failed — trying reversed')
            if self._paint_face_segment(segment, reverse=True):
                logger.info(
                    f'Segment {seg_id} (surface {surface_id}): transition OK (reversed), '
                    f'painted {len(segment["points"])} points'
                )
                painted += 1
                continue

            attempts += 1
            if attempts < max_attempts and queue:
                logger.warn(
                    f'Segment {seg_id} (surface {surface_id}): both directions failed '
                    f'(attempt {attempts}) — re-queued to retry after other segments'
                )
                queue.append((segment, attempts))
            else:
                logger.warn(
                    f'Segment {seg_id} (surface {surface_id}): no valid transition found '
                    f'after {attempts} attempts — skipping'
                )
                permanently_skipped.append(seg_id)

        logger.info(
            f'Execution complete: {painted}/{len(self.valid_face_segments)} face segments painted, '
            f'{len(permanently_skipped)} permanently skipped (ids: {permanently_skipped})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ActionFaceEntrySequence()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()