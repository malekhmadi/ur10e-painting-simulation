"""cal_pattern_ap6.py — phased painting pipeline.

Data flow:
  all_surfaces
    -> surface
      -> zigzag_strokes
        -> ordered painting points (with reachable/unreachable status)

  valid_paint_segments   (built from strokes, splitting AROUND unreachable
                          points — never bridging a gap)
    -> ordered painting points

  transitions             (MoveIt-checked connections between segments)

PHASE 5 NOTE: true sequential transition validity (does the robot actually
reach segment B's start AFTER physically finishing segment A) can only be
verified by really executing strokes in order — MoveIt always plans from
the robot's actual current state, which only reflects "just finished
segment A" if that motion really happened. This file's transition check
is therefore a best-effort static approximation (checked from a fixed
current state), used only for visualization. The real, execution-order-
correct version — with retry/reverse-on-failure — lives in
paintaction_ap6.py, where strokes are actually run in sequence.
"""

from pathlib import Path

import numpy as np
import trimesh
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

from motion_control.cal_patten_ap4 import (
    segment_surfaces,
    generate_passes_for_surface,
    normal_to_quaternion,
    check_pose_reachable,
    read_mesh_scale_from_urdf,
    read_object_pose_from_launch,
    euler_to_quaternion,
    quaternion_to_matrix4,
    ROW_SPACING,
    ROBOT_POSITION,
    ANGLE_RANGES,
    PAINT_DISTANCE,
)


# ---------------------------------------------------------------------------
# Mesh loading (shared, world-space — scale + rotation + translation applied)
# ---------------------------------------------------------------------------

def load_and_prepare_mesh():
    """Load obj.stl, apply URDF scale, launch-file rotation AND translation,
    so the mesh and every point generated from it are in world coordinates."""
    scale = read_mesh_scale_from_urdf()

    mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.apply_scale(scale)

    translation, (roll, pitch, yaw) = read_object_pose_from_launch()
    quat = euler_to_quaternion(roll, pitch, yaw)
    mesh.apply_transform(quaternion_to_matrix4(quat))
    mesh.apply_translation(translation)

    return mesh, translation


def add_object_to_planning_scene(moveit2, logger=None):
    """Register the object mesh as a real MoveIt collision object, so
    reachability/plan checks correctly account for it as an obstacle."""
    scale = read_mesh_scale_from_urdf()
    mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
    translation, (roll, pitch, yaw) = read_object_pose_from_launch()
    quat = euler_to_quaternion(roll, pitch, yaw)

    moveit2.add_collision_mesh(
        filepath=str(mesh_path),
        id='painting_object',
        position=translation.tolist(),
        quat_xyzw=quat.tolist(),
        scale=scale,
        frame_id='world',
    )
    if logger:
        logger.info(f'Added collision object "painting_object" to planning scene at {translation}')


# ---------------------------------------------------------------------------
# PHASE 1 + 2 — generate ALL surfaces, each with its own zigzag stroke(s)
# No reachability filtering happens here. No surface is dropped for any
# reason other than being genuinely too small/degenerate to paint at all.
# ---------------------------------------------------------------------------

def generate_all_surfaces(mesh, row_spacing=ROW_SPACING, robot_position=ROBOT_POSITION,
                           angle_ranges=ANGLE_RANGES, paint_distance=PAINT_DISTANCE,
                           min_facet_area=1e-6, logger=None):
    """
    PHASE 1: detect every valid surface (facet) within the angle/area filters.
    PHASE 2: generate one zigzag stroke per surface, points kept in strict
             row-by-row order — no reachability check, no cross-surface
             connection.

    Returns:
        all_surfaces: list of dicts, one per surface:
            {
                'surface_id': int,
                'normal': (3,) array,
                'strokes': [
                    {
                        'stroke_id': int,
                        'points': (N,3) array, in original zigzag order,
                        'normals': (N,3) array,
                        'reachable': None,   # filled in during Phase 3
                    },
                    ...
                ]
            }
    """
    surfaces = segment_surfaces(mesh, robot_position, angle_ranges, min_facet_area)
    if not surfaces:
        raise RuntimeError('No surfaces passed segmentation/filtering — check angle_ranges or min_facet_area.')

    all_surfaces = []
    skipped = 0

    for surface_id, (face_indices, normal, _) in enumerate(surfaces):
        points, normals = generate_passes_for_surface(mesh, face_indices, normal, row_spacing=row_spacing)

        if len(points) == 0:
            skipped += 1
            if logger:
                logger.warn(f'Surface {surface_id}: produced 0 points, skipped (degenerate/too small)')
            continue

        if paint_distance:
            points = points + normals * paint_distance

        all_surfaces.append({
            'surface_id': surface_id,
            'normal': normal,
            'strokes': [{
                'stroke_id': 0,
                'points': points,
                'normals': normals,
                'reachable': None,
            }],
        })

    if logger:
        logger.info(f'PHASE 1+2: {len(all_surfaces)} surfaces generated ({skipped} skipped as degenerate)')
        for s in all_surfaces:
            for stroke in s['strokes']:
                logger.info(
                    f"  surface {s['surface_id']}: stroke {stroke['stroke_id']} "
                    f"has {len(stroke['points'])} points"
                )

    return all_surfaces


# ---------------------------------------------------------------------------
# PHASE 3 — check reachability WITHOUT destroying structure
# Every point gets a True/False label, in place. Nothing is removed,
# reordered, or concatenated here.
# ---------------------------------------------------------------------------

def check_reachability_preserve_structure(moveit2, all_surfaces, logger=None):
    """
    PHASE 3: classify each point as one of three states:
      - unreachable: no IK solution exists
      - failed_transition: IK works, but a direct plan from the current state
        is not collision-free
      - reachable: IK exists and the direct pose is valid

    The painting pipeline should only drop a point when it is truly
    unreachable (no IK). A direct motion failure is a transition problem,
    not proof that the pose is invalid, and it must be tracked separately.
    """
    total_points = sum(len(stroke['points']) for s in all_surfaces for stroke in s['strokes'])
    checked = 0
    total_reachable = 0

    for surface in all_surfaces:
        for stroke in surface['strokes']:
            reachable_flags = np.zeros(len(stroke['points']), dtype=bool)
            transition_statuses = np.empty(len(stroke['points']), dtype=object)

            for i, (point, normal) in enumerate(zip(stroke['points'], stroke['normals'])):
                quat = normal_to_quaternion(normal)
                result = check_pose_reachable(moveit2, point, quat)

                if result['ik_found']:
                    reachable_flags[i] = True
                    transition_statuses[i] = 'reachable' if result['plan_found'] else 'failed_transition'
                    total_reachable += 1
                else:
                    reachable_flags[i] = False
                    transition_statuses[i] = 'unreachable'

                checked += 1

                if logger and (checked % 25 == 0 or checked == total_points):
                    logger.info(f'PHASE 3: checked {checked}/{total_points} points')

            stroke['reachable'] = reachable_flags
            stroke['transition_status'] = transition_statuses

    if logger:
        logger.info(f'PHASE 3 complete: {total_reachable}/{total_points} points IK-reachable')

    return all_surfaces


# ---------------------------------------------------------------------------
# PHASE 4 — split each stroke into valid_paint_segments AROUND gaps
# Consecutive reachable points stay connected as one segment. An
# unreachable point ends the current segment; the next reachable point
# (if any) starts a new one. Never bridges a gap.
# ---------------------------------------------------------------------------

def split_into_valid_segments(all_surfaces, min_segment_length=2, logger=None):
    """
    PHASE 4: walk each stroke's points in original order. Build up
    contiguous runs of reachable points as separate segments. A run
    shorter than min_segment_length is dropped (too short to be a
    meaningful stroke), but this NEVER causes points from two different
    runs to be joined together.

    Returns:
        valid_paint_segments: list of dicts:
            {
                'segment_id': int,
                'surface_id': int,
                'stroke_id': int,
                'points': (M,3) array,
                'normals': (M,3) array,
            }
    """
    valid_paint_segments = []
    segment_id = 0
    dropped_short_runs = 0

    for surface in all_surfaces:
        for stroke in surface['strokes']:
            points = stroke['points']
            normals = stroke['normals']
            reachable = stroke['reachable']

            run_points, run_normals = [], []

            def flush_run():
                nonlocal segment_id, dropped_short_runs
                if len(run_points) == 0:
                    return
                if len(run_points) >= min_segment_length:
                    valid_paint_segments.append({
                        'segment_id': segment_id,
                        'surface_id': surface['surface_id'],
                        'stroke_id': stroke['stroke_id'],
                        'points': np.array(run_points),
                        'normals': np.array(run_normals),
                    })
                    segment_id += 1
                else:
                    dropped_short_runs += 1

            for point, normal, is_reachable in zip(points, normals, reachable):
                if is_reachable:
                    run_points.append(point)
                    run_normals.append(normal)
                else:
                    flush_run()
                    run_points, run_normals = [], []

            flush_run()  # flush whatever's left at the end of the stroke

    if logger:
        logger.info(
            f'PHASE 4: {len(valid_paint_segments)} valid paint segments created '
            f'({dropped_short_runs} short runs dropped, each < {min_segment_length} points)'
        )
        for seg in valid_paint_segments:
            logger.info(
                f"  segment {seg['segment_id']} (surface {seg['surface_id']}, "
                f"stroke {seg['stroke_id']}): {len(seg['points'])} points"
            )

    return valid_paint_segments


# ---------------------------------------------------------------------------
# PHASE 5 — transitions between valid_paint_segments (best-effort, offline)
# See module docstring: this is an approximation for visualization only.
# Real order-dependent transition success/failure is determined by
# paintaction_ap6.py during actual execution.
# ---------------------------------------------------------------------------

def plan_transitions(moveit2, valid_paint_segments, logger=None):
    """
    PHASE 5 (offline approximation): for each consecutive pair of segments,
    check whether MoveIt can plan to the next segment's start pose. Since
    this never actually executes any motion, every check plans from the
    SAME fixed current robot state — it does not know "the robot just
    finished segment A." Use this only to sanity-check gross reachability
    of transition targets, not as proof a real sequential run will succeed.

    Returns:
        transitions: list of dicts:
            {'from_segment_id':, 'to_segment_id':, 'success': bool}
    """
    transitions = []
    successes = 0

    for i in range(len(valid_paint_segments) - 1):
        seg_a = valid_paint_segments[i]
        seg_b = valid_paint_segments[i + 1]

        target_point = seg_b['points'][0]
        target_normal = seg_b['normals'][0]
        quat = normal_to_quaternion(target_normal)

        trajectory = moveit2.plan(position=target_point.tolist(), quat_xyzw=quat)
        success = trajectory is not None

        transitions.append({
            'from_segment_id': seg_a['segment_id'],
            'to_segment_id': seg_b['segment_id'],
            'success': success,
        })
        successes += int(success)

    if logger:
        logger.info(
            f'PHASE 5 (approximate): {successes}/{len(transitions)} transition targets '
            f'reachable from current state — NOT a guarantee of real sequential success; '
            f'see paintaction_ap6.py for the real execution-order check'
        )

    return transitions


# ---------------------------------------------------------------------------
# Full pipeline (phases 1-5)
# ---------------------------------------------------------------------------

def build_full_pipeline(moveit2, min_segment_length=2, logger=None):
    """Run all 5 phases and return everything: raw surfaces (for debugging/
    visualization), the final valid segments, and the transition checks."""
    mesh, translation = load_and_prepare_mesh()

    all_surfaces = generate_all_surfaces(mesh, logger=logger)
    all_surfaces = check_reachability_preserve_structure(moveit2, all_surfaces, logger=logger)
    valid_paint_segments = split_into_valid_segments(all_surfaces, min_segment_length, logger=logger)
    transitions = plan_transitions(moveit2, valid_paint_segments, logger=logger)

    return mesh, all_surfaces, valid_paint_segments, transitions

def visualize_all_surfaces(mesh, all_surfaces, output_path='/tmp/phase1_surfaces.png', title=None):
    """Requirement 1: show every generated surface, one color each,
    before any stroke generation or filtering."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.08, linewidths=0.1, facecolor='gray'))

    cmap = plt.get_cmap('tab20')
    for surface in all_surfaces:
        color = cmap(surface['surface_id'] % 20)
        for stroke in surface['strokes']:
            ax.scatter(*stroke['points'].T, color=color, s=6)

    ax.set_title(title if title is not None else f"Phase 1: {len(all_surfaces)} surfaces detected")
    _apply_equal_bounds(ax, mesh)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")


def visualize_raw_zigzags(mesh, all_surfaces, output_path='/tmp/phase2_zigzags.png', title=None):
    """Requirement 2: show original zigzag strokes BEFORE reachability
    filtering — connected lines, one color per stroke."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.08, linewidths=0.1, facecolor='gray'))

    cmap = plt.get_cmap('tab20')
    for surface in all_surfaces:
        for stroke in surface['strokes']:
            color = cmap(surface['surface_id'] % 20)
            pts = stroke['points']
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.5)
            ax.scatter(*pts[0], color=color, marker='o', s=60, edgecolors='black')
            ax.scatter(*pts[-1], color=color, marker='X', s=60, edgecolors='black')

    ax.set_title(title if title is not None else "Phase 2: raw zigzag strokes, before reachability filtering")
    _apply_equal_bounds(ax, mesh)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")


def visualize_segments_and_transitions(mesh, all_surfaces, valid_paint_segments, transitions,
                                        output_path='/tmp/phase4_5_segments.png'):
    """Requirements 3-6: green = reachable segments (as separate connected
    runs, gaps NOT bridged), red = unreachable points, orange = failed
    transitions, clearly distinguishing painting strokes (solid green)
    from transition attempts (dashed orange/gray)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.08, linewidths=0.1, facecolor='gray'))

    # red: every truly unreachable point (no IK solution)
    # orange: points that are IK-reachable but whose direct transition is blocked
    for surface in all_surfaces:
        for stroke in surface['strokes']:
            statuses = stroke.get('transition_status', np.array(['reachable'] * len(stroke['points']), dtype=object))
            unreachable_mask = (statuses == 'unreachable')
            if unreachable_mask.any():
                pts = stroke['points'][unreachable_mask]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='red', marker='x', s=40)

            transition_failed_mask = (statuses == 'failed_transition')
            if transition_failed_mask.any():
                pts = stroke['points'][transition_failed_mask]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='orange', marker='D', s=35)

    # green: each valid segment drawn as its OWN connected line — gaps are
    # visually real gaps, never bridged into one continuous stroke
    for seg in valid_paint_segments:
        pts = seg['points']
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='green', linewidth=2.5)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='green', s=15)

    # orange dashed: transition attempts between consecutive segments —
    # solid orange if successful, dashed red-orange if failed
    for t in transitions:
        seg_a = next(s for s in valid_paint_segments if s['segment_id'] == t['from_segment_id'])
        seg_b = next(s for s in valid_paint_segments if s['segment_id'] == t['to_segment_id'])
        p_a = seg_a['points'][-1]
        p_b = seg_b['points'][0]
        style = '-' if t['success'] else '--'
        color = 'orange' if t['success'] else 'darkred'
        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], [p_a[2], p_b[2]],
                color=color, linestyle=style, linewidth=1.0, alpha=0.7)

    ax.set_title(
        "Phase 4/5 — green: valid paint segments (gaps preserved) | "
        "red X: unreachable | orange: transitions (solid=ok, dashed=failed*)\n"
        "*offline approximation — see paintaction_ap6.py for real execution-order transitions"
    )
    _apply_equal_bounds(ax, mesh)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")


def _apply_equal_bounds(ax, mesh):
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    max_range = (bounds[1] - bounds[0]).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)



def main():
    rclpy.init()
    node = Node('cal_pattern_ap6_check')
    callback_group = ReentrantCallbackGroup()
    moveit2 = MoveIt2(
        node=node,
        joint_names=ur.joint_names(),
        base_link_name=ur.base_link_name(),
        end_effector_name=ur.end_effector_name(),
        group_name=ur.MOVE_GROUP_ARM,
        callback_group=callback_group,
    )
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.5)

    logger = node.get_logger()
    add_object_to_planning_scene(moveit2, logger=logger)

    mesh, all_surfaces, valid_paint_segments, transitions = build_full_pipeline(
        moveit2, min_segment_length=2, logger=logger
    )

    visualize_all_surfaces(mesh, all_surfaces)
    visualize_raw_zigzags(mesh, all_surfaces)
    visualize_segments_and_transitions(mesh, all_surfaces, valid_paint_segments, transitions)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()