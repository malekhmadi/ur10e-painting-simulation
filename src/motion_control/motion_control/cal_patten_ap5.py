"""cal_pattern_ap5.py — face-level entry-point reachability checking.

Instead of checking every point on every zigzag (expensive), this checks
only the FIRST point when entering a face. If it fails, the next point in
that same face is tried as an alternative entry point, and so on. Once a
reachable entry point is found, the rest of that face's zigzag is trusted
and included with no further per-point checks. Collision-aware transition
checks only happen when moving from one face to the next — never inside
a face's own zigzag.

This is a deliberate speed/thoroughness tradeoff versus cal_pattern_ap6.py's
exhaustive per-point checking: far fewer MoveIt calls, at the cost of not
guaranteeing every single point mid-zigzag is individually verified.

Data flow (same shape as ap6, for compatibility with shared visualizers):
  all_surfaces
    -> surface
      -> strokes (one per surface)
        -> points, normals, entry_index, entry_attempts

  valid_face_segments
    -> points from entry_index onward (untested points are trusted)

  face_transitions
    -> MoveIt-checked connections between consecutive face segments
"""

import numpy as np
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
)
from motion_control.cal_patten_ap6 import (
    load_and_prepare_mesh,
    add_object_to_planning_scene,
    visualize_all_surfaces,
    visualize_raw_zigzags,
    _apply_equal_bounds,
)


SIMPLIFICATION_THRESHOLD = 1.4
SEGMENT_PRIORITY_Z_WEIGHT = 0.5
SEGMENT_PRIORITY_Y_WEIGHT = 0.5


def _surface_plane_basis(points, normal):
    """Build a stable orthonormal basis on the surface plane for local zigzag analysis."""
    normal = np.asarray(normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    normal = normal / normal_norm

    axis_h = None
    if len(points) >= 2:
        for i in range(1, len(points)):
            ref = points[i] - points[0]
            ref = ref - np.dot(ref, normal) * normal
            ref_norm = np.linalg.norm(ref)
            if ref_norm > 1e-12:
                axis_h = ref / ref_norm
                break

    if axis_h is None:
        for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            projected = axis - np.dot(axis, normal) * normal
            projected_norm = np.linalg.norm(projected)
            if projected_norm > 1e-12:
                axis_h = projected / projected_norm
                break

    if axis_h is None:
        axis_h = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), normal) * normal
        axis_h_norm = np.linalg.norm(axis_h)
        if axis_h_norm < 1e-12:
            axis_h = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), normal) * normal
            axis_h_norm = np.linalg.norm(axis_h)
        if axis_h_norm < 1e-12:
            axis_h = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), normal) * normal
            axis_h_norm = np.linalg.norm(axis_h)
        if axis_h_norm < 1e-12:
            axis_h = np.array([1.0, 0.0, 0.0])
        else:
            axis_h = axis_h / axis_h_norm

    axis_h = axis_h / np.linalg.norm(axis_h)
    axis_v = np.cross(normal, axis_h)
    axis_v_norm = np.linalg.norm(axis_v)
    if axis_v_norm < 1e-12:
        for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            candidate = np.cross(normal, axis)
            candidate_norm = np.linalg.norm(candidate)
            if candidate_norm > 1e-12:
                axis_v = candidate / candidate_norm
                break
    if np.linalg.norm(axis_v) < 1e-12:
        axis_v = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), normal) * normal
        axis_v_norm = np.linalg.norm(axis_v)
        if axis_v_norm < 1e-12:
            axis_v = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), normal) * normal
            axis_v_norm = np.linalg.norm(axis_v)
        if axis_v_norm < 1e-12:
            axis_v = np.array([0.0, 1.0, 0.0])
        else:
            axis_v = axis_v / axis_v_norm
    else:
        axis_v = axis_v / np.linalg.norm(axis_v)

    return axis_h, axis_v


def simplify_small_zigzags(points, normals, surface_id=None,
                          threshold=SIMPLIFICATION_THRESHOLD, logger=None):
    """Collapse tiny local zigzag turns after raw stroke generation.

    The raw generator creates row-by-row zigzags. In narrow regions, one axis of a
    local turn can be much smaller than the other (for example a tiny horizontal
    step inside a much larger vertical pass). In those cases, dropping the short
    side-step keeps the path continuous and ordered while preserving the dominant
    coverage direction.

    The threshold is intentionally easy to tune: lower values preserve more of the
    original zigzag, larger values simplify more aggressively. A default of 0.3
    means the small side-step must be under 30% of the dominant axis before it is
    removed, which keeps normal zigzags intact while reducing narrow-region noise.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)

    if points.size == 0 or len(points) < 3:
        return points, normals
    if len(normals) != len(points):
        return points, normals

    normal = normals[0]
    axis_h, axis_v = _surface_plane_basis(points, normal)

    kept = [0]
    simplified_horizontal = 0
    simplified_vertical = 0

    for i in range(1, len(points) - 1):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[i + 1]

        d_prev = curr - prev
        d_next = nxt - curr

        horizontal_prev = abs(float(np.dot(d_prev, axis_h)))
        vertical_prev = abs(float(np.dot(d_prev, axis_v)))
        horizontal_next = abs(float(np.dot(d_next, axis_h)))
        vertical_next = abs(float(np.dot(d_next, axis_v)))

        horizontal_length = max(horizontal_prev, horizontal_next)
        vertical_length = max(vertical_prev, vertical_next)

        if horizontal_length > 0.0 and vertical_length > 0.0:
            if horizontal_length < threshold * vertical_length:
                simplified_horizontal += 1
                if logger:
                    logger.info(
                        f"Simplified small horizontal zigzag on surface {surface_id} "
                        f"(local span {horizontal_length:.6f} vs {vertical_length:.6f})"
                    )
                continue
            if vertical_length < threshold * horizontal_length:
                simplified_vertical += 1
                if logger:
                    logger.info(
                        f"Simplified small vertical zigzag on surface {surface_id} "
                        f"(local span {vertical_length:.6f} vs {horizontal_length:.6f})"
                    )
                continue

        kept.append(i)

    kept.append(len(points) - 1)
    kept = sorted(set(kept))

    if len(kept) < 2:
        return points, normals

    simplified_points = points[kept]
    simplified_normals = normals[kept]

    if logger and (simplified_horizontal or simplified_vertical):
        logger.info(
            f"surface {surface_id}: simplified {simplified_horizontal} horizontal and "
            f"{simplified_vertical} vertical tiny zigzag apex(es); final path length {len(simplified_points)}"
        )

    return simplified_points, simplified_normals


# ---------------------------------------------------------------------------
# PHASE 1+2 — generate all surfaces + their raw zigzag strokes (identical
# to ap6's version; reused directly rather than duplicated)
# ---------------------------------------------------------------------------

def generate_all_surfaces(mesh, row_spacing=None, robot_position=None,
                           angle_ranges=None, paint_distance=None,
                           min_facet_area=1e-6, logger=None):
    from motion_control.cal_patten_ap4 import ROW_SPACING, ROBOT_POSITION, ANGLE_RANGES, PAINT_DISTANCE
    row_spacing = row_spacing or ROW_SPACING
    robot_position = robot_position if robot_position is not None else ROBOT_POSITION
    angle_ranges = angle_ranges or ANGLE_RANGES
    paint_distance = paint_distance if paint_distance is not None else PAINT_DISTANCE

    surfaces = segment_surfaces(mesh, robot_position, angle_ranges, min_facet_area)
    if not surfaces:
        raise RuntimeError('No surfaces passed segmentation/filtering.')

    all_surfaces = []
    skipped = 0

    for surface_id, (face_indices, normal, _) in enumerate(surfaces):
        raw_points, raw_normals = generate_passes_for_surface(
            mesh, face_indices, normal, row_spacing=row_spacing
        )
        if len(raw_points) == 0:
            skipped += 1
            continue

        points, normals = simplify_small_zigzags(
            raw_points,
            raw_normals,
            surface_id=surface_id,
            threshold=SIMPLIFICATION_THRESHOLD,
            logger=logger,
        )

        if paint_distance:
            points = points + normals * paint_distance

        all_surfaces.append({
            'surface_id': surface_id,
            'normal': normal,
            'strokes': [{
                'stroke_id': 0,
                'points': points,
                'normals': normals,
                'entry_index': None,
                'entry_attempts': 0,
            }],
        })

    if logger:
        logger.info(f'PHASE 1+2: {len(all_surfaces)} surfaces generated ({skipped} skipped as degenerate)')
        for s in all_surfaces:
            for stroke in s['strokes']:
                logger.info(f"  surface {s['surface_id']}: {len(stroke['points'])} points")

    return all_surfaces


# ---------------------------------------------------------------------------
# PHASE 3 — face-entry-point reachability search
# Try point[0]; if unreachable, try point[1]; etc, up to max_entry_attempts.
# Stops as soon as a reachable entry point is found — everything after it
# in that face's zigzag is trusted with NO further per-point checking.
# ---------------------------------------------------------------------------

def find_face_entry_points(moveit2, all_surfaces, max_entry_attempts=10, logger=None):
    """
    For each face's stroke, search forward through its points for the
    first one that's IK-reachable and has a collision-free plan. Records
    entry_index (or None if no point within max_entry_attempts worked).
    """
    faces_reachable = 0
    faces_skipped = 0
    total_entry_checks = 0

    for surface in all_surfaces:
        for stroke in surface['strokes']:
            points = stroke['points']
            normals = stroke['normals']
            attempts = min(max_entry_attempts, len(points))

            found_index = None
            for i in range(attempts):
                quat = normal_to_quaternion(normals[i])
                result = check_pose_reachable(moveit2, points[i], quat)
                total_entry_checks += 1

                if result['ik_found'] and result['plan_found']:
                    found_index = i
                    break

            stroke['entry_index'] = found_index
            stroke['entry_attempts'] = attempts if found_index is None else found_index + 1

            if found_index is not None:
                faces_reachable += 1
                if logger:
                    logger.info(
                        f"  surface {surface['surface_id']}: entry found at point "
                        f"{found_index} (tried {found_index + 1} point(s)), "
                        f"{len(points) - found_index} points trusted for this face"
                    )
            else:
                faces_skipped += 1
                if logger:
                    logger.warn(
                        f"  surface {surface['surface_id']}: NO reachable entry point "
                        f"found in first {attempts} point(s) — face skipped entirely"
                    )

    if logger:
        logger.info(
            f'PHASE 3: {faces_reachable} faces have a valid entry point, '
            f'{faces_skipped} faces skipped entirely, {total_entry_checks} total IK/plan checks made'
        )

    return all_surfaces


# ---------------------------------------------------------------------------
# PHASE 4 — build valid_face_segments from entry_index onward
# Points before entry_index are dropped (they were tested and failed, or
# never tested because search stopped early). Points from entry_index
# onward are all included, untested, as one continuous segment.
# ---------------------------------------------------------------------------

def build_face_segments(all_surfaces, logger=None):
    """
    PHASE 4: for every face with a valid entry_index, build one segment
    containing all points from entry_index to the end of that face's
    zigzag — preserving original order, no internal gap-checking.
    """
    valid_face_segments = []
    segment_id = 0

    for surface in all_surfaces:
        for stroke in surface['strokes']:
            if stroke['entry_index'] is None:
                continue

            entry = stroke['entry_index']
            points = stroke['points'][entry:]
            normals = stroke['normals'][entry:]

            valid_face_segments.append({
                'segment_id': segment_id,
                'surface_id': surface['surface_id'],
                'points': points,
                'normals': normals,
                'skipped_entry_points': stroke['points'][:entry],  # for visualization only
            })
            segment_id += 1

    if logger:
        logger.info(f'PHASE 4: {len(valid_face_segments)} valid face segments built')
        for seg in valid_face_segments:
            logger.info(f"  segment {seg['segment_id']} (surface {seg['surface_id']}): {len(seg['points'])} points")

    return valid_face_segments


# ---------------------------------------------------------------------------
# PHASE 5 — transitions ONLY between faces (never checked inside a face)
# Offline approximation, same caveat as ap6: real order-dependent success
# is only verified by actually executing in paintaction_ap5.py.
# ---------------------------------------------------------------------------

def prioritize_face_segments(valid_face_segments, logger=None,
                            z_weight=SEGMENT_PRIORITY_Z_WEIGHT,
                            y_weight=SEGMENT_PRIORITY_Y_WEIGHT):
    
    """Order segments by one combined centroid priority score.

    This is intentionally a single scalar score and not a primary/secondary sort:
        priority = z_weight * normalized_Z + y_weight * (1 - normalized_Y)

    Higher Z and lower Y raise the score. We normalize Y and Z first so the two
    factors are comparable before combining them. A stable tiebreaker on segment_id
    prevents arbitrary flips when two centroids land on the same score.
    """
    if not valid_face_segments:
        return []

    centroids = []
    for seg in valid_face_segments:
        points = np.asarray(seg['points'], dtype=float)
        if len(points) == 0:
            centroids.append(np.array([0.0, 0.0, 0.0]))
        else:
            centroids.append(points.mean(axis=0))

    centroids = np.asarray(centroids, dtype=float)
    y_values = centroids[:, 1]
    z_values = centroids[:, 2]

    y_min = float(np.min(y_values)) if len(y_values) else 0.0
    y_max = float(np.max(y_values)) if len(y_values) else 0.0
    z_min = float(np.min(z_values)) if len(z_values) else 0.0
    z_max = float(np.max(z_values)) if len(z_values) else 0.0

    if len(y_values) > 1 and not np.isclose(y_max, y_min):
        normalized_y = (y_values - y_min) / (y_max - y_min)
    else:
        normalized_y = np.zeros_like(y_values, dtype=float)

    if len(z_values) > 1 and not np.isclose(z_max, z_min):
        normalized_z = (z_values - z_min) / (z_max - z_min)
    else:
        normalized_z = np.zeros_like(z_values, dtype=float)

    priority_scores = z_weight * normalized_z + y_weight * (-(1.0 - normalized_y))

    ranked_indices = sorted(
        range(len(valid_face_segments)),
        key=lambda idx: (
            -float(priority_scores[idx]),
            int(valid_face_segments[idx].get('segment_id', idx)),
        ),
    )
    ordered_segments = [valid_face_segments[i] for i in ranked_indices]

    if logger:
        logger.info(
            f'PHASE 4.5: segment priority ordering with z_weight={z_weight}, '
            f'y_weight={y_weight}'
        )
        for rank, idx in enumerate(ranked_indices, start=1):
            seg = valid_face_segments[idx]
            centroid = centroids[idx]
            score = float(priority_scores[idx])
            logger.info(
                f"  rank {rank}: segment {seg['segment_id']} surface {seg['surface_id']} "
                f"centroid=(y={centroid[1]:.6f}, z={centroid[2]:.6f}) "
                f"priority={score:.6f}"
            )
        logger.info(
            f"PHASE 4.5: final order = {[seg['segment_id'] for seg in ordered_segments]}"
        )

    return ordered_segments


def plan_face_transitions(moveit2, valid_face_segments, logger=None):
    """For each consecutive pair of face segments, check whether MoveIt can
    plan to the next face's entry pose — this is the ONLY collision-aware
    check that happens between faces; nothing inside a face is re-checked.

    The caller supplies the already-prioritized segment order. The internal point
    order within each segment remains unchanged.
    """
    ordered_segments = list(valid_face_segments)
    transitions = []
    successes = 0

    for i in range(len(ordered_segments) - 1):
        seg_a = ordered_segments[i]
        seg_b = ordered_segments[i + 1]

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
            f'PHASE 5: {successes}/{len(transitions)} face-to-face transitions succeeded '
            f'(offline approximation — see paintaction_ap5.py for real execution-order check)'
        )

    return transitions


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_full_pipeline(moveit2, max_entry_attempts=10, logger=None):
    mesh, translation = load_and_prepare_mesh()

    all_surfaces = generate_all_surfaces(mesh, logger=logger)
    all_surfaces = find_face_entry_points(moveit2, all_surfaces, max_entry_attempts, logger=logger)
    valid_face_segments = build_face_segments(all_surfaces, logger=logger)
    prioritized_segments = prioritize_face_segments(valid_face_segments, logger=logger)
    transitions = plan_face_transitions(moveit2, prioritized_segments, logger=logger)

    return mesh, all_surfaces, valid_face_segments, transitions


# ---------------------------------------------------------------------------
# Visualization — reuses ap6's phase1/phase2 plots directly (relabeled);
# phase4/5 needs its own version since the underlying data (entry-index
# based, not per-point reachable arrays) is genuinely different.
# ---------------------------------------------------------------------------

def visualize_face_segments_and_transitions(mesh, valid_face_segments, transitions,
                                             output_path='/tmp/ap5_face_segments.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.08, linewidths=0.1, facecolor='gray'))

    # red: points that were tried as an entry candidate and rejected
    for seg in valid_face_segments:
        skipped = seg['skipped_entry_points']
        if len(skipped) > 0:
            ax.scatter(skipped[:, 0], skipped[:, 1], skipped[:, 2], color='red', marker='x', s=40)

    # segments colored by execution ORDER (not all uniform green anymore) —
    # early segments = dark blue/purple, later segments = bright yellow,
    # using the same viridis-style progression as other visualizations in
    # this project, so "order" reads consistently across all plots
    cmap = plt.get_cmap('viridis')
    n = len(valid_face_segments)

    for order_index, seg in enumerate(valid_face_segments):
        t = order_index / max(n - 1, 1)
        color = cmap(t)
        pts = seg['points']

        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.5)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=12)

        # label the segment's start point with its execution order number
        start = pts[0]
        ax.text(
            start[0], start[1], start[2], f'{order_index + 1}',
            color='black', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='circle,pad=0.2', facecolor=color, edgecolor='black', alpha=0.9),
        )

    # orange: face-to-face transitions only, still shown as before


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------



def main():
    rclpy.init()
    node = Node('cal_pattern_ap5_check')
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

    mesh, all_surfaces, valid_face_segments, transitions = build_full_pipeline(
        moveit2, max_entry_attempts=10, logger=logger
    )

    visualize_all_surfaces(mesh, all_surfaces, output_path='/tmp/ap5_surfaces.png',
                            title=f"Face-entry mode: {len(all_surfaces)} surfaces detected")
    visualize_raw_zigzags(mesh, all_surfaces, output_path='/tmp/ap5_zigzags.png',
                          title="Raw zigzag strokes per face (before entry-point search)")
    visualize_face_segments_and_transitions(mesh, valid_face_segments, transitions)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()