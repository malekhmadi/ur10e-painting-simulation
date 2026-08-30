"""cal_pattern_ap6.py — generate paintable strokes grouped by surface, with
per-point reachability filtering. Reuses segmentation/pass-generation and
reachability checking from cal_pattern_ap4; does not duplicate them."""

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
    visualize_path,
    visualize_reachability,
)


def build_strokes(mesh, row_spacing=ROW_SPACING, robot_position=ROBOT_POSITION,
                   angle_ranges=ANGLE_RANGES, paint_distance=PAINT_DISTANCE,
                   min_facet_area=1e-6):
    """
    Steps 1-2: segment the mesh into paintable surfaces and generate a
    zigzag stroke per surface. Unlike generate_surface_based_path, this
    keeps each surface's stroke SEPARATE (not concatenated), since later
    steps need to reason about strokes individually.

    Returns:
        list of (points, normals) tuples, one per surface, ordered
        nearest-to-robot first (same ordering segment_surfaces already provides)
    """
    surfaces = segment_surfaces(mesh, robot_position, angle_ranges, min_facet_area)
    if not surfaces:
        raise RuntimeError('No surfaces passed segmentation/filtering.')

    strokes = []
    for face_indices, normal, _ in surfaces:
        points, normals = generate_passes_for_surface(mesh, face_indices, normal, row_spacing=row_spacing)
        if len(points) == 0:
            continue
        if paint_distance:
            points = points + normals * paint_distance
        strokes.append((points, normals))

    if not strokes:
        raise RuntimeError('No strokes were generated from any surface.')

    return strokes


def filter_strokes_by_reachability(moveit2, strokes, min_points_per_stroke=5, logger=None):
    """
    Step 3: check every point in every stroke against MoveIt (IK + collision-
    free plan). Points that fail are dropped, but strokes stay grouped
    (step 4) — a stroke with too few surviving points is dropped entirely
    rather than left as a near-useless single dot.

    Returns:
        filtered_strokes: list of (points, normals), reachable points only,
                          strokes with < min_points_per_stroke removed
        dropped_points, dropped_normals: everything that failed the check,
                          flattened, for visualization/debugging
    """
    filtered_strokes = []
    dropped_points, dropped_normals = [], []
    total_checked = 0
    total_points = sum(len(p) for p, _ in strokes)

    for stroke_points, stroke_normals in strokes:
        kept_points, kept_normals = [], []
        for point, normal in zip(stroke_points, stroke_normals):
            quat = normal_to_quaternion(normal)
            result = check_pose_reachable(moveit2, point, quat)
            total_checked += 1

            if result['ik_found'] and result['plan_found']:
                kept_points.append(point)
                kept_normals.append(normal)
            else:
                dropped_points.append(point)
                dropped_normals.append(normal)

            if logger and (total_checked % 25 == 0 or total_checked == total_points):
                logger.info(f'Checked {total_checked}/{total_points} points')

        if len(kept_points) >= min_points_per_stroke:
            filtered_strokes.append((np.array(kept_points), np.array(kept_normals)))
        elif logger and len(kept_points) > 0:
            logger.info(f'Dropping stroke: only {len(kept_points)} reachable point(s), below minimum {min_points_per_stroke}')

    to_array = lambda lst: np.array(lst) if lst else np.empty((0, 3))
    return filtered_strokes, to_array(dropped_points), to_array(dropped_normals)


def load_and_prepare_mesh():
    """Load obj.stl, apply URDF scale, launch-file rotation, AND translation,
    so the mesh matches world coordinates exactly like the generated points do."""
    scale = read_mesh_scale_from_urdf()

    mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.apply_scale(scale)

    translation, (roll, pitch, yaw) = read_object_pose_from_launch()
    quat = euler_to_quaternion(roll, pitch, yaw)
    mesh.apply_transform(quaternion_to_matrix4(quat))
    mesh.apply_translation(translation)   # <-- add this line

    return mesh, translation

def build_reachable_strokes(moveit2, logger=None):
    """Full steps 1-4 pipeline: load mesh, generate strokes, filter by
    reachability, translate to world coordinates. Returns
    (filtered_strokes, dropped_points, dropped_normals)."""
    mesh, translation = load_and_prepare_mesh()

    strokes = build_strokes(mesh)
    strokes = [(points + translation, normals) for points, normals in strokes]

    filtered_strokes, dropped_points, dropped_normals = filter_strokes_by_reachability(
        moveit2, strokes, logger=logger
    )
    

    return filtered_strokes, dropped_points, dropped_normals

def add_object_to_planning_scene(moveit2, logger=None):
    """
    Step 6: register the object mesh as a real collision object in
    MoveIt's planning scene, so reachability/plan checks correctly account
    for the object as an obstacle. Shared by both the dataset generator
    and the live execution node, so both check against the same reality.
    """
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

        
def add_object_to_planning_scene(moveit2, logger=None):
    """
    Step 6: register the object mesh as a real collision object in
    MoveIt's planning scene, so reachability/plan checks correctly account
    for the object as an obstacle. Shared by the dataset generator,
    the standalone visualization check, and live execution.
    """
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

def visualize_strokes(mesh, strokes, output_path='/tmp/strokes_grouped.png'):
    """Show each surviving stroke in a distinct color, so stroke grouping
    (not just individual point reachability) is visible."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_collection = Poly3DCollection(mesh.triangles, alpha=0.1, linewidths=0.1, facecolor='gray')
    ax.add_collection3d(mesh_collection)

    cmap = plt.get_cmap('tab20')
    for i, (points, normals) in enumerate(strokes):
        color = cmap(i % 20)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2, label=f'stroke {i}')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=15)
        ax.scatter(*points[0], color=color, s=80, marker='o', edgecolors='black')   # stroke start
        ax.scatter(*points[-1], color=color, s=80, marker='X', edgecolors='black')  # stroke end

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Strokes grouped by surface — {len(strokes)} strokes (○=start, ✕=end)")
    if len(strokes) <= 15:
        ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.05, 1))

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    max_range = (bounds[1] - bounds[0]).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")


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

    add_object_to_planning_scene(moveit2, logger=node.get_logger())

    mesh, translation = load_and_prepare_mesh()

    print("Generating strokes...")
    raw_strokes = build_strokes(mesh)
    total_points = sum(len(p) for p, _ in raw_strokes)
    print(f"Generated {len(raw_strokes)} strokes, {total_points} candidate points total")

    print("Checking reachability (this can take a while)...")
    filtered_strokes, dropped_points, dropped_normals = filter_strokes_by_reachability(
        moveit2, raw_strokes, logger=node.get_logger()
    )
    print(f"{len(filtered_strokes)}/{len(raw_strokes)} strokes survived "
          f"({len(dropped_points)} individual points dropped as unreachable)")

    # flatten filtered strokes for the ap4 visualizations, which expect flat arrays
    all_reachable_points = np.vstack([p for p, _ in filtered_strokes]) if filtered_strokes else np.empty((0, 3))
    all_reachable_normals = np.vstack([n for _, n in filtered_strokes]) if filtered_strokes else np.empty((0, 3))

    visualize_path(mesh, all_reachable_points, all_reachable_normals, '/tmp/surface_path.png')
    visualize_reachability(mesh, all_reachable_points, dropped_points, output_path='/tmp/reachability.png')
    visualize_strokes(mesh, filtered_strokes, output_path='/tmp/strokes_grouped.png')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()