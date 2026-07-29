"""cal_pattern_ap5.py — DBSCAN-based ML surface segmentation for painting
path generation. Instead of trimesh.facets (pure coplanar-triangle grouping),
this clusters sampled surface points using both position and normal
similarity, letting density-based clustering discover surface regions."""

from pathlib import Path

import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
from ament_index_python.packages import get_package_share_directory

# reuse shared utilities already built and tested in ap4 — single source of truth
from motion_control.cal_patten_ap4 import (
    euler_to_quaternion,
    quaternion_to_matrix4,
    normal_to_quaternion,
    read_mesh_scale_from_urdf,
    read_object_pose_from_launch,
    get_tangent_frame,
    connect_surfaces_smoothly,

)


# ---------------------------------------------------------------------------
# Shared painting parameters — same convention as ap4
# ---------------------------------------------------------------------------
ROW_SPACING = 0.05
ROBOT_POSITION = np.array([0.0, 0.0, -1.0])
ANGLE_RANGES = ((90.0, 110.0),)
LIFT_DISTANCE = 0.03
MARGIN_DISTANCE = 0.005

NUM_SAMPLE_POINTS = 1000
DBSCAN_EPS = 0.08          # max feature-space distance for two points to be neighbors
DBSCAN_MIN_SAMPLES = 8     # min points to form a dense cluster
SPATIAL_WEIGHT = 1.0       # relative importance of position in clustering
NORMAL_WEIGHT = 2.0        # relative importance of normal direction (higher = stricter "flatness")


# ---------------------------------------------------------------------------
# Step 1 — ML-based surface segmentation via DBSCAN
# ---------------------------------------------------------------------------

def segment_surfaces_dbscan(mesh, num_points=NUM_SAMPLE_POINTS,
                             eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                             spatial_weight=SPATIAL_WEIGHT, normal_weight=NORMAL_WEIGHT):
    """
    Sample points across the mesh surface, then cluster them with DBSCAN
    using a combined [position, normal] feature space — points close together
    in space AND facing a similar direction get grouped into the same
    surface region. Points DBSCAN labels as noise (-1) are discarded.

    Returns:
        clusters: list of (points, normals) tuples, one per discovered cluster,
                  each already representing a paintable surface region
    """
    points, face_indices = trimesh.sample.sample_surface_even(mesh, count=num_points)
    normals = mesh.face_normals[face_indices]

    # normalize each feature block independently before weighting, so position
    # (meters) and normal (unit vectors, already -1..1) contribute comparably
    pos_scale = mesh.bounding_box.extents.max()
    features = np.hstack([
        (points / pos_scale) * spatial_weight,
        normals * normal_weight,
    ])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_

    unique_labels = sorted(set(labels) - {-1})  # drop noise label
    clusters = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = points[mask]
        cluster_normals = normals[mask]
        clusters.append((cluster_points, cluster_normals))

    return clusters


def filter_and_order_clusters(clusters, robot_position=None, angle_ranges=((0.0, 90.0),)):
    """
    Compute each cluster's mean normal + centroid, apply the same angle-range
    facing filter used in ap4, and order clusters nearest-to-robot first.
    """
    kept = []
    for points, normals in clusters:
        mean_normal = normals.mean(axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        centroid = points.mean(axis=0)

        distance = None
        if robot_position is not None:
            to_robot = robot_position - centroid
            distance = np.linalg.norm(to_robot)
            to_robot_dir = to_robot / distance if distance > 1e-9 else to_robot

            cos_alignment = np.clip(np.dot(mean_normal, to_robot_dir), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_alignment))

            if not any(lo <= angle_deg <= hi for lo, hi in angle_ranges):
                continue

        kept.append((points, normals, mean_normal, centroid, distance))

    if robot_position is not None:
        kept.sort(key=lambda item: item[4])

    return kept


# ---------------------------------------------------------------------------
# Step 2 & 3 — painting direction + parallel passes per cluster
# ---------------------------------------------------------------------------

def generate_passes_for_cluster(points, mean_normal, row_spacing=0.05, u_resolution=0.005):
    """
    Build a zigzag raster across a DBSCAN cluster's point cloud, using PCA
    on the in-plane projection to find the cluster's long axis (stroke
    direction), same approach as ap4's per-facet version, adapted to work
    on a scattered point cluster instead of a connected triangle patch.
    """
    origin = points.mean(axis=0)
    u0, v0 = get_tangent_frame(mean_normal)

    relative = points - origin
    coords_2d = np.column_stack([relative @ u0, relative @ v0])

    if len(coords_2d) < 3:
        return np.empty((0, 3)), np.empty((0, 3))

    cov = np.cov(coords_2d.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis_2d = eigvecs[:, np.argmax(eigvals)]
    long_axis_3d = long_axis_2d[0] * u0 + long_axis_2d[1] * v0

    v_axis, u_axis = get_tangent_frame(mean_normal, u_hint=long_axis_3d)

    u = relative @ u_axis
    v = relative @ v_axis
    points_2d = np.column_stack([u, v])

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
    except Exception:
        # degenerate cluster (collinear points, too few for a hull) — skip
        return np.empty((0, 3)), np.empty((0, 3))

    polygon = MplPath(hull_points)

    v_min, v_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    u_min, u_max = points_2d[:, 0].min(), points_2d[:, 0].max()

    num_rows = max(1, int(round((v_max - v_min) / row_spacing)))
    u_samples = np.arange(u_min, u_max + u_resolution, u_resolution)

    ordered_points = []
    for i in range(num_rows + 1):
        v_level = v_min + i * row_spacing
        test_points = np.column_stack([u_samples, np.full_like(u_samples, v_level)])
        inside = polygon.contains_points(test_points)

        if not inside.any():
            continue

        u_inside = u_samples[inside]
        u_start, u_end = u_inside.min(), u_inside.max()

        if i % 2 == 1:
            u_start, u_end = u_end, u_start

        for uu in (u_start, u_end):
            point_3d = origin + uu * u_axis + v_level * v_axis
            ordered_points.append(point_3d)

    ordered_points = np.array(ordered_points)
    if len(ordered_points) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    normals_out = np.tile(mean_normal, (len(ordered_points), 1))
    return ordered_points, normals_out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def generate_dbscan_path(mesh, num_points=NUM_SAMPLE_POINTS, eps=DBSCAN_EPS,
                          min_samples=DBSCAN_MIN_SAMPLES, row_spacing=0.05,
                          robot_position=None, angle_ranges=((0.0, 90.0),),
                          lift_distance=0.03, margin_distance=0.0):
    """Full DBSCAN-based pipeline: sample -> cluster -> filter/order ->
    per-cluster passes -> smooth connection -> one continuous trajectory."""
    clusters = segment_surfaces_dbscan(mesh, num_points=num_points, eps=eps, min_samples=min_samples)
    if not clusters:
        raise RuntimeError('DBSCAN found no clusters — try increasing eps or lowering min_samples.')

    print(f'DBSCAN discovered {len(clusters)} raw clusters')

    filtered = filter_and_order_clusters(clusters, robot_position, angle_ranges)
    if not filtered:
        raise RuntimeError('No clusters passed angle-range filtering — check angle_ranges or robot_position.')

    print(f'{len(filtered)} clusters passed filtering, ordered nearest-to-robot first')

    surface_paths = []
    for points, normals, mean_normal, centroid, distance in filtered:
        path_points, path_normals = generate_passes_for_cluster(points, mean_normal, row_spacing=row_spacing)
        if len(path_points) == 0:
            continue
        if margin_distance:
            path_points = path_points + path_normals * margin_distance
        surface_paths.append((path_points, path_normals))

    if not surface_paths:
        raise RuntimeError('No clusters produced a usable path.')

    return connect_surfaces_smoothly(surface_paths, lift_distance=lift_distance)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_clusters(mesh, clusters, output_path='/tmp/dbscan_clusters.png'):
    """Scatter-plot each DBSCAN cluster in a different color, over the mesh,
    to sanity-check the segmentation before generating paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_collection = Poly3DCollection(mesh.triangles, alpha=0.1, linewidths=0.1, facecolor='gray')
    ax.add_collection3d(mesh_collection)

    cmap = plt.get_cmap('tab20')
    for i, (points, normals) in enumerate(clusters):
        color = cmap(i % 20)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=8, label=f'cluster {i}')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"DBSCAN surface segmentation — {len(clusters)} clusters found")
    if len(clusters) <= 15:
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


def visualize_path(mesh, points, normals, output_path='/tmp/dbscan_path.png', normal_length=0.03):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_collection = Poly3DCollection(mesh.triangles, alpha=0.15, linewidths=0.2)
    ax.add_collection3d(mesh_collection)

    ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=1.5, label="trajectory")

    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=np.arange(len(points)), cmap='viridis', s=15
    )
    fig.colorbar(scatter, ax=ax, label="path order")

    step = max(1, len(points) // 60)
    ax.quiver(
        points[::step, 0], points[::step, 1], points[::step, 2],
        normals[::step, 0], normals[::step, 1], normals[::step, 2],
        length=normal_length, color='red', normalize=True
    )

    ax.scatter(*points[0], s=100, marker='o', color='green', label="START")
    ax.scatter(*points[-1], s=100, marker='X', color='black', label="END")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("DBSCAN-segmented painting path")
    ax.legend()

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    max_range = (bounds[1] - bounds[0]).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
    print(f"Loading mesh: {mesh_path}")

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    scale = read_mesh_scale_from_urdf()
    mesh.apply_scale(scale)

    translation, (roll, pitch, yaw) = read_object_pose_from_launch()
    quat = euler_to_quaternion(roll, pitch, yaw)
    mesh.apply_transform(quaternion_to_matrix4(quat))
    print(f"Applied rotation from launch file: roll={roll}, pitch={pitch}, yaw={yaw}")

    # visualize raw clustering first, before filtering/path generation,
    # to sanity-check DBSCAN parameters
    clusters = segment_surfaces_dbscan(mesh)
    visualize_clusters(mesh, clusters, output_path='/tmp/dbscan_clusters.png')

    points, normals = generate_dbscan_path(
        mesh,
        row_spacing=ROW_SPACING,
        robot_position=ROBOT_POSITION,
        angle_ranges=ANGLE_RANGES,
        lift_distance=LIFT_DISTANCE,
        margin_distance=MARGIN_DISTANCE,
    )

    print(f"Generated {len(points)} total path points across all clusters")

    visualize_path(mesh, points, normals, '/tmp/dbscan_path.png')


if __name__ == '__main__':
    main()