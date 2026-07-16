"""cal_pattern.py — sample a mesh surface, compute normals, and build an ordered path."""

import numpy as np
import trimesh
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def sample_points_with_normals(mesh, num_points=5000, radius=None):
    """
    Sample evenly-spaced points on the mesh surface with their normals.
    Returns:
        points: (N, 3) array of surface positions
        normals: (N, 3) array of corresponding unit surface normals
    """
    points, face_indices = trimesh.sample.sample_surface_even(
        mesh, count=num_points, radius=radius
    )
    normals = mesh.face_normals[face_indices]
    return points, normals
def generate_ordered_path(points, normals, band_height=2.0):
    """
    Order sampled surface points into a raster/zigzag path.
        band_height: thickness of each horizontal band, in the same units as points
    Returns:
        ordered_points: (M, 3) array, M <= N, points sorted into path order
        ordered_normals: (M, 3) array, normals matching ordered_points
    """
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    num_bands = max(1, int(np.ceil((z_max - z_min) / band_height)))

    band_indices = np.clip(
        ((points[:, 2] - z_min) / band_height).astype(int), 0, num_bands - 1
    )

    ordered_points = []
    ordered_normals = []

    for band in range(num_bands):
        mask = band_indices == band
        band_points = points[mask]
        band_normals = normals[mask]

        if len(band_points) == 0:
            continue

        # sort left-to-right along X within this band
        sort_idx = np.argsort(band_points[:, 0])
        band_points = band_points[sort_idx]
        band_normals = band_normals[sort_idx]

        # alternate direction per band for a continuous zigzag sweep
        if band % 2 == 1:
            band_points = band_points[::-1]
            band_normals = band_normals[::-1]

        ordered_points.append(band_points)
        ordered_normals.append(band_normals)

    return np.vstack(ordered_points), np.vstack(ordered_normals)

def main():
    scale = 0.01  # matches obj.urdf's <mesh scale="..."/>

    mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    points, normals = sample_points_with_normals(mesh, num_points=2000)
    print(f'Sampled {len(points)} points (native units)')

    # convert points to real-world meters BEFORE building the path
    points = points * scale
    # normals are directions, not positions — scaling would distort them, so leave as-is
    # (they'll need to stay unit vectors; more on this when we build robot poses)

    ordered_points, ordered_normals = generate_ordered_path(points, normals, band_height=0.05)
    print(f'Ordered into {len(ordered_points)} path points (meters)')
    print(f'First 3 ordered points:\n{ordered_points[:3]}')
    print(f'Last 3 ordered points:\n{ordered_points[-3:]}')

if __name__ == '__main__':
    main()