from __future__ import annotations

import numpy as np
import pyvista as pv

# Unit cube vertices (0–1 cube), then we center and scale by density.
# Order: bottom (0,1,2,3), top (4,5,6,7); 0=(0,0,0), 1=(1,0,0), 2=(1,1,0), 3=(0,1,0), etc.
_CUBE_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)
# 12 triangles per cube (outward-facing); each row is (i, j, k) vertex indices.
_CUBE_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 5, 1],
        [0, 4, 5],
        [3, 2, 6],
        [3, 6, 7],
        [0, 3, 7],
        [0, 7, 4],
        [1, 5, 6],
        [1, 6, 2],
    ],
    dtype=np.int64,
)


def voxelize_mesh(mesh: pv.PolyData, density: float = 0.1) -> pv.PolyData:
    """
    Build a voxel grid (mesh of cubes) that fills the interior of a closed surface.

    Math:
    - The mesh is assumed to be a closed surface with a well-defined inside/outside.
    - A regular 3D grid of candidate voxel centers is built over the mesh bounding box,
      with spacing equal to `density` (so smaller density => finer grid, more voxels).
    - For each grid point we decide inside vs outside using a point-in-mesh test
      (PyVista's select_enclosed_points, which uses VTK's ray-casting).
    - Each interior point becomes the center of a cube of edge length `density`.
      Cube vertices are generated in a vectorized way: one template cube (8 vertices,
      12 triangles) is scaled to half-extent density/2, then translated by each
      interior point; face indices are offset per voxel so all cubes are merged
      into one PolyData (single points array, single faces array).
    - The result is a new mesh made only of cubes (triangulated), suitable for
      visualization or further processing.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Closed surface mesh (e.g. brain envelope).
    density : float, optional
        Grid spacing and voxel edge length (same units as the mesh). Default 0.1.
        Smaller values give a finer, heavier voxel grid.

    Returns
    -------
    pyvista.PolyData
        Triangulated mesh of cubes (one cube per interior grid point).
    """
    density = float(density)
    if density <= 0:
        raise ValueError("density must be positive")

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # Regular 3D grid of candidate voxel centers (numpy-optimized).
    x = np.arange(xmin, xmax + density * 0.5, density)
    y = np.arange(ymin, ymax + density * 0.5, density)
    z = np.arange(zmin, zmax + density * 0.5, density)
    if x.size == 0 or y.size == 0 or z.size == 0:
        return pv.PolyData()

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Point-in-mesh: which grid points lie inside the surface?
    grid_pv = pv.PolyData(grid_points)
    enclosed = grid_pv.select_enclosed_points(mesh)
    mask = np.asarray(enclosed["SelectedPoints"], dtype=np.bool_)
    interior = grid_points[mask]

    n_voxels = interior.shape[0]
    if n_voxels == 0:
        return pv.PolyData()

    # Template cube: center at origin, half-extent = density/2.
    half = density / 2.0
    template_verts = (_CUBE_VERTICES - 0.5) * density  # (8, 3)

    # All cube vertices in one array: (n_voxels, 8, 3) -> (n_voxels * 8, 3).
    all_points = (template_verts[np.newaxis, :, :] + interior[:, np.newaxis, :]).reshape(
        -1, 3
    )

    # Global face indices: each cube's 12 triangles use vertices 8*i .. 8*i+7.
    offsets = 8 * np.arange(n_voxels, dtype=np.int64)
    global_faces = _CUBE_FACES[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
    global_faces = global_faces.reshape(-1, 3)

    # PyVista format: [n, i, j, k, n, i, j, k, ...] for each triangle.
    n_tris = global_faces.shape[0]
    counts = np.full((n_tris, 1), 3, dtype=np.int64)
    faces_flat = np.hstack([counts, global_faces]).ravel()

    return pv.PolyData(all_points, faces_flat)
