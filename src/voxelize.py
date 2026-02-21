from __future__ import annotations

import pyvista as pv


def voxelize_mesh(mesh: pv.PolyData) -> pv.UnstructuredGrid:
    """
    Voxelize a closed surface mesh using a density derived from its extent.

    The voxel size is computed from the mesh bounding box so that the result
    is tractable regardless of coordinate units (e.g. micrometers). The
    X-axis length (max_x - min_x) from mesh.bounds is divided by 30 to get
    the voxel spacing, giving roughly 30 voxels along the longest axis.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Closed surface mesh (e.g. brain region).

    Returns
    -------
    pyvista.UnstructuredGrid
        Voxel grid filling the mesh interior.
    """
    bounds = mesh.bounds
    x_length = bounds[1] - bounds[0]
    if x_length <= 0:
        raise ValueError("Mesh has degenerate X extent.")
    voxel_size = x_length / 30.0
    return mesh.voxelize(spacing=voxel_size)
