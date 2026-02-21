from __future__ import annotations

from typing import Sequence, Union
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import numpy as np
import pyvista as pv
from brainrender import Scene

def _vedo_mesh_to_pyvista(actor):
    """Converts a brainrender (vedo) actor to a PyVista PolyData object."""
    
    # Brainrender sometimes returns a list of actors. If so, grab the first one.
    if isinstance(actor, list):
        actor = actor[0]
        
    # Both Vedo and PyVista run on VTK. 
    # We can just extract the underlying VTK dataset and wrap it instantly!
    try:
        # Newer versions of vedo/brainrender
        if hasattr(actor, 'polydata'):
            vtk_mesh = actor.polydata() 
            return pv.wrap(vtk_mesh)
        elif hasattr(actor, 'dataset'):
            return pv.wrap(actor.dataset)
        else:
            # Ultimate fallback if it's an older vedo object
            return pv.wrap(actor._polydata)
            
    except Exception as e:
        raise RuntimeError(f"Could not extract VTK data from Brainrender actor: {e}")



def get_region_acronyms(atlas_name: str = "allen_mouse_100um") -> list[str]:
    """
    Return a sorted list of all valid region acronyms from the atlas.

    Iterates through atlas.structures.values() and collects the 'acronym'
    field for each structure.
    """
    atlas = BrainGlobeAtlas(atlas_name)
    acronyms = sorted({s["acronym"] for s in atlas.structures.values()})
    return acronyms


def load_root_brain_mesh(atlas_name="allen_mouse_100um"):
    """
    Directly loads the atlas mesh as a PyVista object,
    completely bypassing Brainrender and Vedo!
    """
    atlas = BrainGlobeAtlas(atlas_name)
    mesh_path = atlas.structures["root"]["mesh_filename"]
    return pv.read(mesh_path)


def load_region_mesh(region_name: str, atlas_name: str = "allen_mouse_100um") -> pv.PolyData:
    """
    Load a brain region mesh by acronym using the atlas (BrainGlobe/Brainrender data).

    Parameters
    ----------
    region_name : str
        Atlas structure acronym (e.g. 'CA1', 'MY-mot').
    atlas_name : str, optional
        Atlas identifier. Default 'allen_mouse_100um'.

    Returns
    -------
    pyvista.PolyData
        Mesh for the region.

    Raises
    ------
    ValueError
        If no structure has the given acronym.
    """
    atlas = BrainGlobeAtlas(atlas_name)
    for structure in atlas.structures.values():
        if structure["acronym"] == region_name:
            mesh_path = structure["mesh_filename"]
            return pv.read(mesh_path)
    raise ValueError(f"No atlas structure with acronym '{region_name}'.")


def get_region_full_name(region_acronym: str, atlas_name: str = "allen_mouse_100um") -> str:
    """
    Return the full display name for a region acronym.

    Iterates atlas.structures.values() to find the structure with matching acronym,
    then returns structure['name']. Falls back to the acronym if not found.
    """
    atlas = BrainGlobeAtlas(atlas_name)
    for structure in atlas.structures.values():
        if structure.get("acronym") == region_acronym:
            return structure.get("name", region_acronym)
    return region_acronym

