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
    Load a brain region mesh by name using the atlas (BrainGlobe/Brainrender data).

    Parameters
    ----------
    region_name : str
        Atlas structure acronym (e.g. 'CA1', 'TH', 'CTX').
    atlas_name : str, optional
        Atlas identifier. Default 'allen_mouse_100um'.

    Returns
    -------
    pyvista.PolyData
        Merged mesh for the region (may include multiple subregions).
    """
    atlas = BrainGlobeAtlas(atlas_name)
    if region_name not in atlas.structures:
        raise ValueError(f"Unknown region '{region_name}'. Check atlas structure names.")
    mesh_path = atlas.structures[region_name]["mesh_filename"]
    return pv.read(mesh_path)

