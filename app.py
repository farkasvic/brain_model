from __future__ import annotations

import textwrap
from typing import Tuple

import numpy as np
import streamlit as st
import pyvista as pv
import plotly.graph_objects as go

from src.brain_data import (
    get_region_acronyms,
    get_region_full_name,
    load_region_mesh,
    load_root_brain_mesh,
)

HIGHLIGHT_COLORS = ["blue", "crimson", "mediumseagreen"]


@st.cache_data
def get_cached_root_mesh() -> pv.PolyData:
    """Load the root (whole brain) mesh (cached to avoid repeated disk reads)."""
    return load_root_brain_mesh()


@st.cache_data
def get_cached_region_mesh(region_name: str) -> pv.PolyData:
    """Load a region mesh (cached to avoid repeated disk reads)."""
    return load_region_mesh(region_name)


@st.cache_data
def get_cached_region_name(region_acronym: str) -> str:
    """Full display name for a region acronym (cached)."""
    return get_region_full_name(region_acronym)


def get_voxelized_surface(mesh: pv.PolyData, resolution: int = 30) -> pv.PolyData:
    """
    Voxelize a mesh and return its outer surface as triangulated PolyData for Plotly.

    Density is computed from the mesh X extent and resolution (voxels along that axis).
    """
    x_length = mesh.bounds[1] - mesh.bounds[0]
    density = x_length / resolution
    vox = mesh.voxelize(spacing=density)
    return vox.extract_geometry().triangulate()


def extract_plotly_data(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract vertices and faces from a PyVista mesh for Plotly Mesh3d.

    Returns
    -------
    vertices : np.ndarray
        mesh.points after triangulate().
    faces : np.ndarray
        mesh.faces reshaped to (n_triangles, 3) vertex indices.
    """
    mesh = mesh.triangulate()
    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    return vertices, faces


def main() -> None:
    st.set_page_config(page_title="Neuro-Model: Glass Brain", layout="wide")

    st.title("Neuro-Model")
    st.subheader("Glass brain 3D view")

    with st.sidebar:
        st.header("Controls")
        if "region_acronyms" not in st.session_state:
            with st.spinner("Loading atlas..."):
                try:
                    st.session_state["region_acronyms"] = get_region_acronyms()
                except Exception as exc:
                    st.error(f"Failed to load atlas: {exc}")
                    st.stop()
        region_options = st.session_state["region_acronyms"]
        selected_regions = st.multiselect(
            "Brain Regions",
            options=region_options,
            default=[],
            help="Select regions to highlight on the glass brain.",
        )
        show_glass_brain = st.toggle("Show Glass Brain Context", value=True)
        use_voxels = st.toggle("Enable Voxel Data View", value=False)

    fig = go.Figure()

    # Background context: whole brain (root) — only when toggled on (avoids blocking hover on inner regions)
    if show_glass_brain:
        with st.spinner("Loading whole brain..."):
            try:
                root_mesh = get_cached_root_mesh()
                if use_voxels:
                    root_mesh = get_voxelized_surface(root_mesh)
                verts, faces = extract_plotly_data(root_mesh)
                fig.add_trace(
                    go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color="#B0BEC5",
                        opacity=0.25,
                        name="Whole Brain",
                        hoverinfo="skip",
                        lighting=dict(
                            ambient=0.5,
                            diffuse=0.4,
                            fresnel=2.0,
                            specular=0.5,
                            roughness=0.1,
                        ),
                    )
                )
            except Exception as exc:
                st.error(f"Failed to load whole brain: {exc}")
                st.stop()

    # Highlights: selected regions (full name in legend and hover via name + hoverinfo='name')
    for i, region in enumerate(selected_regions):
        try:
            mesh = get_cached_region_mesh(region)
            if use_voxels:
                mesh = get_voxelized_surface(mesh)
            verts, faces = extract_plotly_data(mesh)
            full_name = get_cached_region_name(region)
            wrapped_name = "<br>".join(textwrap.wrap(full_name, width=30))
            color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=1.0,
                    name=full_name,
                    hovertext=wrapped_name,
                    hoverinfo="text",
                )
            )
        except Exception as exc:
            st.error(f"Error loading {region}: {exc}")

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False,
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False,
            ),
            zaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False,
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
