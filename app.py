from __future__ import annotations

from typing import Tuple

import numpy as np
import streamlit as st
import pyvista as pv
import plotly.graph_objects as go

from src.brain_data import get_region_acronyms, load_region_mesh, load_root_brain_mesh

HIGHLIGHT_COLORS = ["blue", "crimson", "mediumseagreen"]


@st.cache_data
def get_cached_root_mesh() -> pv.PolyData:
    """Load the root (whole brain) mesh (cached to avoid repeated disk reads)."""
    return load_root_brain_mesh()


@st.cache_data
def get_cached_region_mesh(region_name: str) -> pv.PolyData:
    """Load a region mesh (cached to avoid repeated disk reads)."""
    return load_region_mesh(region_name)


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
    st.set_page_config(page_title="Neuro-Canvas: Glass Brain", layout="wide")

    st.title("Neuro-Canvas")
    st.subheader("Glass brain context view")

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

    fig = go.Figure()

    # Background context: whole brain (root)
    with st.spinner("Loading whole brain..."):
        try:
            root_mesh = get_cached_root_mesh()
            verts, faces = extract_plotly_data(root_mesh)
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color="lightgrey",
                    opacity=0.1,
                    name="Whole Brain",
                    hoverinfo="skip",
                )
            )
        except Exception as exc:
            st.error(f"Failed to load whole brain: {exc}")
            st.stop()

    # Highlights: selected regions
    for i, region in enumerate(selected_regions):
        try:
            mesh = get_cached_region_mesh(region)
            verts, faces = extract_plotly_data(mesh)
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
                    name=region,
                )
            )
        except Exception as exc:
            st.error(f"Error loading {region}: {exc}")

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
