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

st.set_page_config(
    page_title="Voxelith",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;600&display=swap');

    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 300 !important;
    }

    html, body, [class*='css'] {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }

    [data-testid="stSidebar"], [class*='stSelectbox'], [class*='stMultiSelect'],
    [class*='stCheckbox'], [class*='stToggle'], button, input, [class*='stTextInput'] {
        border-radius: 0px !important;
    }

    [class*='stSelectbox'] > div, [class*='stMultiSelect'] > div,
    input, [data-baseweb="input"] {
        border-color: #222222 !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    span[data-baseweb='tag'] {
        background-color: #ffffff !important;
        border: 1px solid #ffffff !important;
        border-radius: 0px !important;
    }
    span[data-baseweb='tag'] span {
        color: #000000 !important;
    }
    div[role='listbox'] {
        background-color: #111111 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PATHOLOGY_MAP = {
    "None": [],
    "Ischemic Stroke (MCA)": ["STR", "MOs", "SSp"],
    "Glioblastoma (Butterfly)": ["CC", "MOs", "SSp"],
    "Thalamic Hemorrhage": ["TH"],
}
PATHOLOGY_COLORS = {
    "Ischemic Stroke (MCA)": "#4B0082",
    "Thalamic Hemorrhage": "#4B0082",
    "Glioblastoma (Butterfly)": "#FF4500",
}

HIGHLIGHT_COLORS = [
    "#00FFFF",   # Cyan
    "#FF00FF",   # Magenta
    "#7DF9FF",   # Light Teal
    "#B026FF",   # Neon Purple
    "#FF007F",   # Hot Pink
    "#00FF9D",   # Neon Green
]


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
    st.markdown(
        """
        <div style="padding: 1rem 0; margin-bottom: 1.5rem; text-align: center;">
            <h1 style="font-family: 'Playfair Display', serif; font-size: 4rem; font-weight: 300; color: #ffffff; margin: 0; letter-spacing: 5px;">Voxelith</h1>
            <p style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #888; margin: 0.5rem 0 0 0; letter-spacing: 2px; text-transform: lowercase;">Volumetric Anatomical Matrix & Spatial Intelligence Engine</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Control Panel")
        st.divider()
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
        st.subheader("2. Render Engine")
        show_glass_brain = st.toggle("Show Glass Brain Context", value=True)
        use_voxels = st.toggle("Enable Voxel Data View", value=False)
        use_heatmap = st.toggle("Enable Density Heatmap", value=False)

        st.subheader("3. Research & Clinical Mode")
        pathology_preset = st.selectbox(
            "Simulate Clinical Pathology",
            options=list(PATHOLOGY_MAP.keys()),
            help="Add preset regions for common pathologies.",
        )
        pathology_acronyms = PATHOLOGY_MAP.get(pathology_preset, [])
        regions_to_render = list(
            dict.fromkeys(selected_regions + [a for a in pathology_acronyms if a in region_options])
        )
        lesion_mode = st.toggle("Simulate Pathological Lesion", value=False)
        lesion_severity = 0.5
        if lesion_mode:
            lesion_severity = st.slider(
                "Lesion Severity",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0.1 = Blood Red, 1.0 = Necrotic Black",
            )

        total_volume_mm3 = 0.0
        total_pathological_volume_mm3 = 0.0
        total_lesion_volume_mm3 = 0.0
        total_voxel_count = 0
        effective_resolution_um = None
        if regions_to_render:
            for region in regions_to_render:
                region_mesh = get_cached_region_mesh(region)
                vol_um3 = region_mesh.volume
                if use_voxels:
                    x_length = region_mesh.bounds[1] - region_mesh.bounds[0]
                    voxel_size = x_length / 30.0
                    vox = region_mesh.voxelize(spacing=voxel_size)
                    total_voxel_count += vox.n_cells
                    if effective_resolution_um is None:
                        effective_resolution_um = voxel_size
                    vol_um3 = vox.n_cells * (voxel_size ** 3)
                total_volume_mm3 += vol_um3 / 1e9
                if pathology_preset != "None" and region in pathology_acronyms:
                    total_pathological_volume_mm3 += vol_um3 / 1e9
                is_lesion_rendered = (
                    not use_heatmap
                    and (pathology_preset == "None" or region not in pathology_acronyms)
                )
                if lesion_mode and is_lesion_rendered:
                    total_lesion_volume_mm3 += vol_um3 / 1e9
        if lesion_mode and regions_to_render:
            st.metric("Estimated Lesion Volume", f"{total_lesion_volume_mm3:.2f} mm³")
        if pathology_preset != "None" and pathology_acronyms:
            st.metric("Total Pathological Volume", f"{total_pathological_volume_mm3:.2f} mm³")
        if use_voxels and regions_to_render:
            st.divider()
            st.subheader("Voxel metrics")
            st.metric("Voxel Count", f"{total_voxel_count:,}")
            if effective_resolution_um is not None:
                st.metric("Effective Resolution", f"{effective_resolution_um:.1f} µm")
            st.metric("Total Volume", f"{total_volume_mm3:.2f} mm³")

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

    # Highlights: selected regions + pathology preset
    for i, region in enumerate(regions_to_render):
        try:
            mesh = get_cached_region_mesh(region)
            if use_voxels:
                mesh = get_voxelized_surface(mesh)
            verts, faces = extract_plotly_data(mesh)
            full_name = get_cached_region_name(region)
            wrapped_name = "<br>".join(textwrap.wrap(full_name, width=30))
            if use_heatmap:
                center = np.array(mesh.center)
                distances = np.linalg.norm(verts - center, axis=1)
                d_max = distances.max()
                intensity = (1.0 - (distances / d_max)) if d_max > 0 else np.ones_like(distances)
                mesh_trace = go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    intensity=intensity,
                    colorscale="Viridis",
                    showscale=True,
                    opacity=0.85,
                    name=full_name,
                    hovertext=wrapped_name,
                    hoverinfo="text",
                    lighting=dict(
                        ambient=0.6,
                        diffuse=0.5,
                        roughness=0.1,
                        specular=0.8,
                        fresnel=0.5,
                    ),
                )
            else:
                if pathology_preset != "None" and region in pathology_acronyms:
                    color = PATHOLOGY_COLORS.get(
                        pathology_preset, HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
                    )
                elif lesion_mode:
                    t = (lesion_severity - 0.1) / 0.9
                    r = int(139 + (26 - 139) * t)
                    g = int(0 + (26 - 0) * t)
                    b_val = int(0 + (26 - 0) * t)
                    color = f"rgb({r},{g},{b_val})"
                else:
                    color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
                mesh_trace = go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.85,
                    name=full_name,
                    hovertext=wrapped_name,
                    hoverinfo="text",
                    lighting=dict(
                        ambient=0.6,
                        diffuse=0.5,
                        roughness=0.1,
                        specular=0.8,
                        fresnel=0.5,
                    ),
                )
            fig.add_trace(mesh_trace)
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
