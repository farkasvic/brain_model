# Voxelith

**Volumetric Anatomical Matrix & Spatial Intelligence Engine**

A Streamlit-based 3D brain visualization tool powered by the Allen Mouse Brain Atlas. Voxelith renders interactive glass-brain views with region highlighting, voxelization, pathology simulation, and density heatmaps—suitable for research, teaching, and clinical exploration.

---

## Features

- **Glass brain context** — Transparent whole-brain overlay with optional toggle for unobstructed region interaction
- **Region selection** — Multiselect from dynamically loaded Allen atlas acronyms (CA1, TH, HIP, and more)
- **Voxel mode** — Convert mesh surfaces to voxel grids with metrics (count, resolution, volume)
- **Pathology simulator** — Preset clinical pathologies: Ischemic Stroke (MCA), Glioblastoma (Butterfly), Thalamic Hemorrhage
- **Lesion mode** — Simulate pathological lesions with severity slider (blood red → necrotic black)
- **Density heatmap** — Vertex-intensity coloring by distance from centroid (Viridis colormap)
- **Volume metrics** — Estimated lesion volume, pathological volume, and total voxel volume in mm³

---

## Installation

```bash
git clone https://github.com/your-username/voxelith.git
cd voxelith
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- Streamlit, PyVista, Plotly, BrainGlobe Atlas API, NumPy

---

## Usage

```bash
streamlit run app.py
```

Open the app in your browser. Use the sidebar to:

1. **Select brain regions** from the multiselect
2. **Toggle the glass brain** for context or clearer hover interaction
3. **Enable voxel view** for discretized mesh and metrics
4. **Choose a pathology preset** to auto-add affected regions
5. **Enable lesion mode** to simulate damage with adjustable severity
6. **Enable density heatmap** for centroid-distance visualization

---

## Project structure

```
voxelith/
├── app.py              # Streamlit UI and Plotly rendering
├── src/
│   ├── brain_data.py   # Atlas loading, region mesh & acronym lookup
│   └── voxelize.py     # Mesh voxelization helpers
├── requirements.txt
└── README.md
```

---

## Tech stack

| Layer    | Technology                   |
| -------- | ---------------------------- |
| Frontend | Streamlit, Plotly 3D         |
| 3D       | PyVista                      |
| Atlas    | BrainGlobe Atlas API (Allen) |
| Styling  | Custom CSS, Playfair Display |

---

## License

See [LICENSE](LICENSE) for details.
