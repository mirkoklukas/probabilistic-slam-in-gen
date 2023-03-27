import os

notebooks = [
"99 - Bresenham.ipynb",
"00 - My Utils.ipynb",
"01 - Geometry - Primitives and Raycaster.ipynb",
"11 - Pose.ipynb",
"12 - Measurements.ipynb",
"13 - Beam_Models.ipynb",
"14 - Grid_Map.ipynb",
"10 - GridSLAM_Module.ipynb",
"99 - CUDA_Raycaster_Line_Map.ipynb",
]


if __name__ == '__main__':
    for nb in notebooks:
        nb = nb.replace(" ", "\ ")
        os.system(f"nbx_jl {nb}")
