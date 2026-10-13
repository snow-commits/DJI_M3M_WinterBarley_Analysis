from pathlib import Path
project_dir = Path(r"E:\Projects\DJi/M3M_Project")
img_dir = Path(
    r"E:\Projects\DJi\M3M_WinterBarley_Dataset"
    r"\M3M_WinterBarley_Dataset"
    r"\90m_RGBMSP_RTK_Overlap7080"
    r"\Dataset"
)

boundary_path = Path(r"E:\Projects\DJi\M3M_WinterBarley_Dataset"
                    r"\M3M_WinterBarley_Dataset\boundary.gpkg")

annotation_path = Path(r"E:\Projects\DJi\M3M_WinterBarley_Dataset"
                       r"\M3M_WinterBarley_Dataset\"Image_Annotation")

panel_dir = Path(r"E:\Projects\DJi\M3M_WinterBarley_Dataset"
    r"\M3M_WinterBarley_Dataset"
    r"\90m_RGBMSP_RTK_Overlap7080"
    r"\Reflectance_Panel")
panel_reflectance = {
    "G":0.493,
    "R":0.494,
    "RE":0.494,
    "NIR":0.493
}
