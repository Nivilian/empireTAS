import os
from pathlib import Path
root = Path(r"d:\codeLibrary\empireTAS\src\images\labeledImages")
classes = []
for terrain_dir in root.iterdir():
    if not terrain_dir.is_dir():
        continue
    subdirs = [d for d in terrain_dir.iterdir() if d.is_dir()]
    if subdirs:
        for occ_dir in subdirs:
            occ = occ_dir.name
            class_name = terrain_dir.name if occ in ("free", "空") else f"{terrain_dir.name}_{occ}"
            if class_name not in classes:
                classes.append(class_name)
    else:
        if terrain_dir.name not in classes:
            classes.append(terrain_dir.name)
# Also check for any files directly under terrain_dir
for p in root.iterdir():
    if p.is_dir():
        files = [x for x in p.iterdir() if x.is_file()]
        if files and p.name not in classes:
            classes.append(p.name)
print('CLASSES:', len(classes))
print('\n'.join(classes))
