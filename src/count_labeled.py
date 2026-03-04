import os
root = r"d:\codeLibrary\empireTAS\src\images\labeledImages"
summary = {}
for terrain in sorted(os.listdir(root)):
    tpath = os.path.join(root, terrain)
    if not os.path.isdir(tpath):
        continue
    summary[terrain] = {}
    for occ in sorted(os.listdir(tpath)):
        occp = os.path.join(tpath, occ)
        if not os.path.isdir(occp):
            continue
        files = [f for f in os.listdir(occp) if os.path.isfile(os.path.join(occp,f))]
        summary[terrain][occ] = len(files)
print('COUNTS')
for t, d in summary.items():
    total = sum(d.values())
    print(f"{t}: total={total}")
    for occ, cnt in d.items():
        print(f"  {occ}: {cnt}")
