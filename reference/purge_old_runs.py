# Removes all runs except the newest one

import os
import shutil

output_dir = "/projects/p31689/Image_Analysis/output"

with open("../src/manifest.txt", "rb") as infile:
    paths = [p.decode('UTF-8') for p in infile.read().splitlines()]

for path in paths:
    output_path = os.path.join(output_dir, path.split("/")[-1].split(".")[0])
    for d in sorted(os.listdir(output_path), reverse=False)[:-1]:
        shutil.rmtree(os.path.join(output_path, d))
