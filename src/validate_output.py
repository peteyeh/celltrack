import cv2
import os
import sys

from tqdm import tqdm

output_dir = "/projects/p31689/Image_Analysis/output"

with open("manifest.txt", "rb") as infile:
    paths = [p.decode('UTF-8') for p in infile.read().splitlines()]

failed_paths = []
for path in tqdm(paths):
    num_images = len(cv2.imreadmulti(path, flags=cv2.IMREAD_GRAYSCALE)[1])
    output_path = os.path.join(output_dir, path.split("/")[-1].split(".")[0])
    last_run = sorted(os.listdir(output_path), reverse=True)[0]
    output_path = os.path.join(output_path, last_run)
    if not os.path.exists(os.path.join(output_path, "mask_images")) or \
            num_images != len(os.listdir(os.path.join(output_path, "mask_images"))) or \
            not os.path.exists(os.path.join(output_path, "contoured_images")) or \
            num_images != len(os.listdir(os.path.join(output_path, "contoured_images"))) or \
            not os.path.exists(os.path.join(output_path, "data_labeled.csv")) or \
            not os.path.exists(os.path.join(output_path, "analysis.csv")):
        failed_paths += [path,]

for path in failed_paths:
    print(path)

