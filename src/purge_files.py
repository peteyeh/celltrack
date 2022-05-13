import os
import sys

out_path = "." if len(sys.argv) < 3 else sys.argv[2]  # this directory should already exist

try:
    basename = os.path.basename(sys.argv[1]).split('.')[0]
    base_path = os.path.join(out_path, basename)
    last_run = sorted(os.listdir(base_path), reverse=True)[0]
    mask_path = os.path.join(base_path, last_run, "extracted_features.pickle")
    data_path = os.path.join(base_path, last_run, "data.csv")
    label_path = os.path.join(base_path, last_run, "labels.pickle")
except:
    print("Received invalid image_path. Make sure you execute with:")
    print("  python3 purge_files.py image_path [output_path]")
    sys.exit(1)

if os.path.exists(mask_path):
    os.remove(mask_path)
    print("Purged %s." % mask_path)

if os.path.exists(data_path):
    os.remove(data_path)
    print("Purged %s." % data_path)

if os.path.exists(label_path):
    os.remove(label_path)
    print("Purged %s." % label_path)