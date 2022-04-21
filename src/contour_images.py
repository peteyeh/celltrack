import cv2
import imageio
import os
import pickle
import sys

from joblib import cpu_count, delayed, Parallel
from tqdm import tqdm

from displaytools import get_colormap, get_contoured_image
from improcessing import get_mode, scale_image

if __name__ == "__main__":
    out_path = "." if len(sys.argv) < 3 else sys.argv[2]  # this directory should already exist

    try:
        basename = os.path.basename(sys.argv[1]).split('.')[0]
        base_path = os.path.join(out_path, basename)
        last_run = sorted(os.listdir(base_path), reverse=True)[0]
        mask_path = os.path.join(base_path, last_run, "extracted_features.pickle")
        label_path = os.path.join(base_path, last_run, "labels.pickle")
    except:
        print("Received invalid image_path. Make sure you execute with:")
        print("  python3 cluster_analysis.py image_path [output_path]")
        sys.exit(1)

    try:
        image_stack = list(map(scale_image, cv2.imreadmulti(sys.argv[1], flags=cv2.IMREAD_GRAYSCALE)[1]))
        if len(image_stack) == 0:
            raise Exception()
    except:
        print("Unable to read image stack. Make sure you execute with:")
        print("  python3 maskcreation.py image_path [output_path]")
        sys.exit(1)

    try:
        with open(mask_path, "rb") as infile:
            mask_result = list(map(lambda a: a[1], pickle.load(infile)))
            print("Loaded feature extraction data from %s." % mask_path)
    except:
        print("Unable to load feature extraction data from %s. Did you run ftextract.py?" % mask_path)
        sys.exit(1)

    try:
        with open(label_path, "rb") as infile:
            label_result = pickle.load(infile)
            print("Loaded labels from %s." % label_path)
    except:
        print("Unable to load labels from %s. Did you run kmeans_predict.py?" % label_path)
        sys.exit(1)

    if len(image_stack) != len(mask_result) or len(image_stack) != len(label_result):
    	print("Unable to match image stack to saved mask and label data. Aborting.")
    	sys.exit(1)

    print("Creating %i contoured images for %s:" % (len(image_stack), basename))
    args = zip(image_stack, mask_result, label_result, [get_colormap()]*len(image_stack))
    contoured_images = Parallel(n_jobs=cpu_count())(delayed(get_contoured_image)(*_) for _ in tqdm(args))

    write_path = os.path.join(base_path, last_run, "contoured_images")
    print("Writing mask images to " + write_path + ":")
    os.mkdir(write_path)
    for i in tqdm(range(len(contoured_images))):
        imageio.imwrite((write_path + "/%i.png" % i), contoured_images[i])
