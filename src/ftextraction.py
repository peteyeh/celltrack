import cv2
import numpy as np
import pandas as pd
import six

from collections import OrderedDict
from radiomics import featureextractor
from SimpleITK import GetImageFromArray

def extract_features(image, mask_image):
    num_components, mask_labels, stats, centroids = \
            cv2.connectedComponentsWithStats(mask_image, connectivity=8)

    extractor = initialize_extractor()
    features = OrderedDict()

    for i in range(1, len(stats)):  # index 0 is the background component
        mask = np.uint8(mask_labels == i)
        result = extractor.execute(GetImageFromArray(image),
                                   GetImageFromArray(mask))
        if len(features) == 0:
            features['x'] = [int(centroids[i][0])]
            features['y'] = [int(centroids[i][1])]
            for key, val in six.iteritems(result):
                if key.startswith("original_"):
                    features[key] = [val]
        else:
            features['x'] += [int(centroids[i][0])]
            features['y'] += [int(centroids[i][1])]
            for key, val in six.iteritems(result):
                if key.startswith("original_"):
                    features[key] += [val]

    # POSSIBLE BUG, VERIFY THAT from_dict DOESN'T INFER WRONG TYPES
    # Note that we can return mask_labels here rather than an array of individual
    # masks - our sample order will match the labels in mask_labels
    return pd.DataFrame.from_dict(features).set_index(['x', 'y']), mask_labels

def initialize_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # Enable everything but shape (3D) and glcm.SumAverage
    # NOTE: 'Imc2' seems to cause a "RuntimeWarning: invalid value encountered in sqrt" in stderr.
    # Per pyradiomics docs: "Due to machine precision errors, it is possble that HXY > HXY2,
    # which would result in returning complex numbers. In these cases, a value of 0 is returned
    # for IMC2. This is done on a per-angle basis (i.e. prior to any averaging)."
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=[], shape2D=[],
                                   glcm=['Autocorrelation', 'JointAverage', 'ClusterProminence',
                                         'ClusterShade', 'ClusterTendency', 'Contrast',
                                         'Correlation', 'DifferenceAverage', 'DifferenceEntropy',
                                         'DifferenceVariance', 'JointEnergy', 'JointEntropy',
                                         'Imc1', 'Imc2', 'Idm', 'Idmn', 'Id', 'Idn',
                                         'InverseVariance', 'MaximumProbability', 'SumEntropy',
                                         'SumSquares'],
                                   glszm=[], glrlm=[], ngtdm=[], gldm=[])
    return extractor

if __name__ == "__main__":
    import glob, os, pickle, sys
    from joblib import cpu_count, delayed, Parallel
    from tqdm import tqdm

    from improcessing import *

    try:
        image_stack = list(map(scale_image, cv2.imreadmulti(sys.argv[1], flags=cv2.IMREAD_GRAYSCALE)[1]))
    except:
        print("Unable to read image stack. Make sure you execute with:")
        print("  python3 ftextraction.py image_path [output_path]")
        sys.exit(1)

    out_path = "." if len(sys.argv) < 3 else sys.argv[2]  # this directory should already exist

    try:
        basename = os.path.basename(sys.argv[1]).split('.')[0]
        base_path = os.path.join(out_path, basename)
        last_run = sorted(os.listdir(base_path), reverse=True)[0]
        mask_path = os.path.join(base_path, last_run, "mask_images")
        mask_images = [cv2.imread(_, flags=cv2.IMREAD_GRAYSCALE) for _ in sorted(glob.glob(mask_path + "/*.png"))]
    except:
        print("Unable to find mask images. Did you run maskcreation.py?")
        sys.exit(1)

    if len(image_stack) == len(mask_images):
        print("Matched %i raw images to mask images for %s." % (len(image_stack), basename))
    else:
        print("Unable to match mask images (%i) to raw images (%i). Aborting." %
              (len(mask_images), len(image_stack)))
        sys.exit(1)

    result = \
        Parallel(n_jobs=cpu_count())(
            delayed(extract_features)(*_) for _ in tqdm(zip(image_stack, mask_images)))

    write_path = os.path.join(base_path, last_run, "extracted_features.pickle")
    print("Writing extracted features to " + write_path + ".")
    with open(write_path, "wb") as outfile:
        pickle.dump(result, outfile)
