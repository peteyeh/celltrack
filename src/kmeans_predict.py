import os
import pandas as pd
import pickle
import sys

if __name__ == "__main__":
    try:
        sub_name = os.path.basename(sys.argv[1]).split('.')[0]
        feature_path = "__temp-" + sub_name + "-extracted_features.pickle"
        kmeans_path = "trained-kmeans.pickle"
    except:
        print("Received invalid image_path. Make sure you execute with:")
        print("  python3 kmeans_predict.py [image_path]")
        sys.exit(1)

    try:
        with open(feature_path, "rb") as infile:
            result = pickle.load(infile)
            print("Loaded features from " + feature_path + ".")
    except:
        print("Unable to load features from %s. Did you run ftextraction.py?" % feature_path)
        sys.exit(1)

    try:
        with open(kmeans_path, "rb") as infile:
            scaler, pca, kmeans = pickle.load(infile)
            print("Loaded models from " + kmeans_path + ".")
    except:
        print("Unable to load models from %s. Did you run kmeans_initialize.py?" % kmeans_path)
        sys.exit(1)

    labels = []
    for df, _ in result:
        df = pd.DataFrame(pca.transform(scaler.transform(df)), index=df.index)
        labels += [kmeans.predict(df),]

    write_path = "__temp-" + sub_name + "-labels.pickle"
    print("Writing labels to " + write_path + ".")
    with open(write_path, "wb") as outfile:
        pickle.dump(labels, outfile)

