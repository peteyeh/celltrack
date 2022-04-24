import os
import pandas as pd
import pickle
import sys

from tqdm import tqdm

if __name__ == "__main__":
    out_path = "." if len(sys.argv) < 3 else sys.argv[2]  # this directory should already exist

    try:
        base_path = os.path.join(out_path, "kmeans")
        last_run = sorted(os.listdir(base_path), reverse=True)[0]
        kmeans_path = os.path.join(base_path, last_run)  # ".pickle" is contained within last_run

        base_path = os.path.join(out_path, os.path.basename(sys.argv[1]).split('.')[0])
        last_run = sorted(os.listdir(base_path), reverse=True)[0]
        feature_path = os.path.join(base_path, last_run, "extracted_features.pickle")
    except:
        print("Received invalid image_path. Make sure you execute with:")
        print("  python3 kmeans_predict.py image_path [output_path]")
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

    labels_list = []
    df_list = []
    for i in tqdm(range(len(result))):
        df = result[i][0]
        transformed_df = pd.DataFrame(pca.transform(scaler.transform(df)), index=df.index)
        labels = kmeans.predict(transformed_df)
        labels_list += [labels,]
        # Save pre-transformed DataFrame for CSV exporting
        df = df.reset_index()
        df['image'] = i
        df['label'] = labels
        df_list += [df.set_index(['image', 'x', 'y']),]

    csv_write_path = os.path.join(base_path, last_run, "data_labeled.csv")
    pickle_write_path = os.path.join(base_path, last_run, "labels.pickle")
    print("Writing labels to " + pickle_write_path + " and labeled CSV to " + csv_write_path + ".")
    with open(pickle_write_path, "wb") as outfile:
        pickle.dump(labels_list, outfile)
    pd.concat(df_list).to_csv(csv_write_path)
