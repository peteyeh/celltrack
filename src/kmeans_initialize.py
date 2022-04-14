import numpy as np
import os
import pandas as pd
import pickle
import sys

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def find_k(df):
    silhouette_scores = []
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(df)
        silhouette_scores += [silhouette_score(df, labels),]
    return np.argmax(silhouette_scores[1:]) + 3  # [1:] means we start at k = 3

if __name__ == "__main__":
    try:
        feature_path = "__temp-" + os.path.basename(sys.argv[1]).split('.')[0] + "-extracted_features.pickle"
        with open(feature_path, "rb") as infile:
            df, _ = pickle.load(infile)[int(sys.argv[2])]
            print("Loaded DataFrame from " + feature_path + ".")
        k = int(sys.argv[3]) if len(sys.argv) == 4 else None
    except:
        print("Received invalid parameters. Make sure ftextraction.py has been run and that you execute with:")
        print("  python3 kmeans_initialize.py [image_path] [index] (k)")
        print("If k is not given it will automatically be chosen.")
        sys.exit(1)

    scaler = StandardScaler()
    pca = PCA(n_components='mle')

    scaled = scaler.fit_transform(df)
    dft = pd.DataFrame(pca.fit_transform(scaled), index=df.index)
    if k is None:
        k = find_k(dft)
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(dft)

    write_path = "__temp-" + os.path.basename(sys.argv[1]).split('.')[0] + "-kmeans.pickle"
    print("Writing trained models to " + write_path + ".")
    with open(write_path, "wb") as outfile:
        pickle.dump([scaler, pca, kmeans], outfile)
