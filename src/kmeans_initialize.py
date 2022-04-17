import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import sys

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

if __name__ == "__main__":
    with open(os.path.basename(sys.argv[1]), "rb") as infile:
        image_paths = map(lambda a: a.decode(), infile.read().splitlines())

    out_path = "." if len(sys.argv) < 3 else sys.argv[2]  # this directory should already exist

    full_df = pd.DataFrame()
    for p in image_paths:
        base_path = os.path.join(out_path, os.path.basename(p).split('.')[0])
        last_run = sorted(os.listdir(base_path), reverse=True)[0]
        feature_path = os.path.join(base_path, last_run, "extracted_features.pickle")
        print("Loading features from %s." % feature_path)
        with open(feature_path, "rb") as infile:
            for df, _ in pickle.load(infile):
                full_df = full_df.append(df)

    print("%i data points loaded." % len(full_df))

    if len(full_df) > 40000:
        print("Downsampling to 40000 data points.")
        full_df = full_df.sample(40000, random_state=0)

    scaler = StandardScaler()
    pca = PCA(n_components='mle')

    scaled = scaler.fit_transform(full_df)
    dft = pd.DataFrame(pca.fit_transform(scaled), index=full_df.index)

    if len(sys.argv) > 2:
        k = sys.argv[2]
        print("Training kmeans model with k=%i." % k)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(dft)

        write_path = os.path.join(out_path, "kmeans", datetime.now().strftime("%Y%m%d%H%M%S") + ".pickle")
        print("Writing trained models to " + write_path + ".")
        with open(write_path, "wb") as outfile:
            pickle.dump([scaler, pca, kmeans], outfile)
    else:
        silhouette_scores = []
        k_range = range(2, 15)
        print("Obtaining silhouette scores for k=[%i,%i]." % (k_range[0], k_range[-1]))
        for k in tqdm(k_range):
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(dft)
            silhouette_scores += [silhouette_score(dft, labels),]

        plt.plot(k_range, silhouette_scores,'bx-')
        plt.title("Silhouette Scores")
        plt.xlabel("k")
        plt.savefig(os.path.join(out_path, "silhouette_scores", datetime.now().strftime("%Y%m%d%H%M%S") + ".png"))
