import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import sys

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    with open(os.path.basename(sys.argv[1]), "rb") as infile:
        image_paths = map(lambda a: a.decode(), infile.read().splitlines())

    full_df = pd.DataFrame()
    for p in image_paths:
        feature_path = "__temp-" + os.path.basename(p).split('.')[0] + "-extracted_features.pickle"
        with open(feature_path, "rb") as infile:
            for df, _ in pickle.load(infile):
                full_df = full_df.append(df)

    scaler = StandardScaler()
    pca = PCA(n_components='mle')

    scaled = scaler.fit_transform(full_df)
    dft = pd.DataFrame(pca.fit_transform(scaled), index=full_df.index)

    silhouette_scores = []
    k_range = range(2, 15)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(dft)
        silhouette_scores += [silhouette_score(dft, labels),]

    plt.plot(k_range, silhouette_scores,'bx-')
    plt.title("Silhouette Scores")
    plt.xlabel("k")
    plt.show()

    kmeans = KMeans(n_clusters=int(input("Selected k: ")), random_state=0)
    kmeans.fit(dft)

    write_path = "trained-kmeans.pickle"
    print("Writing trained models to " + write_path + ".")
    with open(write_path, "wb") as outfile:
        pickle.dump([scaler, pca, kmeans], outfile)
