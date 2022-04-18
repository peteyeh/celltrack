import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys

from collections import Counter
from joblib import cpu_count, delayed, Parallel
from tqdm import tqdm

from colormap import get_color

def esf_parallel(k, labels, mask_labels):
    masks = np.array([np.uint8(mask_labels==idx) for idx in np.where(labels==k)[0]+1])
    if len(masks) == 0:
        return 0, 0
    sizes = list(map(np.sum, masks))
    return np.sum(sizes), int(np.mean(sizes))

if __name__ == "__main__":
    out_path = "." if len(sys.argv) < 3 else sys.argv[2]  # this directory should already exist

    try:
        base_path = os.path.join(out_path, os.path.basename(sys.argv[1]).split('.')[0])
        last_run = sorted(os.listdir(base_path), reverse=True)[0]
        mask_path = os.path.join(base_path, last_run, "extracted_features.pickle")
        label_path = os.path.join(base_path, last_run, "labels.pickle")
    except:
        print("Received invalid image_path. Make sure you execute with:")
        print("  python3 cluster_analysis.py image_path [output_path]")
        sys.exit(1)

    try:
        with open(mask_path, "rb") as infile:
            mask_result = pickle.load(infile)
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

    nc = 1
    for l in label_result:
        nc = max(nc, max(l)+1)
    counts = np.zeros((len(label_result), nc))
    total_sizes = np.zeros((len(label_result), nc))
    mean_sizes = np.zeros((len(label_result), nc))
    print("Processing %i sets of label data." % len(label_result))
    for i in tqdm(range(len(label_result))):
        _, mask_labels = mask_result[i]
        labels = label_result[i]

        c = dict(Counter(labels))
        for k in range(nc):
            if k not in c:
                c[k] = 0
        counts[i] = [a[1] for a in sorted(c.items(), key=lambda a: a[0])]

        args = zip(range(nc), [labels]*nc, [mask_labels]*nc)
        fl_result = Parallel(n_jobs=cpu_count())(delayed(esf_parallel)(*_) for _ in args)
        fl_result = np.array(fl_result, dtype=object)
        total_sizes[i] = fl_result[:,0]
        mean_sizes[i] = fl_result[:,1]

    counts = counts.T
    total_sizes = total_sizes.T
    mean_sizes = mean_sizes.T

    ### PLOT CLASS COUNTS ###
    sorted_data = sorted(zip(counts, list(range(len(counts))),
                         np.mean(counts, axis=1), np.std(counts, axis=1)),
                     key=lambda a: a[0][-1])
    bottom = np.zeros(counts.shape[1])
    handles = []
    plt.title("Class Distributions")
    plt.ylabel("Cell Count (Mask Regions)")
    plt.xlabel("Frame")
    plt.xlim([0,len(sorted_data[0][0])])
    for c, k, _, _ in sorted_data:
        handles += [plt.bar(range(len(c)), c, bottom=bottom, color=colormap.get_color(k)),]
        bottom += c
    plt.legend(reversed(handles),
               [("Class %i: %.2f [%.2f]" % a[1:]) for a in reversed(sorted_data)],
               title="Mean [Std Dev]")
    write_path = os.path.join(base_path, last_run, "class_counts_barplot.png")
    plt.savefig(write_path)
    print("Saved %s." % write_path)

    plt.figure(figsize=(12,5))
    plt.title("Class Distributions")
    plt.ylabel("Cell Count (Mask Regions)")
    plt.xlabel("Frame")
    plt.xlim([0,len(sorted_data[0][0])])
    for k in range(len(counts)):
        plt.plot(counts[k], color=colormap.get_color(k),
                 label=("Class %i: %.2f [%.2f]" % (k, np.mean(counts[k]), np.std(counts[k]))))
    plt.legend(title="Mean [Std Dev]")
    write_path = os.path.join(base_path, last_run, "class_counts_lineplot.png")
    plt.savefig(write_path)
    print("Saved %s." % write_path)

    ### PLOT CLASS AREAS ###
    meanstd = list(zip(np.mean(total_sizes, axis=1), np.std(total_sizes, axis=1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.set_title("Mean Area")
    ax1.set_ylabel("Pixel Count")
    ax1.set_xlabel("Frame")
    for k in range(len(mean_sizes)):
        ax1.plot(mean_sizes[k], color=colormap.get_color(k), label=("Class %i" % k))
    ax2.set_title("Total Area: Mean [Std Dev]")
    ax2.set_ylabel("Pixel Count")
    ax2.set_xlabel("Frame")
    for k in range(len(total_sizes)):
        ax2.plot(total_sizes[k], color=colormap.get_color(k),
                 label=("Class %i: %.2f [%.2f]" % (k, *meanstd[k])))
    ax2.legend()
    write_path = os.path.join(base_path, last_run, "class_areas.png")
    plt.savefig(write_path)
    print("Saved %s." % write_path)
