import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.data_utils import CIFAR10_CLASSES
from src.evaluate_utils import evaluate_predictions, save_results_table

def main():
    train_X = np.load("./embeddings/train_clip.npy")
    train_y = np.load("./embeddings/y_train.npy")
    test_X = np.load("./embeddings/test_clip.npy")
    test_y = np.load("./embeddings/y_test.npy")
    text_X = np.load("./embeddings/text_clip.npy")

    rows = []
    zs_preds = np.argmax(test_X @ text_X.T, axis=1)
    np.save("./embeddings/zeroshot_preds.npy", zs_preds)
    rows.append(evaluate_predictions(test_y, zs_preds, "Zero-Shot CLIP", "CLIP Text-Image"))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
        "SVM (SVC)": SVC(kernel="rbf", random_state=42),
    }
    for name, clf in models.items():
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)
        rows.append(evaluate_predictions(test_y, preds, name, "CLIP Embeddings"))
        if "SVM" in name:
            np.save("./embeddings/clip_svm_preds.npy", preds)

    base_csv = "./results/pixel_baseline_metrics.csv"
    if os.path.exists(base_csv):
        df_base = pd.read_csv(base_csv)
        all_rows = df_base.to_dict("records") + rows
    else:
        all_rows = rows
    save_results_table(all_rows, "./results/full_benchmark_comparison.csv")

    coords_2d = PCA(n_components=2, random_state=42).fit_transform(test_X)
    clusters = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(test_X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=clusters, cmap="tab10", s=20, alpha=0.8)
    plt.title("CLIP Embeddings Clustered with K-Means (PCA 2D View)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig("./results/clusters.png", dpi=150)
    plt.close()
    print("Saved ./results/clusters.png")

if __name__ == "__main__":
    main()