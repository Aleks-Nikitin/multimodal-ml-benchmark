import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.data_utils import load_cifar_subset
from src.evaluate_utils import evaluate_predictions, save_results_table

def run_pixel_baselines():
    print("Loading stratified CIFAR-10 subset (1,000 train / 250 test)...")
    X_train_img, y_train, X_test_img, y_test = load_cifar_subset()

    X_train_flat = X_train_img.reshape(len(X_train_img), -1).astype(np.float32)
    X_test_flat = X_test_img.reshape(len(X_test_img), -1).astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=200, C=0.05, solver="lbfgs", random_state=42),
        "Linear/RBF SVM (SVC)": SVC(kernel="rbf", C=1.0, random_state=42),
        "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=42),
    }

    results = []
    os.makedirs("./embeddings", exist_ok=True)
    for name, clf in classifiers.items():
        print(f"Training {name} on raw 3072-d pixels...")
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        results.append(evaluate_predictions(y_test, preds, name, "Raw Pixels (3072-d)"))
        if "SVM" in name:
            np.save("./embeddings/raw_svm_preds.npy", preds)

    save_results_table(results, "./results/pixel_baseline_metrics.csv")

if __name__ == "__main__":
    run_pixel_baselines()