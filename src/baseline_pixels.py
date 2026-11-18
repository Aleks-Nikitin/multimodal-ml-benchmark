import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from src.data_utils import load_cifar_subset

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
    }

    for name, clf in classifiers.items():
        print(f"Training {name} on raw 3072-d pixels...")
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Raw Pixel Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    run_pixel_baselines()