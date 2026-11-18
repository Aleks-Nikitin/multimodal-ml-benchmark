import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_predictions(y_true, y_pred, model_name, representation_type):
    """Computes Accuracy, Macro-Precision, Macro-Recall, and Macro-F1."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "Representation": representation_type,
        "Model": model_name,
        "Accuracy (%)": round(acc * 100, 2),
        "Macro-Precision": round(prec, 4),
        "Macro-Recall": round(rec, 4),
        "Macro-F1": round(f1, 4),
    }

def save_results_table(results_list, out_path="./results/benchmark_metrics.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(out_path, index=False)
    print("\n=== Benchmark Summary Table ===")
    print(df.to_markdown(index=False))
    return df