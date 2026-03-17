import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from torchvision.datasets import DatasetFolder
from collections import Counter


# update these


csv_paths = {
    "dipg": r"Z:\Yina\onevsall\dipg\resnext50_32x4d.fb_swsl_ig1b_ft_in1k_pos0_bs4_lr0.0001\resnext50_32x4d.fb_swsl_ig1b_ft_in1k_pos0_bs4_lr0.0001_test_preds.csv",
    "medulloblastoma": r"Z:\Yina\onevsall\medulloblastoma\resnext50_32x4d.fb_swsl_ig1b_ft_in1k_pos1_bs4_lr0.0001\resnext50_32x4d.fb_swsl_ig1b_ft_in1k_pos1_bs4_lr0.0001_test_preds.csv",
    "plgg": r"Z:\Yina\onevsall\plgg\resnext50_32x4d.fb_swsl_ig1b_ft_in1k_pos2_bs4_lr0.0001\resnext50_32x4d.fb_swsl_ig1b_ft_in1k_pos2_bs4_lr0.0001_test_preds.csv",
}
model_name = "rexnext50"

test_dataset = DatasetFolder(
    "data_output/splits/test",
    loader=np.load,
    extensions=[".npy"]
)

class_to_idx = test_dataset.class_to_idx.items()
class_names = {v: k for k, v in class_to_idx}
save_dir = "ensemble_results"
os.makedirs(save_dir, exist_ok=True)


dfs = []

for class_name, path in csv_paths.items():

    df = pd.read_csv(path)

    df = df.rename(columns={
        "logit": f"logit_{class_name}",
        "prob": f"prob_{class_name}"
    })

    dfs.append(df)

merged = dfs[0]

for df in dfs[1:]:
    merged = merged.merge(df, on=["file", "true_label"])

print("Merged shape:", merged.shape)

class_names = list(csv_paths.keys())

logit_cols = [f"logit_{c}" for c in class_names]
prob_cols = [f"prob_{c}" for c in class_names]

logits = merged[logit_cols].values
probs = merged[prob_cols].values

true_labels = merged["true_label"].values

counts = Counter(true_labels)

print("\nTest set class distribution:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {counts[i]}")

pred_labels = np.argmax(logits, axis=1)


accuracy = accuracy_score(true_labels, pred_labels)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average="macro"
)

precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average="micro"
)

precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average=None
)

print("\nOverall Metrics")
print("Accuracy:", accuracy)

print("\nMacro")
print("Precision:", precision_macro)
print("Recall:", recall_macro)
print("F1:", f1_macro)

print("\nMicro")
print("Precision:", precision_micro)
print("Recall:", recall_micro)
print("F1:", f1_micro)

print("\nPer Class")
for i, name in enumerate(class_names):
    print(
        f"{name} | "
        f"Precision {precision_class[i]:.4f} "
        f"Recall {recall_class[i]:.4f} "
        f"F1 {f1_class[i]:.4f}"
    )


cm = confusion_matrix(true_labels, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues")
plt.title("Multiclass Confusion Matrix")
plt.tight_layout()

plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}.png"))
plt.close()


y_true_bin = label_binarize(true_labels, classes=range(len(class_names)))

fpr = {}
tpr = {}
roc_auc = {}

plt.figure()

for i, class_name in enumerate(class_names):

    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(
        fpr[i],
        tpr[i],
        label=f"{class_name} (AUC={roc_auc[i]:.3f})"
    )


fpr["micro"], tpr["micro"], _ = roc_curve(
    y_true_bin.ravel(),
    probs.ravel()
)

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.plot(
    fpr["micro"],
    tpr["micro"],
    linestyle=":",
    linewidth=3,
    label=f"micro-average (AUC={roc_auc['micro']:.3f})"
)


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))

mean_tpr = np.zeros_like(all_fpr)

for i in range(len(class_names)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= len(class_names)

roc_auc["macro"] = auc(all_fpr, mean_tpr)

plt.plot(
    all_fpr,
    mean_tpr,
    linestyle="--",
    linewidth=3,
    label=f"macro-average (AUC={roc_auc['macro']:.3f})"
)

plt.plot([0, 1], [0, 1], "--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Multiclass ROC Curves")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(os.path.join(save_dir, f"roc_curves_{model_name}.png"))
plt.close()


merged["pred_label"] = pred_labels

merged.to_csv(
    os.path.join(save_dir, f"{model_name}.csv"),
    index=False
)

print("\nSaved results to:", save_dir)