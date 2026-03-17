import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, WeightedRandomSampler

import timm_3d

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

def to_one_vs_all(labels, positive_class):
    return (labels == positive_class).float()

def pad_to_shape(volume, target_shape):
    """Pads 3D numpy array (D,H,W) to target_shape with zeros."""
    pad_width = []
    for vol_dim, target_dim in zip(volume.shape, target_shape):
        delta = max(target_dim - vol_dim, 0)
        before = delta // 2
        after = delta - before
        pad_width.append((before, after))
    return np.pad(volume, pad_width, mode="constant", constant_values=0)


def get_all_shapes(paths):
    shapes = []
    for p in paths:
        for root, _, files in os.walk(p):
            for f in files:
                if f.endswith(".npy"):
                    vol = np.load(os.path.join(root, f))
                    vol = np.transpose(vol, (2, 0, 1))
                    shapes.append(vol.shape)
    return shapes


def custom_loader(path, target_shape, pad=True):
    vol = np.load(path)
    vol = np.transpose(vol, (2, 0, 1))
    if pad:
        vol = pad_to_shape(vol, target_shape)
    return torch.tensor(vol, dtype=torch.float32)


class AddChannel:
    def __call__(self, x):
        return x.unsqueeze(0)
    
class CustomDataset(DatasetFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path

class TumourClassifier3D(nn.Module):
    def __init__(self, backbone_name='efficientnet_b1.ft_in1k', pretrained=True, num_classes=1):
        super().__init__()
        self.model = timm_3d.create_model(
            backbone_name, pretrained=pretrained, in_chans=1, num_classes=0
        )
        self.classifier = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        labels_map,
        pos_class,
        test_loader=None,
        device="cuda",
        save_dir="results",
        patience=10,
        learning_rate=1e-4,
        exp_name="experiment"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.labels_map = labels_map
        self.pos_class = pos_class # int
        self.pos_label = labels_map[pos_class]

        labels = np.array([y for _, y in train_loader.dataset.samples])
        pos = np.sum(labels == pos_class)
        neg = np.sum(labels != pos_class)

        pos_weight = torch.tensor([neg / pos], device=device)

        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

        self.best_val_loss = float("inf")
        self.patience = patience
        self.counter = 0

        self.exp_name = exp_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # CSV log path
        self.csv_path = os.path.join(save_dir, f"{exp_name}_metrics.csv")

        # Initialize CSV
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])  # filled during training

    def log_epoch_to_csv(self, epoch, train_loss, val_loss):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

    def log_test_metrics_to_csv(self, precision, recall, f1):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["Test Results"])
            writer.writerow(["precision", "recall", "f1"])
            writer.writerow([precision, recall, f1])

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y, _ in tqdm(self.train_loader):
            x = x.to(self.device)
            y = to_one_vs_all(y, self.pos_class)
            y = y.float().to(self.device)

            preds = self.model(x).squeeze(-1) 
            loss = self.criterion(preds, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y, _ in self.val_loader:
                x = x.to(self.device)
                y = to_one_vs_all(y, self.pos_class)
                y = y.float().to(self.device)
                preds = self.model(x).squeeze(-1) 
                loss = self.criterion(preds, y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)


    def _collect_outputs(self, loader):
        self.model.eval()

        preds_list, logits_list, probs_list, labels_list, files_list = [], [], [], [], []

        with torch.no_grad():
            for x, y, paths in loader:
                x = x.to(self.device)

                logits = self.model(x).squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                preds_list.append(preds.cpu().numpy())
                logits_list.append(logits.cpu().numpy())
                probs_list.append(probs.cpu().numpy())
                labels_list.append(y.cpu().numpy())

                files_list.extend([os.path.basename(p) for p in paths])


        preds = np.concatenate(preds_list)
        logits = np.concatenate(logits_list)
        probs = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)

        return preds, logits, probs, labels, files_list

    def plot_loss_curve(self):
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.save_dir, f"{self.exp_name}_loss_curve.png"))
        plt.close()
    
    def save_probs(self, loader, split="test"):

        preds, logits, probs, labels, files = self._collect_outputs(loader)

        save_path = os.path.join(
            self.save_dir,
            f"{self.exp_name}_{split}_preds.csv"
        )

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "true_label", "logit", "prob"])

            for file, label, logit, prob in zip(files, labels, logits, probs):
                writer.writerow([file, int(label), float(logit), float(prob)])

    def plot_auc_curve(self, loader, split="val"):
        preds, _, probs, labels, _ = self._collect_outputs(loader)

        # Binary ROC: treat POSITIVE_CLASS as the "positive"
        binary_labels = (labels == self.pos_class).astype(int)

        fpr, tpr, _ = roc_curve(binary_labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"{self.pos_label} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({split})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{self.exp_name}_roc_curve_{split}.png"))
        plt.close()

    def compute_confusion_metrics(self, loader, split="test"):
        preds, _, probs, labels, _ = self._collect_outputs(loader)

        # text_preds  = [self.labels_map[int(p)] for p in preds]
        # text_labels = [self.labels_map[int(l)] for l in labels]

        # all_label_names = list(self.labels_map.values())

        text_preds  = [self.pos_label if p == 1 else "negative" for p in preds]
        text_labels = [self.pos_label if l == self.pos_class else "negative" for l in labels]
        all_label_names = ["negative", self.pos_label]

        cm = confusion_matrix(text_labels, text_preds, labels=all_label_names)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=all_label_names
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix ({split})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{self.exp_name}_confusion_matrix_{split}.png"))
        plt.close()

        # Precision/Recall/F1 using pos_label text
        precision, recall, f1, _ = precision_recall_fscore_support(
            text_labels, text_preds,
            pos_label=self.pos_label,
            average="binary"
        )

        return precision, recall, f1

    def _check_early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir, f"{self.exp_name}_best_model.pt")
            )
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                return True
        return False

    def fit(self, epochs=10):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            self.log_epoch_to_csv(epoch, train_loss, val_loss)

            if self._check_early_stopping(val_loss):
                break

        self.plot_loss_curve()
        self.plot_auc_curve(self.val_loader, split="val")

    def test(self):
        self.model.load_state_dict(
            torch.load(os.path.join(self.save_dir, f"{self.exp_name}_best_model.pt"))
        )
        self.plot_auc_curve(self.test_loader, split="test")
        self.save_probs(self.test_loader, split="test")
        precision, recall, f1 = self.compute_confusion_metrics(self.test_loader)
        print("\nTest Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        self.log_test_metrics_to_csv(precision, recall, f1)
        return precision, recall, f1


# Experiment Runner
def run_experiment(backbone_name, labels_map, pos_class, loaders, device, results_dir, epochs, learning_rate, batch_size):

    exp_name = f"{backbone_name}_pos{pos_class}_bs{batch_size}_lr{learning_rate}"
    save_dir = os.path.join(results_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nRunning experiment: {exp_name} on {device}\n")

    model = TumourClassifier3D(backbone_name=backbone_name, pretrained=True)

    trainer = Trainer(
        model,
        train_loader=loaders[0],
        val_loader=loaders[1],
        test_loader=loaders[2],
        labels_map=labels_map,
        pos_class=pos_class,
        device=device,
        save_dir=save_dir,
        patience=5,
        learning_rate=learning_rate,
        exp_name=exp_name
    )

    trainer.fit(epochs=epochs)
    trainer.test()


def main():
    torch.manual_seed(888)
    np.random.seed(888)

    parser = argparse.ArgumentParser(description="3D Brain Tumour Classification")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--positive_class",type=int,required=True,help="Class index to treat as positive in one-vs-all")

    parser.add_argument("--backbones", nargs="+", default=[
        "resnet18.a1_in1k",
        "resnext50_32x4d.fb_swsl_ig1b_ft_in1k",
        "tf_efficientnet_b2.in1k",
        "convnext_tiny",
        "densenet121"
    ])

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    shapes = get_all_shapes([args.train_path, args.val_path, args.test_path])
    target_shape = tuple(np.max(np.array(shapes), axis=0))
    print("Target shape:", target_shape)

    transform = AddChannel()

    train_dataset = CustomDataset(
        args.train_path,
        loader=lambda p: custom_loader(p, target_shape),
        transform=transform,
        extensions=[".npy"]
    )
    val_dataset = CustomDataset(
        args.val_path,
        loader=lambda p: custom_loader(p, target_shape),
        transform=transform,
        extensions=[".npy"]
    )
    test_dataset = CustomDataset(
        args.test_path,
        loader=lambda p: custom_loader(p, target_shape),
        transform=transform,
        extensions=[".npy"]
    )

    labels_map = {v: k for k, v in train_dataset.class_to_idx.items()}
    assert train_dataset.class_to_idx == val_dataset.class_to_idx
    assert train_dataset.class_to_idx == test_dataset.class_to_idx

    train_labels = [label for _, label in train_dataset.samples]


    class_counts = {
        1: sum(1 for y in train_labels if y == args.positive_class),
        0: sum(1 for y in train_labels if y != args.positive_class)
    }

    # weight = inverse frequency
    weights = []
    for y in train_labels:
        if y == args.positive_class:
            weights.append(1.0 / class_counts[1])
        else:
            weights.append(1.0 / class_counts[0])

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    loaders = (
        DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    )

    for backbone in args.backbones:
        run_experiment(
            backbone_name=backbone,
            loaders=loaders,
            device=device,
            results_dir=args.results_dir,
            labels_map=labels_map,
            pos_class=args.positive_class,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()