import os, glob, random, torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb
from model import LCNN


class PTDataset(Dataset):
    def __init__(self, real_files, fake_files):
        self.files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        mel = torch.load(path)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        return mel, label


def pad_collate(batch):
    mels, labels = zip(*batch)
    max_len = max(m.shape[-1] for m in mels)
    padded = torch.zeros(len(mels), 1, mels[0].shape[1], max_len)
    for i, m in enumerate(mels):
        padded[i, 0, :, :m.shape[-1]] = m[0] if m.dim() == 3 else m
    return padded, torch.tensor(labels)


def train(config):
    LOCAL = config["data_dir"]
    DRIVE = "/content/drive/MyDrive/deepvoice"

    real_train = glob.glob(f"{LOCAL}/real_train/*.pt")
    real_val   = glob.glob(f"{LOCAL}/real_val/*.pt")
    fake_train = (
        glob.glob(f"{LOCAL}/fake_train_sampled/*.pt") +
        glob.glob(f"{LOCAL}/elevenlabs_train/*.pt") +
        glob.glob(f"{LOCAL}/elevenlabs_train_add2/*.pt") +
        glob.glob(f"{LOCAL}/elevenlabs_train_add3/*.pt")
    )
    fake_val = (
        glob.glob(f"{LOCAL}/fake_val/*.pt") +
        glob.glob(f"{LOCAL}/elevenlabs_val/*.pt")
    )

    print(f"train - real: {len(real_train)}, fake: {len(fake_train)}")
    print(f"val   - real: {len(real_val)}, fake: {len(fake_val)}")

    train_loader = DataLoader(PTDataset(real_train, fake_train),
                              batch_size=32, shuffle=True, num_workers=4, collate_fn=pad_collate)
    val_loader   = DataLoader(PTDataset(real_val, fake_val),
                              batch_size=32, shuffle=False, num_workers=4, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    wandb.init(project="deepvoice", name=config["run_name"])
    best_f1 = 0

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        for step, (mel, label) in enumerate(train_loader):
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(mel)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | loss: {loss.item():.4f}")
                wandb.log({"step_loss": loss.item()})

        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for mel, label in val_loader:
                mel, label = mel.to(device), label.to(device)
                out = model(mel)
                val_loss += criterion(out, label).item()
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        f1  = f1_score(all_labels, all_preds)
        pr  = precision_score(all_labels, all_preds)
        rc  = recall_score(all_labels, all_preds)

        wandb.log({"train_loss": train_loss/len(train_loader),
                   "val_loss": val_loss/len(val_loader),
                   "val_acc": acc, "val_f1": f1,
                   "val_precision": pr, "val_recall": rc,
                   "epoch": epoch+1})

        print(f"Epoch {epoch+1}/{config['epochs']} | loss: {train_loss/len(train_loader):.4f} | val_acc: {acc:.4f} | val_f1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{DRIVE}/best_model_tts_v6.pt")
            print(f"  → best model 저장 (f1: {f1:.4f})")

    wandb.finish()
    print(f"학습 완료 | best f1: {best_f1:.4f}")


if __name__ == "__main__":
    config = {
        "data_dir": "/content/features",
        "run_name": "tts_lcnn_v6",
        "epochs": 30,
    }
    train(config)
