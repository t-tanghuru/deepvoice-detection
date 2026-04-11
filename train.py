import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from preprocess.feature_extract import extract_melspectrogram
from model.cnn_classifier import DeepVoiceCNN

class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = extract_melspectrogram(self.file_paths[idx])
        mel = torch.FloatTensor(mel).unsqueeze(0)  # (1, 128, 128)
        label = self.labels[idx]
        return mel, label

def load_data():
    real_dir = "data/real_normalized"
    fake_dir = "data/fake_normalized"
    
    file_paths, labels = [], []
    
    # 실제 음성 (label=0)
    for f in os.listdir(real_dir):
        if f.endswith('.wav') or f.endswith('.mp3'):
            file_paths.append(os.path.join(real_dir, f))
            labels.append(0)
    
    # 합성 음성 (label=1)
    for f in os.listdir(fake_dir):
        if f.endswith('.wav') or f.endswith('.mp3'):
            file_paths.append(os.path.join(fake_dir, f))
            labels.append(1)
    
    print(f"실제 음성: {labels.count(0)}개")
    print(f"합성 음성: {labels.count(1)}개")
    return file_paths, labels

def train():
    # 데이터 로드
    file_paths, labels = load_data()
    
    # 학습/검증 분리 (8:2)
    X_train, X_val, y_train, y_val = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = VoiceDataset(X_train, y_train)
    val_dataset = VoiceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 모델, 손실함수, 옵티마이저
    model = DeepVoiceCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 학습
    epochs = 30
    best_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for mels, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for mels, labels_batch in val_loader:
                outputs = model(mels)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels_batch.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        # 최고 모델 저장
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "model/best_model.pt")
            print(f"모델 저장 (F1: {f1:.4f})")
    
    print(f"학습 완료! 최고 F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()
