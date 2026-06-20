import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            MFM(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, 3, padding=1),
            MFM(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 128, 3, padding=1),
            MFM(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def pad_or_trim_mel(mel, target_frames=3000):
    """Pad or trim Whisper log-Mel tensors to Whisper encoder's expected length."""
    if mel.dim() == 4:
        mel = mel.squeeze(1)
    if mel.dim() != 3:
        raise ValueError(f"Expected mel shape (batch, n_mels, frames), got {tuple(mel.shape)}")

    frames = mel.shape[-1]
    if frames < target_frames:
        mel = F.pad(mel, (0, target_frames - frames))
    elif frames > target_frames:
        mel = mel[..., :target_frames]
    return mel


class WhisperEncoderLCNN(nn.Module):
    """Frozen Whisper encoder + lightweight LCNN-style temporal classifier.

    Input is a Whisper-compatible log-Mel spectrogram with shape:
    - (batch, 80, frames), or
    - (batch, 1, 80, frames)

    The model pads/trims frames to 3000, runs the Whisper encoder, then
    classifies the encoder sequence output.
    """

    def __init__(self, whisper_size="base", freeze_whisper=True, dropout=0.5):
        super().__init__()
        import whisper

        self.whisper_size = whisper_size
        self.freeze_whisper = freeze_whisper
        whisper_model = whisper.load_model(whisper_size)
        self.encoder = whisper_model.encoder

        if freeze_whisper:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            MFM(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            MFM(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def encode(self, mel):
        mel = pad_or_trim_mel(mel, target_frames=3000)
        mel = mel.to(next(self.encoder.parameters()).dtype)
        if self.freeze_whisper:
            with torch.no_grad():
                encoded = self.encoder(mel)
        else:
            encoded = self.encoder(mel)
        return encoded

    def forward(self, mel):
        encoded = self.encode(mel)  # (batch, frames/2, 512)
        features = encoded.transpose(1, 2)  # (batch, 512, frames/2)
        return self.classifier(features)
