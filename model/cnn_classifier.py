import torch
import torch.nn as nn

class DeepVoiceCNN(nn.Module):
    def __init__(self):
        super(DeepVoiceCNN, self).__init__()
        
        # 특징 추출 레이어
        self.conv_layers = nn.Sequential(
            # 1번째 Conv 블록
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # 2번째 Conv 블록
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # 3번째 Conv 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # 분류 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 0: 실제, 1: 합성
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    model = DeepVoiceCNN()
    print(model)
    
    # 테스트 입력 (배치 1, 채널 1, 128x128)
    test_input = torch.randn(1, 1, 128, 128)
    output = model(test_input)
    print(f"입력 shape: {test_input.shape}")
    print(f"출력 shape: {output.shape}")
    print("모델 설계 완료")
