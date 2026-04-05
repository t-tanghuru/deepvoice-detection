cd ~/deepvoice-detection
cat > README.md << 'EOF'
# 🎙️ DeepVoice Detection
딥러닝 기반 보이스피싱 및 AI 합성 음성 탐지 시스템

## 개발 환경
| 항목 | 버전 |
|------|------|
| OS | Ubuntu 20.04 (WSL2) |
| Python | 3.10 |
| PyTorch | 2.11.0+cpu |
| librosa | 0.11.0 |

## 설치 방법
```bash
conda create -n deepvoice python=3.10
conda activate deepvoice
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install librosa numpy pandas scikit-learn matplotlib
```

## 프로젝트 구조
deepvoice-detection/
├── data/           # 음성 데이터 (gitignore)
├── preprocess/     # 전처리 코드
├── model/          # 딥러닝 모델
├── train.py        # 학습 스크립트
└── inference.py    # 추론 스크립트
