# 🎙️ DeepVoice Detection
딥러닝 기반 보이스피싱 및 AI 합성 음성 탐지 시스템

## 📌 프로젝트 개요
실제 음성과 AI 합성 음성을 구별하는 웹 기반 탐지 서비스.
안드로이드 전용 실시간 통화 분석과 달리, **브라우저에서 음성 파일을 업로드**하면 즉시 판별 가능.

## 🎯 목표
- 탐지 정확도 90% 이상, F1-score 0.90 이상
- 음성 1건당 분석 응답 시간 3초 이내
- 실제/합성 음성 각각 5,000개 이상 학습

## 🛠️ 개발 환경
| 항목 | 버전 |
|------|------|
| OS | Ubuntu 20.04 (WSL2) |
| Python | 3.10 |
| PyTorch | 2.11.0+cpu |
| librosa | 0.11.0 |

## 📦 설치 방법
```bash
conda create -n deepvoice python=3.10
conda activate deepvoice
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install librosa numpy pandas scikit-learn matplotlib
pip install gtts pydub
```

## 📁 프로젝트 구조
```
deepvoice-detection/
├── data/
│   ├── real/        # AIHub 실제 음성 데이터
│   └── fake/        # 합성 음성 데이터
├── preprocess/
│   ├── augment_tts.py      # gTTS + pydub 합성 음성 생성
│   └── feature_extract.py  # MFCC, Mel 스펙트로그램 추출
├── model/
│   └── cnn_classifier.py   # CNN 탐지 모델
├── train.py         # 학습 스크립트
└── inference.py     # 추론 스크립트
```

## 🗂️ 데이터 구성
| 종류 | 방법 | 목표 수량 |
|------|------|----------|
| 실제 음성 | AIHub 자유대화 음성 | 5,000개 |
| TTS 합성 | gTTS + pydub 증강 | 2,500개 |
| VC 합성 | Google Colab RVC | 2,500개 |

## 🧠 모델 구조
- **입력**: Mel 스펙트로그램 (128 x 128)
- **구조**: 3층 CNN + Fully Connected Layer
- **출력**: 실제(0) / 합성(1) 이진 분류

## 📅 개발 일지
### 1주차
#(2026.04.05)
- WSL2 + Ubuntu 20.04 설치
- Miniconda 및 deepvoice 환경 구성
- PyTorch, librosa 등 패키지 설치
- AIHub 자유대화 음성(일반남여) 샘플 2,000개 확보
- 깃허브 레포 생성 및 초기 구조 설정
#(2026.04.06)
- gTTS 설치 및 한국어 음성 생성 테스트
- gTTS + pydub 데이터 증강 스크립트 구현 (40개 생성)
- MFCC 및 Mel 스펙트로그램 특징 추출 코드 구현
- CNN 딥보이스 탐지 모델 설계

## 🔗 참고 자료
- [AIHub 한국어 음성 데이터](https://aihub.or.kr)
- [librosa 공식 문서](https://librosa.org)
- [PyTorch 공식 문서](https://pytorch.org)
