# DeepVoice Detection

딥보이스 피싱 탐지를 위한 음성 위조 탐지 모델 코드입니다.

## 최종 TTS 탐지 모델 구조

최종 모델은 Whisper base encoder와 LCNN-style classifier를 결합한 구조입니다.

Audio -> 16kHz mono -> Whisper-compatible log-Mel spectrogram -> Whisper base encoder -> encoder representation -> LCNN-style classifier -> REAL / FAKE

## 주요 파일

| 파일 | 역할 |
|---|---|
| model/model.py | MFM, LCNN, WhisperEncoderLCNN 모델 구조 정의 |
| model/scripts/extract_features.py | 음성 전처리 및 log-Mel feature 추출 |
| model/scripts/train.py | 기존 LCNN 모델 학습 |
| model/scripts/evaluate.py | 기존 LCNN 모델 평가 |
| model/scripts/predict.py | 기존 LCNN 모델 단일 오디오 추론 |
| model/scripts/train_whisper_encoder.py | 최종 Whisper encoder 기반 모델 학습 |
| model/scripts/evaluate_whisper_encoder.py | 최종 모델 평가, F1/EER/min-DCF 계산 |
| model/scripts/predict_whisper_encoder.py | 최종 모델 단일 오디오 추론 |
| model/scripts/logmel_gradient_heatmap_whisper_encoder.py | 입력 log-Mel gradient 기반 판별 근거 heatmap 생성 |

## 주의

모델 checkpoint(.pt), 음성 데이터, feature 파일, API key는 GitHub에 업로드하지 않습니다.
