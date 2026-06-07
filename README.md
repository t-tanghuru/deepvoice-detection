# DeepVoice Detection

TTS 합성음성(Fake) vs 실제음성(Real) 탐지 모델

## 모델 구조
- Whisper(base) 특징 추출 + LCNN 분류
- val F1: 0.9983
- holdout F1: 0.83 (Roger, Will, Callum)

## 설치
pip install -r requirements.txt

## 사용법

### 단일 파일 추론
python predict.py audio.mp3

### 학습
python train.py

### 평가
python evaluate.py

## 데이터 구성
- Real: AIHUB 감성 음성 데이터 (25,555개 train / 4,445개 val)
- Fake: Edge TTS + gTTS (8,000개 샘플링) + ElevenLabs 다화자 (11,460개)

## holdout 평가 결과
| 화자 | 탐지율 |
|------|--------|
| Roger | 0.8538 |
| Will | 0.8000 |
| Callum | 0.4846 |
| **전체 F1** | **0.83** |
