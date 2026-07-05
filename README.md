# DeepVoice Detection

A web-based deepfake-voice detection system that distinguishes TTS-synthesized and RVC-converted speech from real human speech.

The model feeds Whisper base encoder representations into an LCNN-style classifier, running two independent binary detectors for TTS-synthesized speech and RVC-converted speech.

## Model Architecture

- Architecture: Whisper base encoder + LCNN-style classifier
- Input flow: Audio → Whisper-compatible log-Mel → Whisper base encoder → encoder representation → classifier
- The log-Mel is not used as a standalone classification feature; it serves only as the input format for the Whisper encoder, and classification is performed on the encoder representation.
- The TTS detector and RVC detector are independent binary classifiers (not a single multi-class model).
- Training: Adam, lr=1e-4 / TTS 30 epochs, RVC 10 epochs
- Operating thresholds: TTS 0.28, RVC 0.20 (internal decision criteria)
- Evaluation metrics: F1-score, EER, normalized min-DCF, holdout performance

## Performance

### TTS Detection

| Evaluation | Real data | Fake data | F1 | EER | norm. min-DCF |
| --- | --- | --- | --- | --- | --- |
| Base evaluation | real_val 4,445 | fake eval set 1,289 | 1.0000 | 0.00% | 0.0000 |
| Real-speech holdout | real_raw_holdout 3,780 | fake eval set 1,289 | 0.9996 | 0.01% | 0.0003 |

The base-evaluation F1 of 1.0000 results from a validation set where the real/fake score distributions are fully separated, so it is not interpreted as error-free performance in general conditions. A separate real_raw_holdout (unseen during training) was evaluated to verify generalization.

### RVC Voice-Conversion Detection

| Evaluation | Real data | RVC data | F1 | EER | norm. min-DCF |
| --- | --- | --- | --- | --- | --- |
| Validation | rvc_real_val 200 | KANE·Nell_V2 val 200 | 1.0000 | 0.00% | 0.0000 |
| Joonjong holdout | real_raw_holdout 3,780 | Joonjong 541 | 0.9881 | 0.00% | 0.0000 |
| NELL_KLM43x4 holdout | real_raw_holdout 3,780 | NELL_KLM43x4 566 | 0.9886 | 0.18% | 0.0019 |
| Combined holdout | real_raw_holdout 3,780 | Joonjong + NELL_KLM43x4 1,107 | 0.9942 | 0.12% | 0.0019 |

The RVC detector was retrained separately using the same Whisper encoder + LCNN-style classifier architecture as the TTS detector.

## Dataset

### Real speech

- Source: AIHub free-conversation speech (general male/female)
- Counts: real_sampled 30,000 / real_val 4,445 / real_raw_holdout 3,780
- Preprocessing: normalized to 16 kHz mono, split at the speaker level
- Augmentation: MP3 compression, telephone-quality, noise, speed, volume (also applied to real speech to keep real/fake balance)
- Final training input: 38,501 augmented real_sampled features

### Fake (synthesized / converted) speech

- TTS: Edge TTS·gTTS (fake_train 23,031) combined with ElevenLabs multi-speaker sets for a total of 38,501
- TTS eval set: fake_val 639 + elevenlabs_val 260 + holdout_v2 390 = 1,289
- RVC: rvc_fake_train_kane_nell 800 + rvc_real_train_balanced 800 (training); rvc_fake_val_kane_nell 200 + rvc_real_val_balanced 200 (validation)
- RVC holdout: Joonjong 541, NELL_KLM43x4 566 (unseen during training)

Final TTS training was balanced so that the augmented real and fake inputs each contained 38,501 samples.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single-file inference

```bash
python predict.py audio.mp3
```

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

## Related Repository

- Web (frontend / backend): https://github.com/jeonghanhee/deepvoice-detection-web

## Limitations and Future Work

- Current evaluation is based on self-built datasets and self-defined holdouts. Comparative evaluation on public datasets such as ASVspoof, ADD, and CFAD is needed.
- The RVC holdouts (Joonjong, NELL_KLM43x4) are also based on self-built converted speech, so generalization to diverse unseen TTS/VC engines remains future work.

---

# DeepVoice Detection

웹 기반 딥보이스 판별 시스템 — TTS 합성음성 및 RVC 변조음성 vs 실제음성 탐지 모델

Whisper base encoder representation을 LCNN-style classifier로 분류하는 구조로, TTS 합성음성과 RVC 변조음성을 각각 독립적으로 탐지합니다.

## 모델 구조

- 아키텍처: Whisper base encoder + LCNN-style classifier
- 입력 흐름: Audio → Whisper-compatible log-Mel → Whisper base encoder → encoder representation → classifier
- log-Mel은 단독 분류 특징이 아니라 Whisper encoder의 입력 형식으로만 사용하며, 실제 분류에는 encoder representation을 사용
- TTS 탐지 모델과 RVC 탐지 모델은 각각 독립적인 이진 분류기 (하나의 다중분류기가 아님)
- 학습 설정: Adam, lr=1e-4 / TTS 30 epoch, RVC 10 epoch
- 운영 threshold: TTS 0.28, RVC 0.20 (내부 판정 기준)
- 평가 지표: F1-score, EER, normalized min-DCF, holdout 성능

## 성능

### TTS 탐지

| 평가 구분 | Real 데이터 | Fake 데이터 | F1 | EER | norm. min-DCF |
| --- | --- | --- | --- | --- | --- |
| 기본 평가 | real_val 4,445 | fake 평가셋 1,289 | 1.0000 | 0.00% | 0.0000 |
| 실제음성 holdout | real_raw_holdout 3,780 | fake 평가셋 1,289 | 0.9996 | 0.01% | 0.0003 |

기본 평가의 F1 1.0000은 해당 검증셋에서 real/fake score 분포가 완전히 분리되어 산출된 값이므로, 일반 환경에서 오류가 없다는 의미로 해석하지 않습니다. 학습에 사용하지 않은 real_raw_holdout을 별도로 평가하여 일반화 성능을 보완적으로 확인하였습니다.

### RVC 변환음성 탐지

| 평가 구분 | Real 데이터 | RVC 데이터 | F1 | EER | norm. min-DCF |
| --- | --- | --- | --- | --- | --- |
| Validation | rvc_real_val 200 | KANE·Nell_V2 val 200 | 1.0000 | 0.00% | 0.0000 |
| Joonjong holdout | real_raw_holdout 3,780 | Joonjong 541 | 0.9881 | 0.00% | 0.0000 |
| NELL_KLM43x4 holdout | real_raw_holdout 3,780 | NELL_KLM43x4 566 | 0.9886 | 0.18% | 0.0019 |
| 통합 holdout | real_raw_holdout 3,780 | Joonjong + NELL_KLM43x4 1,107 | 0.9942 | 0.12% | 0.0019 |

RVC 탐지 모델은 TTS 탐지와 동일한 Whisper encoder + LCNN-style classifier 구조로 별도 재학습하였습니다.

## 데이터 구성

### Real (실제 음성)

- 출처: AIHub 자유대화 음성 (일반남여)
- 수량: real_sampled 30,000 / real_val 4,445 / real_raw_holdout 3,780
- 전처리: 16 kHz mono 정규화, 화자 단위 분리
- 증강: MP3 압축·전화음질·노이즈·속도·볼륨 (real/fake 균형을 위해 real에도 동일 계열 적용)
- 최종 학습 입력: real_sampled 증강 feature 38,501

### Fake (합성·변조 음성)

- TTS: Edge TTS·gTTS (fake_train 23,031) + ElevenLabs 다화자 계열을 합쳐 총 38,501로 구성
- TTS 평가셋: fake_val 639 + elevenlabs_val 260 + holdout_v2 390 = 1,289
- RVC: rvc_fake_train_kane_nell 800 + rvc_real_train_balanced 800 (학습), rvc_fake_val_kane_nell 200 + rvc_real_val_balanced 200 (검증)
- RVC holdout: Joonjong 541, NELL_KLM43x4 566 (학습 미사용)

최종 TTS 학습은 real 증강 입력과 fake 증강 입력을 각각 38,501로 맞추어 균형화한 뒤 진행하였습니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 단일 파일 추론

```bash
python predict.py audio.mp3
```

### 학습

```bash
python train.py
```

### 평가

```bash
python evaluate.py
```

## 관련 저장소

- 웹 (프론트엔드·백엔드): https://github.com/jeonghanhee/deepvoice-detection-web

## 한계 및 향후 계획

- 현재 평가는 자체 구축 데이터셋과 자체 holdout 기준이므로, ASVspoof·ADD·CFAD 등 공인 데이터셋 기반 비교 평가가 필요합니다.
- RVC holdout(Joonjong, NELL_KLM43x4)도 자체 구축 변환 음성 기준이므로, 다양한 unseen TTS·VC 엔진에 대한 일반화 검증이 후속 과제로 남아 있습니다.

