# DeepVoice Detection

This repository contains the AI model code used for a DeepVoice detection project.
The project covers both TTS-based synthetic speech detection and RVC-based voice conversion detection.

This repository includes model definitions, training scripts, evaluation scripts, single-audio prediction scripts, Colab notebooks, and a log-Mel gradient heatmap script.

> Model checkpoints (`.pt`), raw audio data, extracted feature files, and API keys are not uploaded to GitHub because of file size and security limitations.

## 1. Final Model Architecture

The final model combines a Whisper base encoder with an LCNN-style classifier.

```text
Audio
-> 16 kHz mono
-> Whisper-compatible log-Mel spectrogram
-> Whisper base encoder
-> encoder representation
-> LCNN-style classifier
-> REAL / FAKE
```

The log-Mel spectrogram is not used as the final classification feature by itself.
It is used as the input format for the Whisper base encoder.
The classifier uses the encoder representation produced by the Whisper encoder.

## 2. Final Evaluation Summary

### 2.1 TTS Detection Model

- Model: `best_model_tts_whisper_encoder_lcnn.pt`
- Operating threshold: `0.28`
- Validation F1: `1.0000`
- Validation EER: `0.00%`
- Validation normalized min-DCF: `0.0000`
- Separate holdout evaluation: `real_raw_holdout` 3,780 files + fake evaluation set 1,289 files
- Separate holdout F1: `0.9996`
- Separate holdout EER: `0.01%`
- Separate holdout normalized min-DCF: `0.0003`

The validation F1 of 1.0000 and EER of 0.00% mean that the real/fake score distributions were completely separated on that validation set.
This does not mean that the model has zero error in all real-world environments.
Therefore, an additional evaluation was performed using `real_raw_holdout`, which was not used for training.

### 2.2 RVC Detection Model

- Model: `best_model_rvc_whisper_encoder_lcnn.pt`
- Evaluation threshold: `0.20`
- Validation F1: `1.0000`
- Validation EER: `0.00%`
- Validation normalized min-DCF: `0.0000`
- Combined Joonjong + NELL_KLM43x4 holdout F1: `0.9942`
- Combined holdout EER: `0.12%`
- Combined holdout normalized min-DCF: `0.0019`
- Combined holdout best threshold scan: `0.35`

The RVC detection model was separately retrained using the same Whisper encoder + LCNN-style classifier structure.
The RVC result is based on a self-built RVC dataset.
Evaluation on public VC datasets remains future work.

## 3. Dataset Summary

### 3.1 Real Speech

- AIHub free conversation speech data
- `real_sampled`: 30,000 real speech files for training
- `real_val`: 4,445 real speech files for validation
- `real_raw_holdout`: 3,780 real speech files not used for training
- Real training data was augmented with MP3 compression, telephone-quality conversion, noise, speed change, and volume change.

### 3.2 TTS Synthetic Speech

- Edge TTS, gTTS, and ElevenLabs synthetic speech data
- Final fake training features: 38,501 files
- Fake evaluation set: `fake_val` 639 + `elevenlabs_val` 260 + `holdout_v2` 390 = 1,289 files

### 3.3 RVC Voice Conversion Data

- Training and validation: `vc_fake/KANE`, `vc_fake/Nell_V2`
- Holdout: `vc_holdout/Joonjong` 541 files, `vc_holdout/NELL_KLM43x4` 566 files

## 4. Preprocessing and Augmentation

Preprocessing was designed not as arbitrary manipulation, but as a way to unify input conditions and simulate real service environments.

- 16 kHz mono conversion: unifies sampling rate and audio channel format
- MP3 compression: simulates lossy compressed audio uploaded by users
- Telephone-quality conversion: simulates an 8 kHz band-limited call environment
- Noise addition: reflects background noise in recording environments
- Speed change: reflects speech rate variation
- Volume change: reflects input loudness variation

At an earlier stage, augmentation was applied mainly to fake data.
This could cause shortcut learning, where the model learns audio quality differences instead of synthetic speech characteristics.
To reduce this risk, similar augmentation was also applied to real speech, and the final real/fake training feature counts were balanced at 38,501 each.

## 5. Main Files

| File | Purpose |
|---|---|
| `model/model.py` | Defines MFM, LCNN, and WhisperEncoderLCNN models |
| `model/scripts/extract_features.py` | Audio preprocessing and Whisper-compatible log-Mel generation |
| `model/scripts/train.py` | Legacy LCNN training script |
| `model/scripts/evaluate.py` | Legacy LCNN evaluation script |
| `model/scripts/predict.py` | Legacy LCNN single-audio prediction script |
| `model/scripts/train_whisper_encoder.py` | Whisper encoder-based TTS/RVC model training script |
| `model/scripts/evaluate_whisper_encoder.py` | TTS/RVC evaluation script with F1, EER, and min-DCF |
| `model/scripts/predict_whisper_encoder.py` | Whisper encoder-based single-audio prediction script |
| `model/scripts/logmel_gradient_heatmap_whisper_encoder.py` | Generates an input log-Mel gradient heatmap |
| `notebooks/rvc_whisper_encoder_retrain_colab.ipynb` | Colab notebook for RVC retraining and evaluation |
| `notebooks/rvc_result_viewer_colab.ipynb` | Colab notebook for summarizing RVC result files |

## 6. Single Audio Prediction Example

```bash
python model/scripts/predict_whisper_encoder.py input.wav   --model-path /path/to/best_model_tts_whisper_encoder_lcnn.pt   --threshold 0.28
```

## 7. Evaluation Example

```bash
python model/scripts/evaluate_whisper_encoder.py   --data-dir /path/to/whisper_features   --model-path /path/to/best_model_tts_whisper_encoder_lcnn.pt   --real-dirs real_val   --fake-dirs fake_val,elevenlabs_val,holdout_v2   --batch-size 32   --num-workers 0
```

## 8. RVC Notebook Workflow

The RVC workflow is documented in the Colab notebooks under `notebooks/`.

- `rvc_whisper_encoder_retrain_colab.ipynb`: creates balanced RVC train/validation splits, trains the Whisper encoder + LCNN-style classifier model, and evaluates holdout RVC speakers.
- `rvc_result_viewer_colab.ipynb`: reads saved evaluation text files and summarizes F1, EER, normalized min-DCF, and confusion matrices.

## 9. Limitations and Future Work

- Current results are based on self-built datasets and self-built holdout sets.
- Evaluation with public spoofing datasets such as ASVspoof, ADD, and CFAD is needed.
- Additional unseen TTS/VC engines and real call-recording conditions should be tested.
- For web deployment, formal concurrent-user load testing and long-running monitoring are still needed.
