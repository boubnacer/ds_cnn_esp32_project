# Manual Download Guide

## Option A — Use the pretrained TFLite model directly (EASIEST)

The DS-CNN model is already available as a TFLite file in the MLCommons repo:

1. Go to: https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting/trained_models
2. Download `ds_cnn_s_quantized.tflite`
3. Place it in: `models/ds_cnn_float32.tflite`

## Option B — Train from scratch using TensorFlow's tutorial

TensorFlow has an official simple audio recognition tutorial:
https://www.tensorflow.org/tutorials/audio/simple_audio

Run their notebook — at the end you'll have a trained model you can export.

## Option C — Use the ARM ML-KWS-for-MCU repo

```bash
git clone https://github.com/ARM-software/ML-KWS-for-MCU
cd ML-KWS-for-MCU
# Follow their README to get the pretrained DS-CNN models
```

## Speech Commands Dataset

Download manually from:
https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Extract to: `data/test_samples/`

## Dataset Structure Expected
```
data/test_samples/
├── yes/   (contains .wav files)
├── no/
├── up/
├── down/
├── left/
├── right/
├── on/
├── off/
├── stop/
├── go/
└── _background_noise_/
```
