# VoxCeleb-ESP Dataset

Code and reproducibility resources for building and evaluating a VoxCeleb-style development dataset for Spanish speaker recognition.

## Overview

This project creates a development dataset from online videos of 500 public figures from Spain. The pipeline combines face recognition, face tracking, active-speaker verification, voice activity detection, audio-quality checks, duplicate removal, and balanced segment selection.

The dataset was created to support speaker recognition research and the adaptation of pretrained speaker verification models. It complements the evaluation-oriented VoxCeleb-ESP resource with a separate set of speakers for development and training.

## Dataset summary

| Property | Value |
|---|---:|
| Speakers | 500 |
| Female / male labels | 250 / 250 |
| Utterances | 25,000 |
| Utterances per speaker | 50 |
| Source videos represented | 2,108 |
| Total duration | 16.59 hours |
| Mean utterance duration | 2.39 seconds |
| Audio format | Mono WAV, 16 kHz |
| Integrity errors | 0 |

Every speaker is represented by clips from at least two source videos.

## Repository structure

```text
configs/          Public pipeline configuration
evaluation/       Speaker-disjoint experimental split
manifests/        Speaker, video, and exclusion manifests
metadata/         Final speaker-level summary
models/           Instructions for obtaining external model files
references/       Instructions for preparing reference face images
reports_final/    Final validation reports
results/          Global and gender-based evaluation results
scripts/          Configuration, setup, execution, and evaluation tools
src/              Dataset creation pipeline
```

The repository does not contain source videos, extracted WAV files, reference face images, or pretrained model weights.

## Pipeline

The main processing stages are:

1. Validate the speaker and video manifest
2. Check the source videos and reference images
3. Process long videos in temporal windows
4. Detect, track, and identify the target face
5. Verify that the visible person is speaking using SyncNet
6. Apply voice activity and audio-quality filters
7. Remove exact and temporal duplicates
8. Rank candidates and select 50 clips per speaker
9. Export metadata, trial lists, and validation reports

## Requirements

The pipeline was developed for Linux with Python 3.10, FFmpeg, FFprobe, CUDA, and NVIDIA GPU.

Create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and ONNX Runtime may require versions compatible with the local CUDA installation.

Install the official SyncNet implementation and pretrained weights:

```bash
bash scripts/setup_syncnet.sh
```

InsightFace obtains the `buffalo_l` model pack through its standard setup process.

## Input preparation

Place each source video at:

```text
data/videos/<youtube_id>.mp4
```

Place two reference images for each speaker at the paths recorded in `manifests/speakers.csv`, for example:

```text
references/id000001_1.jpg
references/id000001_2.jpg
```

Reference images and source videos must be obtained independently and used according to their original terms.

## Configuration

Generate the complete configuration from the public manifests:

```bash
python scripts/build_config_from_manifests.py \
  --base-config configs/pipeline.yaml \
  --speakers manifests/speakers.csv \
  --videos manifests/videos.csv \
  --exclude-speakers-file manifests/voxceleb_esp_test_speakers.txt \
  --min-videos-per-speaker 2 \
  --out config.generated.yaml
```

Check the required files and dependencies before processing:

```bash
PYTHONPATH=src python scripts/preflight_pipeline.py \
  --config config.generated.yaml
```

## Running the pipeline

Run the complete workflow:

```bash
bash scripts/run_pipeline.sh config.generated.yaml
```

The script executes the following stages:

```text
candidates - approve-auto - export - validate
```

A stage can also be executed separately:

```bash
PYTHONPATH=src python src/TFM_pipeline_voxceleb_style.py \
  --config config.generated.yaml \
  --stage candidates
```

Candidate records and completed processing units are saved incrementally, allowing interrupted runs to continue.

## Evaluation

The 500 speakers were divided at speaker level into:

| Partition | Speakers | Utterances | Purpose |
|---|---:|---:|---|
| Training | 400 | 20,000 | Model adaptation |
| Validation | 50 | 2,500 | Checkpoint selection |
| Test | 50 | 2,500 | Final evaluation |

There is no speaker overlap between the partitions. Trial A uses same-video target pairs, while Trial B uses different-video target pairs.

Two VoxCeleb2-pretrained models were evaluated before and after adaptation:

| Model | Trial A base | Trial A adapted | Trial B base | Trial B adapted |
|---|---:|---:|---:|---:|
| ResNetSE34L | 7.664% | **5.908%** | 14.008% | **10.426%** |
| ResNetSE34V2 | 4.520% | **4.186%** | **9.350%** | 9.720% |

These values are internal Equal Error Rate results. They are not directly comparable with the official VoxCeleb-ESP benchmark because different speakers, clips, and trial lists are used. Complete EER and minDCF values are available in `results/`.

## Limitations

The dataset contains public figures and does not represent the complete population of Spain. Recording conditions vary across videos, and automatic filtering cannot completely remove identity or active-speaker errors. Gender, region, and professional category are descriptive metadata and are not prediction targets.

## Distribution

This repository follows a pointer-based approach. It provides code, source identifiers, metadata summaries, validation reports, and evaluation results. It does not redistribute:

- YouTube videos
- Extracted speech recordings
- Reference face images
- Pretrained model weights
- Cookies, credentials, logs, or temporary files

## Licence

The licences for the original code and released metadata must be confirmed before making the repository public. Third-party videos, images, models, and code remain subject to their original terms.

