# Dataset card

## Summary

This development dataset contains Spanish speech from 500 public figures from Spain. It includes 25,000 utterances obtained from 2,108 source videos, with exactly 50 utterances per speaker. The total duration is 16.59 hours, and the mean utterance duration is 2.39 seconds. The speaker list contains 250 people labelled as female and 250 as male.

## Intended use

The dataset was created for research on speaker recognition, speaker verification, speaker embeddings, and the adaptation of pretrained models. It also supports the study of same-video and different-video verification conditions.

## Construction

The pipeline combines face detection and recognition, face tracking, SyncNet active-speaker verification, voice activity detection, audio-quality checks, duplicate removal, and balanced segment selection.

## Limitations

The speakers are public figures and do not represent the complete population of Spain. Recording conditions vary across videos, and automatic filtering cannot completely remove identity or active-speaker errors. The demographic labels are descriptive and should not be used as prediction targets.

## Distribution

The repository provides code, source identifiers, metadata, validation reports, and evaluation results. It does not redistribute source videos, reference face images, extracted audio, or pretrained model weights. These materials remain subject to their original terms and must be obtained independently.
