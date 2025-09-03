# README.md

# Oloche-AI Cardiologist: ECG Arrhythmia Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Hugging Face](https://img.shields.io/badge/-Hugging%20Face%20Space-yellow)

A state-of-the-art deep learning system for automated detection and classification of cardiac arrhythmias from raw single-lead ECG signals. This project implements a 1D Convolutional Neural Network (CNN) trained on the MIT-BIH Arrhythmia Database to provide real-time, explainable diagnostic support.

**Access the website of this work:** [https://huggingface.co/spaces/eijeoloche1/ECG_Arrhythmia_Classification](https://huggingface.co/spaces/eijeoloche1/ECG_Arrhythmia_Classification)  
** Researcher:** *Eije, Oloche Celestine

## Project Overview

Cardiac arrhythmias are a leading cause of morbidity and mortality worldwide. This project leverages deep learning to create an assistive tool that automatically classifies heartbeats into five distinct arrhythmia categories, providing clinicians and researchers with instantaneous analysis.

The system demonstrates a complete pipeline from model design and training to deployment as a publicly accessible web application, showcasing the potential of AI-driven diagnostic tools in biomedical engineering.

## Dataset

This system was trained and evaluated on the **MIT-BIH Arrhythmia Database** [1], the gold-standard benchmark for evaluating arrhythmia detectors.

- **Source:** PhysioNet (https://physionet.org/content/mitdb/1.0.0/)
- **Contents:** 48 half-hour two-lead ECG recordings from 47 subjects
- **Annotations:** Over 110,000 beats annotated by expert cardiologists
- **Preprocessing:** Model trained on single-lead (MLII) signals with heartbeat extraction and normalization

*[1] Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.*

##  Model Architecture

The system uses a custom 1D Convolutional Neural Network implemented in PyTorch:

**Network Architecture (ECGCNN Class):**
- **Input Layer:** Preprocessed 1D ECG signals of 180 samples
- **Feature Extraction:** Four convolutional blocks with:
  - 1D Convolutional Layers (increasing filters: 32, 64, 128, 256)
  - Batch Normalization
  - ReLU Activation
  - Max Pooling
- **Classifier Head:** Fully connected layers with dropout (p=0.5)
- **Output Layer:** 5 neurons with Softmax activation

## Arrhythmia Classes

The system classifies heartbeats into five categories according to AAMI standards:

| Class | AAMI Symbol | Description |
| :---: | :---: | :--- |
| **N** | N | Normal Beat |
| **L** | N | Left Bundle Branch Block Beat |
| **R** | N | Right Bundle Branch Block Beat |
| **V** | V | Premature Ventricular Contraction |
| **A** | S | Atrial Premature Beat |

## How to use this work

1.  **Access Demo:** Visit the [Hugging Face Space](https://huggingface.co/spaces/eijeoloche1/ECG_Arrhythmia_Classification)
2.  **Input Data:** Provide 180 comma-separated numerical values representing a single ECG heartbeat
3.  **Analyze:** Click "Analyze ECG" or press Enter
4.  **Interpret Results:** Review prediction, confidence score, probability distribution, and signal visualization

Dependencies: torch>=2.0.0, gradio>=4.0.0, numpy>=1.21.0, scikit-learn>=1.0.0, matplotlib>=3.5.0

## Disclaimer
This tool is a prototype for RESEARCH AND DEMONSTRATION PURPOSES ONLY. It is NOT a certified medical device.
Not for diagnostic use - always consult qualified healthcare professionals
Predictions are computational opinions, not definitive diagnoses
Author assumes no liability for any use or misuse of this application

## Researcher
Eije, Oloche Celestine

## License
This project is licensed under the MIT License. See LICENSE file for details.
