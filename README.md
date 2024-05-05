# Contactless Fingerprint Anti-Spoofing: An Unsupervised Deep Learning Approach
This repository contains the implementation of an unsupervised deep learning approach for presentation attack detection (PAD) in contactless fingerprint recognition systems. The proposed method aims to address the limitations of existing methods, which often rely on both genuine (bonafide) and spoofed samples during the training phase, hindering their ability to generalize to unseen attack types.
# Overview
Contactless fingerprint recognition offers improved user convenience and hygiene compared to traditional contact-based systems. However, it is more susceptible to presentation attacks, such as photo paper, paper printouts, and various display attacks. This vulnerability poses a challenge for implementing contactless fingerprint biometrics securely.
The proposed approach combines an unsupervised autoencoder with a convolutional block attention module to learn the underlying patterns of genuine fingerprint images without exposure to any spoofed samples during training. This unsupervised learning strategy enables the model to detect presentation attacks effectively during the testing phase, even for previously unseen attack types.
# Results
Our method achieved an average Bona fide Presentation Attack Detection Error Rate (BPCER) of 0.96% with an Attack Presentation Classification Error Rate (APCER) of 1.6% when evaluated against various types of presentation attacks involving spoofed samples.
# Repository Structure
Data_Prep.py: Utility functions for data preprocessing, evaluation metrics, and visualization.

train.py: Script for training the unsupervised model on genuine fingerprint samples.

test.py: Script for evaluating the trained model on genuine and spoofed fingerprint samples.

requirements.txt: List of required Python packages and dependencies.

README.md: This file, providing an overview of the repository and instructions for running the code.
