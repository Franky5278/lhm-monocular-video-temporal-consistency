# Temporally Consistent LHM-based Animatable Human Reconstruction

## Overview

This repository documents a URECA research project investigating temporal consistency in LHM-based animatable human reconstruction from monocular videos.

The project studies how a single-image reconstruction pipeline can be extended toward temporally stable video-based avatar reconstruction through motion smoothing, keyframe selection, rendering restoration and quantitative evaluation.

## Research Objectives

- Reproduce representative animatable human reconstruction baselines
- Investigate temporal instability in frame-by-frame reconstruction
- Improve motion consistency across monocular video sequences
- Evaluate the trade-off between temporal smoothness and motion fidelity
- Explore multi-keyframe and render-restoration strategies

## Baselines Reproduced

- LHM
- GaussianAvatar
- 3DGS-Avatar

## Temporal Consistency Methods

The motion pipeline was modified using:

- Gaussian prefiltering
- Joint-adaptive exponential moving average (EMA)
- Translation-velocity EMA

Prototype experiments reduced:

- Pose jitter by ~74.7%
- Translation jitter by ~60.0%

while revealing a trade-off between temporal smoothness and motion fidelity.

## Multi-Keyframe Reconstruction

A multi-keyframe source-management prototype was developed to investigate:

- Source-frame selection
- Keyframe switching
- Temporal consistency
- Reconstruction robustness across changing poses and viewpoints

## Render Restoration

A pretrained Stable Diffusion-based restoration prototype was explored to study improvements in:

- Single-frame visual quality
- Rendering artifacts
- Temporal appearance consistency

## Evaluation

A held-out-view evaluation pipeline was developed using:

- PSNR
- Foreground PSNR
- LPIPS
- Temporal delta error

The evaluation also investigates limitations of 2D foreground alignment compared with full camera K/R/T-aligned rendering.

## Pipeline

```text
Monocular Video
      ↓
Frame / Keyframe Selection
      ↓
LHM Reconstruction
      ↓
Motion Estimation
      ↓
Temporal Filtering
      ↓
Animatable 3D Gaussian Avatar
      ↓
Render Restoration
      ↓
Quantitative Evaluation
