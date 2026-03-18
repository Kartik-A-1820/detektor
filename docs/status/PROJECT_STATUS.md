# Project Status

## Current Maturity

Detektor is currently at an **early production-ready / public release candidate** stage for local experimentation and single-machine deployment.

What is already in place:

- model architecture for detection and instance segmentation
- training, validation, and inference entrypoints
- production hardening utilities
- ONNX export path
- local FastAPI serving and CLI inference
- lightweight smoke tests and integration checks
- dataset YAML auto-configuration for YOLO/Roboflow-style exports

## What This Project Is Good For Today

- rapid experimentation on a local workstation
- single-GPU inference and validation workflows
- ONNX export preparation
- lightweight local serving
- developer-friendly ML project bootstrapping

## Known Limitations

- no TensorRT runtime yet
- no distributed or multi-GPU training
- limited dataset-loader variety
- lightweight metrics and validation only
- no pretrained model zoo packaged yet

## Roadmap

Planned future work includes:

- TensorRT runtime
- ONNX Runtime benchmarking
- distributed training
- multi-GPU training
- model zoo
- pretrained weights
- richer dataset loader support
- deployment packaging improvements


