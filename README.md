# Awesome research works for on-device AI systems

A curated list of research works on efficient on-device AI systems, methods, and applications for mobile and edge devices.

<!-- ACM ***MobiSys***, ACM ***MobiCom***, ACM ***Sensys***, ACM ***EuroSys***, ACM ***IPSN***, ACM ***ASPLOS***, USENIX ***NSDI***, USENIX ***ATC***, ***MLSys***, ... -->

## By Topic

#### Hardware-aware Attention Acceleration Methods
- [MLSys 2025] MAS-Attention: Memory-Aware Stream Processing for Attention Acceleration on Resource-Constrained Edge Devices [[paper]](https://arxiv.org/pdf/2411.17720)
- [MLSys 2025] Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking [[paper]](https://arxiv.org/pdf/2412.01380)
- [NeurIPS 2022] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness [[paper]](https://arxiv.org/pdf/2205.14135)

#### On-device LLM Inference Systems
- [arXiv 2025] HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators [[paper]](https://arxiv.org/pdf/2501.14794)
- [ASPLOS 2025] Fast On-device LLM Inference with NPUs [[paper]](https://arxiv.org/abs/2407.05858) [[code]](https://github.com/UbiquitousLearning/mllm)
- [arXiv 2024] PowerInfer-2: Fast Large Language Model Inference on a Smartphone [[paper]](https://arxiv.org/abs/2406.06282)
- [MobiCom 2024] MELTing point: Mobile Evaluation of Language Transformers [[paper]](https://arxiv.org/abs/2403.12844) [[code]](https://github.com/brave-experiments/MELT-public)
- [MobiCom 2024] Mobile Foundation Model as Firmware [[paper]](https://arxiv.org/pdf/2308.14363) [[code]](https://github.com/UbiquitousLearning/MobileFM)

#### Inference Acceleration Systems with Heterogeneous Computing Processors (e.g., CPU, GPU, NPU, etc.)
- [MobiSys 2024] Pantheon: Preemptible Multi-DNN Inference on Mobile Edge GPUs [[paper]](https://dl.acm.org/doi/pdf/10.1145/3643832.3661878)
- [MobiCom 2024] Perceptual-Centric Image Super-Resolution using Heterogeneous Processors on Mobile Devices [[paper]](https://dl.acm.org/doi/10.1145/3636534.3690698)
- [Sensys 2023] Miriam: Exploiting Elastic Kernels for Real-time Multi-DNN Inference on Edge GPU [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625789)
- [MobiSys 2023] NN-Stretch: Automatic Neural Network Branching for Parallel Inference on Heterogeneous Multi-Processors [[paper]](https://dl.acm.org/doi/pdf/10.1145/3472381.3479910)
- [ATC 2023] Decentralized Application-Level Adaptive Scheduling for Multi-Instance DNNs on Open Mobile Devices [[paper]](https://www.usenix.org/system/files/atc23-sung.pdf)
- [IPSN 2023] PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators [[paper]](https://dl.acm.org/doi/pdf/10.1145/3583120.3587045)
- [SenSys 2022] BlastNet: Exploiting Duo-Blocks for Cross-Processor Real-Time DNN Inference [[paper]](https://dl.acm.org/doi/pdf/10.1145/3560905.3568520)
- [MobiSys 2022] Band: Coordinated Multi-DNN Inference on Heterogeneous Mobile Processors [[paper]](https://dl.acm.org/doi/pdf/10.1145/3498361.3538948)
- [MobiSys 2022] CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices [[paper]](https://dl.acm.org/doi/pdf/10.1145/3498361.3538932)

#### Adaptive Inference Systems for Optimized Resource Utilization
- [RTSS 2024] FLEX: Adaptive Task Batch Scheduling with Elastic Fusion in Multi-Modal Multi-View Machine Perception [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10844787)
- [MobiCom 2024] Panopticus: Omnidirectional 3D Object Detection on Resource-constrained Edge Devices [[paper]](https://arxiv.org/pdf/2410.01270)
- [MobiSys 2023] OmniLive: Super-Resolution Enhanced 360° Video Live Streaming for Mobile Devices [[paper]](https://dl.acm.org/doi/pdf/10.1145/3581791.3596851)
- [MobiSys 2023] HarvNet: Resource-Optimized Operation of Multi-Exit Deep Neural Networks on Energy Harvesting Devices [[paper]](https://dl.acm.org/doi/abs/10.1145/3581791.3596845)
- [MobiCom 2022] NeuLens: Spatial-based Dynamic Acceleration of Convolutional Neural Networks on Edge [[paper]](https://dl.acm.org/doi/pdf/10.1145/3495243.3560528)
- [MobiCom 2021] Flexible high-resolution object detection on edge devices with tunable latency [[paper]](https://dl.acm.org/doi/abs/10.1145/3447993.3483274)

#### On-device Training, Model Adaptation Systems
- [SenSys 2024] AdaShadow: Responsive Test-time Model Adaptation in Non-stationary Mobile Environments [[paper]](https://arxiv.org/pdf/2410.08256)
- [SenSys 2023] EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625793)
- [MobiCom 2023] Cost-effective On-device Continual Learning over Memory Hierarchy with Miro [[paper]](https://dl.acm.org/doi/pdf/10.1145/3570361.3613297)
- [MobiCom 2023] AdaptiveNet: Post-deployment Neural Architecture Adaptation for Diverse Edge Environments [[paper]](https://dl.acm.org/doi/pdf/10.1145/3570361.3592529)
- [MobiSys 2023] ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection [[paper]](https://dl.acm.org/doi/pdf/10.1145/3581791.3596852)
- [SenSys 2023] On-NAS: On-Device Neural Architecture Search on Memory-Constrained Intelligent Embedded Systems [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625814)
- [MobiCom 2022] Mandheling: mixed-precision on-device DNN training with DSP offloading [[paper]](https://dl.acm.org/doi/abs/10.1145/3495243.3560545)
- [MobiSys 2022] Memory-efficient DNN training on mobile devices [[paper]](https://dl.acm.org/doi/abs/10.1145/3498361.3539765)

#### Profilers
- [SenSys 2023] nnPerf: Demystifying DNN Runtime Inference Latency on Mobile Platforms [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625797)
- [MobiSys 2021] nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices [[paper]](https://dl.acm.org/doi/10.1145/3458864.3467882)

## By Conference (2025~)

#### MLSys 2025
- MAS-Attention: Memory-Aware Stream Processing for Attention Acceleration on Resource-Constrained Edge Devices [[paper]](https://arxiv.org/pdf/2411.17720)
- Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking [[paper]](https://arxiv.org/pdf/2412.01380)
- TurboAttention: Efficient attention approximation for High Throughputs LLMs [[paper]](https://arxiv.org/pdf/2412.08585)
- SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention [[paper]](https://arxiv.org/pdf/2406.15486)
- LeanAttention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers [[paper]](https://arxiv.org/pdf/2405.10480)

#### ASPLOS 2025
- Fast On-device LLM Inference with NPUs [[paper]](https://arxiv.org/abs/2407.05858) [[code]](https://github.com/UbiquitousLearning/mllm)

#### EuroSys 2025
- Flex: Fast, Accurate DNN Inference on Low-Cost Edges Using Heterogeneous Accelerator Execution [[paper]]()
- T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge [[paper]]()

#### SOSP 2025

#### MobiSys 2025

#### MobiCom 2025

#### Preprint
- HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators [[paper]](https://arxiv.org/pdf/2501.14794)