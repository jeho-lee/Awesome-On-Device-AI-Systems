# Awesome research works for On-device AI Systems

A curated list of research works on efficient on-device AI systems, methods, and applications for mobile and edge devices.

<!-- ### Efficient On-device AI Systems for Mobile and Edge Devices -->

<!-- ACM ***MobiSys***, ACM ***MobiCom***, ACM ***Sensys***, ACM ***EuroSys***, ACM ***IPSN***, ACM ***ASPLOS***, USENIX ***NSDI***, USENIX ***ATC***, ***MLSys***, ... -->

#### Efficient Inference using Heterogeneous Processors (e.g., CPU, GPU, NPU, etc.)
- [MobiCom 2024] Perceptual-Centric Image Super-Resolution using Heterogeneous Processors on Mobile Devices [[paper]](https://dl.acm.org/doi/10.1145/3636534.3690698)
- [Sensys 2023] Miriam: Exploiting Elastic Kernels for Real-time Multi-DNN Inference on Edge GPU [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625789)
- [MobiSys 2023] NN-Stretch: Automatic Neural Network Branching for Parallel Inference on Heterogeneous Multi-Processors [[paper]](https://dl.acm.org/doi/pdf/10.1145/3472381.3479910)
- [ATC 2023] Decentralized Application-Level Adaptive Scheduling for Multi-Instance DNNs on Open Mobile Devices [[paper]](https://www.usenix.org/system/files/atc23-sung.pdf)
- [IPSN 2023] PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators [[paper]](https://dl.acm.org/doi/pdf/10.1145/3583120.3587045)
- [SenSys 2022] BlastNet: Exploiting Duo-Blocks for Cross-Processor Real-Time DNN Inference [[paper]](https://dl.acm.org/doi/pdf/10.1145/3560905.3568520)
- [MobiSys 2022] Band: Coordinated Multi-DNN Inference on Heterogeneous Mobile Processors [[paper]](https://dl.acm.org/doi/pdf/10.1145/3498361.3538948)
- [MobiSys 2022] CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices [[paper]](https://dl.acm.org/doi/pdf/10.1145/3498361.3538932)

#### Adaptive Inference for Optimized Resource Utilization
- [MobiCom 2024] Panopticus: Omnidirectional 3D Object Detection on Resource-constrained Edge Devices [[paper]](https://arxiv.org/pdf/2410.01270)
- [MobiSys 2023] OmniLive: Super-Resolution Enhanced 360° Video Live Streaming for Mobile Devices [[paper]](https://dl.acm.org/doi/pdf/10.1145/3581791.3596851)
- [MobiSys 2023] HarvNet: Resource-Optimized Operation of Multi-Exit Deep Neural Networks on Energy Harvesting Devices [[paper]](https://dl.acm.org/doi/abs/10.1145/3581791.3596845)
- [MobiCom 2022] NeuLens: Spatial-based Dynamic Acceleration of Convolutional Neural Networks on Edge [[paper]](https://dl.acm.org/doi/pdf/10.1145/3495243.3560528)
- [MobiCom 2021] Flexible high-resolution object detection on edge devices with tunable latency [[paper]](https://dl.acm.org/doi/abs/10.1145/3447993.3483274)

#### On-device LLM
- [arXiv 2024] PowerInfer-2: Fast Large Language Model Inference on a Smartphone [[paper]](https://arxiv.org/abs/2406.06282)
- [ASPLOS 2025] Empowering 1000 tokens/second on-deviceLLM prefilling with mllm-NPU [[paper]](https://arxiv.org/pdf/2407.05858) [[code]](https://github.com/UbiquitousLearning/mllm)
- [MobiCom 2024] MELTing point: Mobile Evaluation of Language Transformers [[paper]](https://arxiv.org/abs/2403.12844) [[code]](https://github.com/brave-experiments/MELT-public)
- [MobiCom 2024] Mobile Foundation Model as Firmware [[paper]](https://arxiv.org/pdf/2308.14363) [[code]](https://github.com/UbiquitousLearning/MobileFM)

#### On-device Training, Model Adaptation
- [SenSys 2024] AdaShadow: Responsive Test-time Model Adaptation in Non-stationary Mobile Environments [[paper]](https://arxiv.org/pdf/2410.08256)
- [SenSys 2023] EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625793)
- [MobiCom 2023] Cost-effective On-device Continual Learning over Memory Hierarchy with Miro [[paper]](https://dl.acm.org/doi/pdf/10.1145/3570361.3613297)
- [MobiCom 2023] AdaptiveNet: Post-deployment Neural Architecture Adaptation for Diverse Edge Environments [[paper]](https://dl.acm.org/doi/pdf/10.1145/3570361.3592529)
- [MobiSys 2023] ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection [[paper]](https://dl.acm.org/doi/pdf/10.1145/3581791.3596852)
- [SenSys 2023] On-NAS: On-Device Neural Architecture Search on Memory-Constrained Intelligent Embedded Systems [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625814)
- [MobiCom 2022] Mandheling: mixed-precision on-device DNN training with DSP offloading [[paper]](https://dl.acm.org/doi/abs/10.1145/3495243.3560545)
- [MobiSys 2022] Memory-efficient DNN training on mobile devices [[paper]](https://dl.acm.org/doi/abs/10.1145/3498361.3539765)

#### Edge-Cloud Collaborative Inference
- [MobiSys 2024] ARISE: High-Capacity AR Offloading Inference Serving via Proactive Scheduling [[paper]](https://dl.acm.org/doi/10.1145/3643832.3661894)
- [MobiSys 2024] CoActo: CoActive Neural Network Inference Offloading with Fine-grained and Concurrent Execution [[paper]](https://dl.acm.org/doi/10.1145/3643832.3661885)
- [MobiCom 2023] AccuMO: Accuracy-Centric Multitask Offloading in Edge-Assisted Mobile Augmented Reality [[paper]](https://dl.acm.org/doi/pdf/10.1145/3570361.3592531)
- [MobiSys 2022] DeepMix: Mobility-aware, Lightweight, and Hybrid 3D Object Detection for Headsets [[paper]](https://doi.org/10.1145/3498361.3538945)

#### Profilers
- [SenSys 2023] nnPerf: Demystifying DNN Runtime Inference Latency on Mobile Platforms [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625797)
- [MobiSys 2021] nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices [[paper]](https://dl.acm.org/doi/10.1145/3458864.3467882)


<!-- ### Efficient AI methods

#### Elastic Neural Networks
- [ICML 2024] FLEXTRON: Many-in-One Flexible Large Language Model [[paper]](https://arxiv.org/pdf/2406.10260)
- [CVPR 2023 *Highlight*] Stitchable Neural Networks [[paper]](https://arxiv.org/abs/2302.06586) [[code]](https://github.com/ziplab/SN-Net)  

#### Efficient LLM/VLMs
- [ICML 2024] MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases [[paper]](https://arxiv.org/pdf/2402.14905) [[code]](https://github.com/facebookresearch/MobileLLM)
- [arXiv 2023] MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices [[paper]](https://arxiv.org/abs/2312.16886) [[code]](https://github.com/Meituan-AutoML/MobileVLM?tab=readme-ov-file)

#### Quantization
- [MLSys 2024 *Best Paper*] AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration [[paper]](https://arxiv.org/pdf/2306.00978) [[code]](https://github.com/mit-han-lab/llm-awq)

#### Pruning and Compression
- [CVPR 2023] DepGraph: Towards Any Structural Pruning [[paper]](https://arxiv.org/abs/2301.12900) [[code]](https://github.com/VainF/Torch-Pruning)
- [ICML 2023] Efficient Latency-Aware CNN Depth Compression via Two-Stage Dynamic Programming [[paper]](https://arxiv.org/abs/2301.12187) [[code]](https://github.com/snu-mllab/Efficient-CNN-Depth-Compression)
- [NeurIPS 2022] Structural Pruning via Latency-Saliency Knapsack [[paper]](https://arxiv.org/abs/2210.06659) [[code]](https://github.com/NVlabs/HALP)

#### Efficient Vision Transformer (ViT)
- [ICLR 2023 top 5%] Token Merging: Your ViT but Faster [[paper]](https://arxiv.org/abs/2210.09461) [[code]](https://github.com/facebookresearch/ToMe)
- [ICCV 2023] Rethinking Vision Transformers for MobileNet Size and Speed [[paper]](https://arxiv.org/abs/2212.08059) [[code]](https://github.com/snap-research/EfficientFormer)
- [ICCV 2023] EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction [[paper]](https://arxiv.org/abs/2205.14756) [[code]](https://github.com/mit-han-lab/efficientvit)
- [CVPR 2023] SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer [[paper]](https://arxiv.org/abs/2303.17605) [[code]](https://github.com/mit-han-lab/sparsevit)
- [CVPR 2022 *Oral*] PoolFormer: MetaFormer Is Actually What You Need for Vision [[paper]](https://arxiv.org/abs/2111.11418) [[code]](https://github.com/sail-sg/poolformer) -->
