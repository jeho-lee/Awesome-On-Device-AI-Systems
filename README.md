# Awesome research works for on-device AI systems

A curated list of research works on efficient on-device AI systems, methods, and applications for mobile and edge devices.

Note: Some of the works are designed for inference acceleration on cloud/server infrastructure, which has much higher computational resources, but I also include them here if they can be potentially generalized to on-device inference use cases.

<!-- ACM ***MobiSys***, ACM ***MobiCom***, ACM ***Sensys***, ACM ***EuroSys***, ACM ***IPSN***, ACM ***ASPLOS***, USENIX ***NSDI***, USENIX ***ATC***, ***MLSys***, ... -->

#### Attention Acceleration
- [[MLSys 2025] MAS-Attention: Memory-Aware Stream Processing for Attention Acceleration on Resource-Constrained Edge Devices](https://arxiv.org/pdf/2411.17720)
- [[MLSys 2025] TurboAttention: Efficient attention approximation for High Throughputs LLMs](https://arxiv.org/pdf/2412.08585)
- [[ASPLOS 2023] FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks](https://dl.acm.org/doi/10.1145/3575693.3575747)
- [[NeurIPS 2022] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)

#### LLM Inference Acceleration on Mobile SoCs
- [[arXiv 2025] HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators](https://arxiv.org/pdf/2501.14794)
- [[ASPLOS 2025] Fast On-device LLM Inference with NPUs](https://arxiv.org/abs/2407.05858)
- [[arXiv 2024] PowerInfer-2: Fast Large Language Model Inference on a Smartphone](https://arxiv.org/abs/2406.06282)

#### Compiler-based ML Optimization
- [[MLSys 2025] TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives](https://arxiv.org/pdf/2503.20313)
- [[ASPLOS 2024] SmartMem: Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile](https://dl.acm.org/doi/pdf/10.1145/3620666.3651384)
- [[ASPLOS 2024] SoD<sup>2</sup>: Statically Optimizing Dynamic Deep Neural Network Execution](https://dl.acm.org/doi/pdf/10.1145/3617232.3624869)
- [[MICRO 2023] Improving Data Reuse in NPU On-chip Memory with Interleaved Gradient Order for DNN Training](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10411391)
- [[MICRO 2022] GCD<sup>2</sup>: A Globally Optimizing Compiler for Mapping DNNs to Mobile DSPs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9923837)
- [[PLDI 2021] DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083)

#### Hardware-aware Quantization
- [[arXiv 2025] HALO: Hardware-aware quantization with low critical-path-delay weights for LLM acceleration](https://arxiv.org/pdf/2502.19662)
- [[ISCA 2025] MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization](https://arxiv.org/pdf/2411.05282)
- [[MLSys 2024] AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/pdf/2306.00978)
- [[ISCA 2023] OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization](https://arxiv.org/abs/2304.07493)

#### Inference Acceleration using Heterogeneous Computing Processors (e.g., CPU, GPU, NPU, etc.)
- [[MobiSys 2025] ARIA: Optimizing Vision Foundation Model Inference on Heterogeneous Mobile Processors for Augmented Reality](https://arxiv.org/pdf/2501.14794)
- [[PPoPP 2024] Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous SoCs](https://dl.acm.org/doi/pdf/10.1145/3627535.3638502)
- [[MobiSys 2024] Pantheon: Preemptible Multi-DNN Inference on Mobile Edge GPUs](https://dl.acm.org/doi/pdf/10.1145/3643832.3661878)
- [[MobiCom 2024] Perceptual-Centric Image Super-Resolution using Heterogeneous Processors on Mobile Devices](https://dl.acm.org/doi/10.1145/3636534.3690698)
- [[Sensys 2023] Miriam: Exploiting Elastic Kernels for Real-time Multi-DNN Inference on Edge GPU](https://dl.acm.org/doi/10.1145/3625687.3625789)
- [[MobiSys 2023] NN-Stretch: Automatic Neural Network Branching for Parallel Inference on Heterogeneous Multi-Processors](https://dl.acm.org/doi/pdf/10.1145/3472381.3479910)
- [[ATC 2023] Decentralized Application-Level Adaptive Scheduling for Multi-Instance DNNs on Open Mobile Devices](https://www.usenix.org/system/files/atc23-sung.pdf)
- [[IPSN 2023] PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators](https://dl.acm.org/doi/pdf/10.1145/3583120.3587045)
- [[SenSys 2022] BlastNet: Exploiting Duo-Blocks for Cross-Processor Real-Time DNN Inference](https://dl.acm.org/doi/pdf/10.1145/3560905.3568520)
- [[MobiSys 2022] Band: Coordinated Multi-DNN Inference on Heterogeneous Mobile Processors](https://dl.acm.org/doi/pdf/10.1145/3498361.3538948)
- [[MobiSys 2022] CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices](https://dl.acm.org/doi/pdf/10.1145/3498361.3538932)

#### Adaptive Inference for Optimized Resource Utilization
- [[RTSS 2024] FLEX: Adaptive Task Batch Scheduling with Elastic Fusion in Multi-Modal Multi-View Machine Perception](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10844787)
- [[MobiCom 2024] Panopticus: Omnidirectional 3D Object Detection on Resource-constrained Edge Devices](https://arxiv.org/pdf/2410.01270)
- [[MobiSys 2023] OmniLive: Super-Resolution Enhanced 360° Video Live Streaming for Mobile Devices](https://dl.acm.org/doi/pdf/10.1145/3581791.3596851)
- [[MobiSys 2023] HarvNet: Resource-Optimized Operation of Multi-Exit Deep Neural Networks on Energy Harvesting Devices](https://dl.acm.org/doi/abs/10.1145/3581791.3596845)
- [[MobiCom 2022] NeuLens: Spatial-based Dynamic Acceleration of Convolutional Neural Networks on Edge](https://dl.acm.org/doi/pdf/10.1145/3495243.3560528)
- [[MobiCom 2021] Flexible high-resolution object detection on edge devices with tunable latency](https://dl.acm.org/doi/abs/10.1145/3447993.3483274)

#### On-device Training, Model Adaptation
- [[ASPLOS 2025] Nazar: Monitoring and Adapting ML Models on Mobile Devices](https://dl.acm.org/doi/pdf/10.1145/3669940.3707246)
- [[SenSys 2024] AdaShadow: Responsive Test-time Model Adaptation in Non-stationary Mobile Environments](https://arxiv.org/pdf/2410.08256)
- [[SenSys 2023] EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge](https://dl.acm.org/doi/10.1145/3625687.3625793)
- [[MobiCom 2023] Cost-effective On-device Continual Learning over Memory Hierarchy with Miro](https://dl.acm.org/doi/pdf/10.1145/3570361.3613297)
- [[MobiCom 2023] AdaptiveNet: Post-deployment Neural Architecture Adaptation for Diverse Edge Environments](https://dl.acm.org/doi/pdf/10.1145/3570361.3592529)
- [[MobiSys 2023] ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection](https://dl.acm.org/doi/pdf/10.1145/3581791.3596852)
- [[SenSys 2023] On-NAS: On-Device Neural Architecture Search on Memory-Constrained Intelligent Embedded Systems](https://dl.acm.org/doi/10.1145/3625687.3625814)
- [[MobiCom 2022] Mandheling: mixed-precision on-device DNN training with DSP offloading](https://dl.acm.org/doi/abs/10.1145/3495243.3560545)
- [[MobiSys 2022] Memory-efficient DNN training on mobile devices](https://dl.acm.org/doi/abs/10.1145/3498361.3539765)

#### Profilers
- [[MobiCom 2024] MELTing point: Mobile Evaluation of Language Transformers](https://arxiv.org/abs/2403.12844) [[code]](https://github.com/brave-experiments/MELT-public)
- [[SenSys 2023] nnPerf: Demystifying DNN Runtime Inference Latency on Mobile Platforms](https://dl.acm.org/doi/10.1145/3625687.3625797)
- [[MobiSys 2021] nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices](https://dl.acm.org/doi/10.1145/3458864.3467882)

## By Conference (2025~)

<details>
<summary>MLSys 2025</summary>

- [[MLSys 2025] MAS-Attention: Memory-Aware Stream Processing for Attention Acceleration on Resource-Constrained Edge Devices](https://arxiv.org/pdf/2411.17720)
- [[MLSys 2025] Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking](https://arxiv.org/pdf/2412.01380)
- [[MLSys 2025] TurboAttention: Efficient attention approximation for High Throughputs LLMs](https://arxiv.org/pdf/2412.08585)
- [[MLSys 2025] SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention](https://arxiv.org/pdf/2406.15486)
- [[MLSys 2025] LeanAttention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers](https://arxiv.org/pdf/2405.10480)

</details>

<details>
<summary>ASPLOS 2025</summary>

- [[Fast On-device LLM Inference with NPUs]](https://arxiv.org/abs/2407.05858)
- Energy-aware Scheduling and Input Buffer Overflow Prevention for Energy-harvesting Systems
- Generalizing Reuse Patterns for Efficient DNN on Microcontrollers
- Nazar: Monitoring and Adapting ML Models on Mobile Devices

</details>

<details>
<summary>EuroSys 2025</summary>

- Flex: Fast, Accurate DNN Inference on Low-Cost Edges Using Heterogeneous Accelerator Execution
- T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge

</details>

<details>
<summary>SOSP 2025</summary>

</details>

<details>
<summary>MobiSys 2025</summary>

- ARIA: Optimizing Vision Foundation Model Inference on Heterogeneous Mobile Processors for Augmented Reality

</details>

<details>
<summary>MobiCom 2025</summary>

</details>

<details>
<summary>Preprint 2025</summary>

- HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators

</details>