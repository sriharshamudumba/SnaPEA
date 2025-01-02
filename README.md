# SnaPEA
SnaPEA is an innovative framework designed to optimize neural network inference by dynamically skipping redundant activations, resulting in faster computation and reduced resource usage—all while maintaining or improving model accuracy.

Key Features
Accelerated Inference:
ResNet: Achieved a 74% speedup (runtime reduced from 20.12s to 5.19s) with an improved accuracy of 92.53%.
GoogLeNet: Delivered a 67% speedup (runtime reduced from 17.66s to 5.83s) with 88.51% accuracy.
VGG-Net: Improved inference time by 68% (runtime reduced from 22.46s to 7.10s) with an accuracy of 87.32%.
Seamless Integration: Built on top of popular AI frameworks like TensorFlow and PyTorch, making it easy to integrate into existing workflows.
Resource Efficiency: Reduces computational overhead, paving the way for scalable and energy-efficient AI systems.
How It Works
SnaPEA leverages the sparsity in neural activations during inference. By dynamically identifying and pruning non-contributing activations, it minimizes unnecessary computations, achieving significant performance gains.

Why Use SnaPEA?
Speed: Faster inference for real-time applications.
Efficiency: Optimized resource utilization for edge devices.
Scalability: Supports a wide range of neural network architectures.
