// examples/xor_mlp.cpp
// Simple XOR Neural Network Example
// Demonstrates basic tensor operations

#include <simd/simd.hpp>
#include <iostream>
#include <random>

using namespace micrograd::simd;

int main() {
    std::cout << "=== XOR MLP Example ===\n\n";
    std::cout << simd_info() << "\n\n";
    
    // XOR truth table
    // Input      Output
    // [0, 0]  -> 0
    // [0, 1]  -> 1
    // [1, 0]  -> 1
    // [1, 1]  -> 0
    
    // Create input data (4 samples, 2 features)
    AlignedTensor<float> X({4, 2});
    X[0] = 0.0f; X[1] = 0.0f;  // [0, 0]
    X[2] = 0.0f; X[3] = 1.0f;  // [0, 1]
    X[4] = 1.0f; X[5] = 0.0f;  // [1, 0]
    X[6] = 1.0f; X[7] = 1.0f;  // [1, 1]
    
    // Target outputs
    AlignedTensor<float> Y({4, 1});
    Y[0] = 0.0f;  // XOR(0,0) = 0
    Y[1] = 1.0f;  // XOR(0,1) = 1
    Y[2] = 1.0f;  // XOR(1,0) = 1
    Y[3] = 0.0f;  // XOR(1,1) = 0
    
    std::cout << "XOR Problem:\n";
    std::cout << "  Input X:\n";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "    [" << X[i*2] << ", " << X[i*2+1] << "] -> " << Y[i] << "\n";
    }
    std::cout << "\n";
    
    // Initialize weights randomly
    // Hidden layer: 2 inputs -> 4 hidden neurons
    auto W1 = AlignedTensor<float>::randn({2, 4});
    auto b1 = AlignedTensor<float>::zeros({1, 4});
    
    // Output layer: 4 hidden -> 1 output
    auto W2 = AlignedTensor<float>::randn({4, 1});
    auto b2 = AlignedTensor<float>::zeros({1, 1});
    
    // Scale weights for better initialization
    for (size_t i = 0; i < W1.size(); ++i) W1[i] *= 0.5f;
    for (size_t i = 0; i < W2.size(); ++i) W2[i] *= 0.5f;
    
    std::cout << "Network Architecture: 2 -> 4 (ReLU) -> 1\n";
    std::cout << "Parameters: " << (2*4 + 4 + 4*1 + 1) << "\n\n";
    
    // Forward pass demonstration
    std::cout << "Forward pass with random weights:\n";
    
    // Hidden layer: Z1 = X @ W1 + b1, H = ReLU(Z1)
    auto Z1 = matmul(X, W1);
    // Add bias (simplified - broadcasting not implemented)
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            Z1[i * 4 + j] += b1[j];
        }
    }
    auto H = relu(Z1);
    
    // Output layer: Z2 = H @ W2 + b2, Y_pred = sigmoid(Z2)
    auto Z2 = matmul(H, W2);
    for (size_t i = 0; i < 4; ++i) {
        Z2[i] += b2[0];
    }
    
    // Sigmoid activation
    AlignedTensor<float> Y_pred({4, 1});
    sigmoid_f32(Y_pred.data(), Z2.data(), 4);
    
    std::cout << "  Predictions (before training):\n";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "    [" << X[i*2] << ", " << X[i*2+1] << "] -> " 
                  << std::fixed << std::setprecision(4) << Y_pred[i] 
                  << " (target: " << Y[i] << ")\n";
    }
    
    std::cout << "\n";
    std::cout << "Note: This example shows forward pass only.\n";
    std::cout << "Full training requires autograd (backward pass).\n";
    std::cout << "\n=== Example Complete ===\n";
    
    return 0;
}

