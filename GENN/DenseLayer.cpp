#include "pch.h"
#include "DenseLayer.h"

namespace cpu {


cpu::Vector softmax(const cpu::Vector& v) {
    cpu::Vector r(v.s);
    float sum = 0.0f;
    for (int i = 0; i < v.s; i++) {
        r[i] = expf(v[i]);
        sum += r[i];
    }
    for (int i = 0; i < v.s; i++) {
        r[i] = r[i] / sum;
    }
    return r;
}


float crossEntropySoftmax(const Vector& output, int label) {
    float denom = 0.0f;
    Vector sm(output.s);
    for (int i = 0; i < output.s; i++) {
        sm[i] = expf(output[i]);
        denom += sm[i];
    }
    return -logf(sm[label] / denom);
}


DenseLayer::DenseLayer(int in, int out, Activation a, bool io) : 
    W(out, in), b(out), 
    gradW(out, in), gradb(out), 
    inSize(in), outSize(out), 
    input(in), output(out),
    dInput(in), dOutput(out),
    activation(a), isOutput(io) {
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<float> d{ 0,1 };

    for (int i = 0; i < W.h; i++) {
        for (int j = 0; j < W.w; j++) {
            W(i, j) = static_cast<float>(d(gen));;
        }
        b[i] = static_cast<float>(d(gen));
    }
    zeroGrad();
}


float gradActivation(float x, Activation a) {
    if (a == Activation::linear) {
        return 1.0f;
    }
    else {
        return x * (1 - x);
    }
}


Vector gradActivation(const Vector& v, Activation a) {
    Vector r(v.s);
    for (int i = 0; i < r.s; i++) {
        r[i] = gradActivation(v[i], a);
    }
    return r;
}


void DenseLayer::forward() {
    output = W.mul(input) + b;
    for (int i = 0; i < output.s; i++) {
        if (activation == Activation::sigmoid) {
            output[i] = 1.0f / (1.0f + expf(-output[i]));
        }
    }
}


void DenseLayer::backward() {
    Vector gradAct = gradActivation(output, activation);
    for (int i = 0; i < gradW.h; i++) {
        for (int j = 0; j < gradW.w; j++) {
            gradW(i, j) += dOutput[i] * input[j] * gradAct[i];
        }
        gradb[i] += gradAct[i] * dOutput[i];
    }

    Vector dInput = Vector(input.s);
    for (int j = 0; j < dInput.s; j++) {
        dInput[j] = 0.0f;
        for (int i = 0; i < dOutput.s; i++) {
            dInput[j] += W(i, j) * dOutput[i] * gradAct[i];
        }
    }
}


void DenseLayer::step(float eps) {
    for (int i = 0; i < W.h; i++) {
        for (int j = 0; j < W.w; j++) {
            W(i, j) -= eps * gradW(i, j);
        }
        b[i] -= eps * gradb[i];
    }
}


void DenseLayer::zeroGrad() {
    for (int i = 0; i < gradW.h; i++) {
        for (int j = 0; j < gradW.w; j++) {
            gradW(i, j) = 0.0f;
        }
        gradb[i] = 0.0f;
    }
}


void DenseLayer::initBackProp(int label) {
    if (!isOutput)
        return;
    dOutput = softmax(output);
    for (int i = 0; i < dOutput.s; i++) {
        dOutput[i] -= i == label ? 1.0f : 0.0f;
    }
}


float DenseLayer::loss(int label) {
    return crossEntropySoftmax(output, label);
}


int DenseLayer::argmax() {
    return output.argmax();
}

}