#pragma once

#include "Vector.h"
#include "Matrix.h"


namespace cpu {


enum Activation { sigmoid, linear };


class DenseLayer {
public:
    bool isOutput = false;

    int inSize;
    int outSize;
    Activation activation;

    Vector input;
    Vector dInput;
    Vector output;
    Vector dOutput;

    Matrix W;
    Vector b;
    Matrix gradW;
    Vector gradb;

    DenseLayer(int in, int out, Activation a, bool io = false);

    void forward();
    void backward();
    void step(float learningRate);
    void zeroGrad();
    void initBackProp(int label);
    float loss(int label);
    int argmax();
};


}