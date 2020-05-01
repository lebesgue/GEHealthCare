#pragma once

#include "cuda.h"


namespace cuda {


typedef struct {
	float * data = nullptr;
	int w = 0;
	int h = 0;
} Matrix;


typedef struct {
	float * data = nullptr;
	int s = 0;
} Vector;


struct DenseLayer {
    bool isOutput;

    int in;
    int out;
    int activation;

    Vector input;
    Vector dInput;
    Vector output;
    Vector dOutput;

    Matrix W;
    Vector b;
    Matrix gradW;
    Vector gradb;

    inline DenseLayer(int in, int out, int act, bool isOutput) : 
        in(in), out(out), activation(act), isOutput(isOutput) {
        initLayer();
    };
    void forward();
    void backward();
    void zeroGrad();
    void step(float learningRate);
    void initBackProp(int label);
    float loss(int label);
    int argmax();
    void initLayer();
    void destroyLayer();
};


void toGpu(float** dst, float** src, int s);
void fromGpu(float** dst, float** src, int s);



}
