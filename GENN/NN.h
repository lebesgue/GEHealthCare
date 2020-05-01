#pragma once

#include <random>
#include "Matrix.h"
#include "Vector.h"


namespace nn {


class Network {

    enum NetworkStatus { zero, training, paused };

private:
    std::vector<int> currentPredictions;
    std::vector<float> loss;

public:
    std::mutex nnMutex;

    std::vector<pf::DenseLayer> layers;

    NetworkStatus status = NetworkStatus::zero;
    float learningRate = 0.01f;

    int epoch = 0;

    std::vector<pf::Vector> images;
    std::vector<int> labels;

    std::vector<int> testOrder;
    std::vector<pf::Vector> testImages;
    std::vector<int> testLabels;

    cpu::Matrix confusionMatrix;

    Network();

    void initLayers();

    void forward(int p);
    void backward(int label);
    void step();
    void zeroGrad();

    void setPosition(int n);
    int getPosition();

    float meanLoss();
    void setLoss(float loss);
    std::vector<float> getLoss();

    void startTraining();
    void pauseTraining();
    void resumeTraining();
    void stopTraining();
    bool isTraining();
    bool isPaused();

    pf::Vector& getImage(int p);
    int getLabel(int p);

    std::vector<int> getPredictions(int n);

    void setTrainData(const std::vector<cpu::Matrix>& images, const std::vector<int>& labels);
    void setTestData(const std::vector<cpu::Matrix>& images, const std::vector<int>& labels);

    float trainPrecision();
    float testPrecision();

    void train();
    int predict(int p);
    float test(int n);
};

}


