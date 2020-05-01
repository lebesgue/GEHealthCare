#include "pch.h"
#include "NN.h"
#include "math.h"

namespace nn {


Network::Network() {
    initLayers();
}


void Network::initLayers() {


#ifdef CUDA
    cuda::DenseLayer layer1(784, 64, 0, false);
    cuda::DenseLayer layer2(64, 64, 0, false);
    cuda::DenseLayer layer3(64, 10, 2, true);

    layers = { layer1, layer2, layer3 };

    auto initRandom = [](float* x, int n) {
        std::random_device rd{};
        std::mt19937 gen{ rd() };
        std::normal_distribution<> d{ 0,1 };

        for (int i = 0; i < n; i++) {
            x[i] = static_cast<float>(d(gen));
        }
    };

    for (int i = 0; i < (int)layers.size(); i++) {
        cuda::DenseLayer& cl = layers[i];

        float* x = (float *)malloc(cl.W.h * cl.W.w * sizeof(float));
        initRandom(x, cl.W.h * cl.W.w);
        cuda::toGpu(&cl.W.data, &x, cl.W.h * cl.W.w);

        x = (float *)realloc(x, cl.b.s * sizeof(float));
        initRandom(x, cl.b.s);
        cuda::toGpu(&cl.b.data, &x, cl.b.s);

    }
#else
    layers = {
        cpu::DenseLayer(784, 64, cpu::Activation::sigmoid),
        cpu::DenseLayer(64, 64, cpu::Activation::sigmoid),
        cpu::DenseLayer(64, 10, cpu::Activation::linear, true)
    };
#endif // CUDA


}


void Network::forward(int p) {
    layers[0].input = getImage(p);
    for (int i = 0; i < layers.size(); ++i) {
        layers[i].forward();
        if (i < layers.size() - 1) {
            layers[i + 1].input = layers[i].output;
        }
    }
}


void Network::backward(int label) {
    layers.back().initBackProp(label);
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        layers[i].backward();
        if (i > 0) {
            layers[i - 1].dOutput = layers[i].dInput;
        }
    }
}

void Network::step() {
    for (auto& layer : layers) {
        layer.step(learningRate);
    }
}

void Network::zeroGrad() {
    for (auto& layer : layers) {
        layer.zeroGrad();
    }
}


void Network::setPosition(int n) {
    std::lock_guard<std::mutex> guard(nnMutex);
    currentPredictions.emplace_back(n);
}


int Network::getPosition() {
    return static_cast<int>(currentPredictions.size());
}


float Network::meanLoss() {
    std::lock_guard<std::mutex> guard(nnMutex);
    auto b = loss.size() >= (int)1000 ? loss.end() - 1000 : loss.begin();
    return std::accumulate(b, loss.end(), 0.0f) / std::min(1000.0f, static_cast<float>(loss.size()));
}


void Network::setLoss(float loss) {
    std::lock_guard<std::mutex> guard(nnMutex);
    this->loss.emplace_back(loss);
}

std::vector<float> Network::getLoss() {
    std::lock_guard<std::mutex> guard(nnMutex);
    return loss;
}


void Network::startTraining() {
    status = NetworkStatus::training;
    currentPredictions.clear();
    loss.clear();
    initLayers();
    zeroGrad();
    train();
}


void Network::stopTraining() {
    std::lock_guard<std::mutex> guard(nnMutex);
    status = Network::zero;
}


void Network::pauseTraining() {
    std::lock_guard<std::mutex> guard(nnMutex);
    status = NetworkStatus::paused;
}


void Network::resumeTraining() {
    status = NetworkStatus::training;
    train();
}


bool Network::isTraining() {
    std::lock_guard<std::mutex> guard(nnMutex);
    return status == NetworkStatus::training;
}


bool Network::isPaused() {
    std::lock_guard<std::mutex> guard(nnMutex);
    return status == NetworkStatus::paused;
}

pf::Vector& Network::getImage(int p) {
    if (status == NetworkStatus::training) {
        return images[p];
    }
    else {
        return testImages[testOrder[p]];
    }
}

int Network::getLabel(int p) {
    if (status == NetworkStatus::training) {
        return labels[p];
    }
    else {
        return testLabels[testOrder[p]];
    }
}

std::vector<int> Network::getPredictions(int n) {
    std::lock_guard<std::mutex> guard(nnMutex);
    int s = std::min((int)currentPredictions.size(), n);
    return std::vector<int>(currentPredictions.end() - n, currentPredictions.end());
}


cpu::Vector prepareInput(const cpu::Matrix& image) {
    cpu::Vector input(image.w * image.h);
    memcpy(input.data, image.data, input.s * sizeof(float));
    for (int i = 0; i < input.s; i++) {
        input[i] = (input[i] / 255.0f - 0.1307f) / 0.3081f;
    }
    return input;
}


void Network::setTrainData(const std::vector<cpu::Matrix>& images, const std::vector<int>& labels) {
#ifdef CUDA
    this->images = std::vector<cuda::Vector>(images.size());
    for (int i = 0; i < (int)images.size(); i++) {
        cpu::Vector pInput = prepareInput(images[i]);
        cuda::toGpu(&this->images[i].data, &pInput.data, images[i].w * images[i].h);
    }
#else
    this->images = std::vector<cpu::Vector>(images.size());
    for (int i = 0; i < (int)images.size(); i++) {
        this->images[i] = prepareInput(images[i]);
}
#endif // CUDA
    this->labels = labels;
}

void Network::setTestData(const std::vector<cpu::Matrix>& images, const std::vector<int>& labels) {
#ifdef CUDA
    this->testImages = std::vector<cuda::Vector>(images.size());
    for (int i = 0; i < (int)images.size(); i++) {
        cpu::Vector pInput = prepareInput(images[i]);
        cuda::toGpu(&this->testImages[i].data, &pInput.data, images[i].w * images[i].h);
    }
#else
    this->testImages = std::vector<cpu::Vector>(images.size());
    for (int i = 0; i < (int)images.size(); i++) {
        this->testImages[i] = prepareInput(images[i]);
    }
#endif // CUDA
    this->testLabels = labels;
}


float Network::trainPrecision() {
    int p = getPosition();
    int size = static_cast<int>(images.size());
    int start = (p / size) * size;
    int acc = 0;
    for (int i = 0; i < p % size; i++) {
        acc += currentPredictions[start + i] == labels[i] ? 1 : 0;
    }
    return static_cast<float>(acc) / (p % size);
}

float Network::testPrecision() {
    float x = 0;
    float sum = 0;
    for (int i = 0; i < confusionMatrix.h; i++) {
        for (int j = 0; j < confusionMatrix.w; j++) {
            sum += confusionMatrix(i, j);
        }
        x += confusionMatrix(i, i);
    }
    return x / (sum + 1e-6f);
}


void Network::train() {
    int n = getPosition();
    while (isTraining()) {
        if (n % (int)images.size() == 0 && n > 0) {
            ++epoch;
        }

        int p = n % (int)images.size();
        forward(p);
        backward(getLabel(p));
        setPosition(layers.back().argmax());
        setLoss(layers.back().loss(getLabel(p)));

        if (n % 50 == 0 && n > 0) {
            step();
            zeroGrad();
        }
        if (n > 0 && n % 10000 == 0) {
            test(10000);
        }

        ++n;
    }
}

int Network::predict(int p) {
    forward(p);
    return layers.back().argmax();
}


float Network::test(int n) {
    int correct = 0;
    std::random_shuffle(testOrder.begin(), testOrder.end());
    confusionMatrix = cpu::Matrix(10, 10);
    for (int i = 0; i < n; i++) {
        int pred = predict(i);
        if (pred == getLabel(i))
            ++correct;
        confusionMatrix(pred, getLabel(i)) += 1.0f;
    }
    return static_cast<float>(correct) / n;
}

}

