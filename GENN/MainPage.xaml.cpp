//
// MainPage.xaml.cpp
// Implementation of the MainPage class.
//

#include "pch.h"
#include "MainPage.xaml.h"
#include "plot.hpp"

using namespace GENN;

using namespace Platform;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::UI::Core;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Controls::Primitives;
using namespace Windows::UI::Xaml::Data;
using namespace Windows::UI::Xaml::Input;
using namespace Windows::UI::Xaml::Media;
using namespace Windows::UI::Xaml::Navigation;
using namespace Windows::System::Threading;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

MainPage::MainPage() : labels(1) {
    images = { cpu::Matrix(1, 1) };
    dashboard = cv::Mat(400, 400, CV_8UC3, cv::Scalar(0));
    InitializeComponent();
}


void MainPage::updateLayout(ThreadPoolTimer^ timer) {
    trainProgress->Dispatcher->RunAsync(
        Windows::UI::Core::CoreDispatcherPriority::Normal, 
        ref new Windows::UI::Core::DispatchedHandler([this] { 
            int p = network.getPosition() % (int)images.size();
            trainProgress->Value = p;
            epochText->Text = "" + (network.epoch + 1);
            lossText->Text = "" + network.meanLoss();
            trPrecText->Text = "" + network.trainPrecision();
            testPrecText->Text = "" + network.testPrecision();
        })
    );
}


void GENN::MainPage::updateImages(ThreadPoolTimer^ timer) {
    trainProgress->Dispatcher->RunAsync(
        Windows::UI::Core::CoreDispatcherPriority::Normal,
        ref new Windows::UI::Core::DispatchedHandler([this] {
            int p = network.getPosition();
            if (p > 64) {
                std::vector<cpu::Matrix> currentImages(images.begin() + p - 16, images.begin() + p);
                std::vector<int> predictions = network.getPredictions(16);
                dashboard = drawDashboard(currentImages, predictions, network.getLoss(), network.confusionMatrix);
                updateDashboard();
            }
        })
    );
}


void GENN::MainPage::loadMNIST(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e) {
    std::thread t([this]() {
        read_Mnist("train-images.idx3-ubyte", images);
        read_Mnist_Label("train-labels.idx1-ubyte", labels);
        read_Mnist("t10k-images.idx3-ubyte", testImages);
        read_Mnist_Label("t10k-labels.idx1-ubyte", testLabels);
        testOrder.resize(testImages.size());
        for (int i = 0; i < testOrder.size(); i++) {
            testOrder[i] = i;
        }
        network.testOrder = testOrder;
        network.setTrainData(images, labels);
        network.setTestData(testImages, testLabels);
        
    });
    t.join();
    trainProgress->Maximum = 60000.0;
    startButton->IsEnabled = true;
}


void GENN::MainPage::updateDashboard() {
    // Create the WriteableBitmap
    WriteableBitmap^ bitmap = ref new WriteableBitmap(dashboard.cols, dashboard.rows);

    // Get access to the pixels
    IBuffer^ buffer = bitmap->PixelBuffer;
    unsigned char* dstPixels = nullptr;

    // Obtain IBufferByteAccess
    ComPtr<IBufferByteAccess> pBufferByteAccess;
    ComPtr<IInspectable> pBuffer((IInspectable*)buffer);
    pBuffer.As(&pBufferByteAccess);

    // Get pointer to pixel bytes
    HRESULT get_bytes = pBufferByteAccess->Buffer(&dstPixels);
    if (get_bytes == S_OK) {
        memcpy(dstPixels, dashboard.data, dashboard.step[1] * dashboard.cols * dashboard.rows);

        // Set the bitmap to the Image element
        canvas->Source = bitmap;
    }
    else {
        printf("Error loading image into buffer\n");
    }
}


void GENN::MainPage::startTraining(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e) {
    if (network.isTraining()) {
        network.stopTraining();
        trainThread.join();
        uiUpdateTimer->Cancel();
        dashboardUpdateTimer->Cancel();
        startButton->Content = "Start Training";
        testButton->IsEnabled = true;
        pauseButton->IsEnabled = false;
    }
    else if (!network.isPaused()) {
        startButton->Content = "Stop Training";
        pauseButton->Content = "Pause Training";
        trPrecText->Text = "";
        testPrecText->Text = "";
        testButton->IsEnabled = false;
        pauseButton->IsEnabled = true;
        trainThread = std::thread(&nn::Network::startTraining, &network);

        TimeSpan period;
        period.Duration = 10000000;
        uiUpdateTimer = ThreadPoolTimer::CreatePeriodicTimer(ref new TimerElapsedHandler(this, &MainPage::updateLayout), period);

        period.Duration = 10000000;
        dashboardUpdateTimer = ThreadPoolTimer::CreatePeriodicTimer(ref new TimerElapsedHandler(this, &MainPage::updateImages), period);
    }
    else {
        uiUpdateTimer->Cancel();
        dashboardUpdateTimer->Cancel();
        startButton->Content = "Start Training";
        network.stopTraining();
        pauseButton->IsEnabled = false;
        testButton->IsEnabled = true;
    }
}


void GENN::MainPage::pauseTraining(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e) {
    if (network.isTraining()) {
        network.pauseTraining();
        trainThread.join();
        uiUpdateTimer->Cancel();
        dashboardUpdateTimer->Cancel();
        pauseButton->Content = "Resume Training";
        testButton->IsEnabled = true;
    }
    else {
        pauseButton->Content = "Pause Training";
        testButton->IsEnabled = false;
        trainThread = std::thread(&nn::Network::resumeTraining, &network);

        TimeSpan period;
        period.Duration = 10000000;
        uiUpdateTimer = ThreadPoolTimer::CreatePeriodicTimer(ref new TimerElapsedHandler(this, &MainPage::updateLayout), period);

        period.Duration = 10000000;
        dashboardUpdateTimer = ThreadPoolTimer::CreatePeriodicTimer(ref new TimerElapsedHandler(this, &MainPage::updateImages), period);
    }
}


void GENN::MainPage::testNN(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e) {
    float prec = network.test(1000);
    testPrecText->Text = "" + prec;

    cv::Mat confMatrix = plotConfusionMatrix(network.confusionMatrix);
    cv::Mat roi = dashboard(cv::Rect(cv::Point(200, 40), cv::Size(200, 200)));
    confMatrix.convertTo(confMatrix, CV_8UC3);
    cv::cvtColor(confMatrix, confMatrix, CV_GRAY2BGRA);
    confMatrix.copyTo(roi);
    updateDashboard();
}
