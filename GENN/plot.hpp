#pragma once

#include "pch.h"
#include "NN.h"


std::vector<float> wAvg(const std::vector<float>& values, int n) {
    std::vector<float> r(0);
    for (int i = 0; (i + 1) * n < values.size(); ++i) {
        r.emplace_back(std::accumulate(values.begin() + i * n, values.begin() + (i + 1) * n, 0.0f) / n);
    }
    return r;
}


cv::Mat plotConfusionMatrix(const cpu::Matrix& confusionMatrix) {
    cv::Mat cmImg(200, 200, CV_32F, cv::Scalar(0, 0, 0));
    cpu::Vector rowSums(confusionMatrix.h);
    for (int i = 0; i < confusionMatrix.h; i++) {
        for (int j = 0; j < confusionMatrix.w; j++) {
            rowSums[i] += confusionMatrix.data[i * confusionMatrix.w + j];
        }
    }

    for (int i = 0; i < confusionMatrix.h; i++) {
        for (int j = 0; j < confusionMatrix.w; j++) {
            float val = 255 * (confusionMatrix.data[i * confusionMatrix.w + j] / rowSums[i]);
            cv::rectangle(cmImg, cv::Rect(cv::Point(i * 20, j * 20), cv::Point((i + 1) * 20, (j + 1) * 20)), cv::Scalar(val), CV_FILLED, CV_AA);
        }
    }

    return cmImg;
}


cv::Mat plotLoss(const std::vector<float>& loss) {
    cv::Mat plot(200, 400, CV_32F, cv::Scalar(0));

    cv::line(plot, cv::Point(20, 10), cv::Point(20, 200), cv::Scalar(180), 1, CV_AA);
    cv::line(plot, cv::Point(0, 190), cv::Point(400, 190), cv::Scalar(180), 1, CV_AA);

    std::vector<float> avgLoss = wAvg(loss, 1000);
    int n = static_cast<int>(avgLoss.size());
    if (n < 2)
        return plot;

    int dn = 380 / (n - 1);
    float maxLoss = static_cast<float>(*std::max_element(avgLoss.begin(), avgLoss.end()) + 0.25f);
    int dticks = static_cast<int>(180.0 / maxLoss);

    for (int i = 1; i * dticks <= 180; ++i) {
        cv::line(plot, cv::Point(15, 190 - i * dticks), cv::Point(20, 190 - i * dticks), cv::Scalar(180), 1, CV_AA);
        cv::putText(plot, std::to_string(i), cv::Point(5, 190 - i * dticks + 3), CV_FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(180), 1, CV_AA);
    }

    auto transform = [&](float loss, int iter) {
        int x = 20 + dn * iter;
        int y = 190 - static_cast<int>(loss / maxLoss * 180);
        return cv::Point(x, y);
    };

    for (int i = 0; i < n - 1; i++) {
        cv::Point p1 = transform(avgLoss[i], i);
        cv::Point p2 = transform(avgLoss[i + 1], i + 1);
        cv::line(plot, p1, p2, cv::Scalar(255), 1, CV_AA);
    }

    return plot;
}


cv::Mat drawDashboard(
    const std::vector<cpu::Matrix>& displayImages, 
    const std::vector<int>& predictions, 
    const std::vector<float>& loss,
    const cpu::Matrix& confusionMatrix) {
    cv::Mat dashboard(470, 400, CV_32F, cv::Scalar(0));

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cv::Mat smallImage = cv::Mat(cv::Size(28, 28), CV_32F, displayImages[i * 4 + j].data);
            int tlx = 50 * i;
            int tly = 50 * j + 40;
            cv::Rect roi(cv::Point(tlx + 11, tly + 4), cv::Size(28, 28));
            cv::Mat destinationROI = dashboard(roi);
            smallImage.copyTo(destinationROI);
            cv::putText(dashboard,
                        std::to_string(predictions[i * 4 + j]),
                        cv::Point(tlx + 20, tly + 46),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(255), 1, CV_AA);
        }
    }
    cv::putText(dashboard, "SAMPLE IMAGES", cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, CV_AA);
    cv::putText(dashboard, "CONFUSION MATRIX", cv::Point(220, 20), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, CV_AA);
    cv::putText(dashboard, "MEAN LOSS (LAST 1000 SAMPLES)", cv::Point(20, 260), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, CV_AA);

    cv::Mat roi = dashboard(cv::Rect(cv::Point(0, 270), cv::Size(400, 200)));
    plotLoss(loss).copyTo(roi);
    roi = dashboard(cv::Rect(cv::Point(200, 40), cv::Size(200, 200)));
    plotConfusionMatrix(confusionMatrix).copyTo(roi);

    dashboard.convertTo(dashboard, CV_8UC3);
    cv::cvtColor(dashboard, dashboard, CV_GRAY2BGRA);

    return dashboard;
}