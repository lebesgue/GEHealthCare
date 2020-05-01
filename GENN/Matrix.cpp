#include "pch.h"
#include "Matrix.h"


namespace cpu {


Matrix::Matrix(int h, int w) : w(w), h(h) {
    data = (float*)malloc(w * h * sizeof(float));
    for (int i = 0; i < w * h; i++) {
        data[i] = 0.f;
    }
}


Matrix::Matrix(const Matrix& m) : w(m.w), h(m.h) {
    data = (float*)malloc(w * h * sizeof(float));
    for (int i = 0; i < w * h; i++) {
        data[i] = m.data[i];
    }
}


Matrix& Matrix::operator=(const Matrix& m) {
    if (this != &m) {
        w = m.w;
        h = m.h;
        free(data);
        data = (float*)malloc(w * h * sizeof(float));
        memcpy(data, m.data, w * h * sizeof(float));
    }
    return *this;
}


Matrix::~Matrix() {
    free(data);
}


float& Matrix::operator()(int row, int col) {
    if (row >= h)
        throw std::runtime_error("Overindexing Matrix. Rows: " + std::to_string(h + 1) + ", index: " + std::to_string(row));
    if (col >= w)
        throw std::runtime_error("Overindexing Matrix. Columns: " + std::to_string(w + 1) + ", index: " + std::to_string(col));

    return data[row * w + col];
}

const float& Matrix::operator()(int row, int col) const {
    if (row >= h)
        throw std::runtime_error("Overindexing Matrix. Rows: " + std::to_string(h + 1) + ", index: " + std::to_string(row));
    if (col >= w)
        throw std::runtime_error("Overindexing Matrix. Columns: " + std::to_string(w + 1) + ", index: " + std::to_string(col));

    return data[row * w + col];
}


Matrix Matrix::mul(const Matrix& m) {
    Matrix r(m.w, h);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < m.w; j++) {
            r(i, j) = 0.0f;
            for (int k = 0; k < m.h; k++) {
                r(i, j) += (*this)(i, k) * m(k, j);
            }
        }
    }
    return r;
}


Vector Matrix::mul(const Vector& v) {
    Vector r(h);
    for (int i = 0; i < h; i++) {
        r[i] = 0.0f;
        for (int j = 0; j < v.s; j++) {
            r[i] += (*this)(i, j) * v[j];
        }
    }
    return r;
}


}