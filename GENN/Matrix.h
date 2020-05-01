#pragma once

#include "Vector.h"


namespace cpu {


class Matrix {
public:
    float* data;
    int w;
    int h;

    Matrix() = default;
    Matrix(int h, int w);
    Matrix(const Matrix& m);
    Matrix& operator= (const Matrix& m);
    ~Matrix();

    float& operator() (int row, int col);
    const float& operator() (int row, int col) const;

    Matrix mul(const Matrix& m);
    Vector mul(const Vector& v);
};


}
