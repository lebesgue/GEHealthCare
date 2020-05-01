#include "pch.h"
#include "Vector.h"


namespace cpu {


Vector::Vector(int s = 0) : s(s) {
    data = (float*)malloc(s * sizeof(float));
    for (int i = 0; i < s; i++) {
        data[i] = 0.f;
    }
}


Vector::Vector(const Vector& v) : s(v.s) {
    data = (float*)malloc(s * sizeof(float));
    for (int i = 0; i < s; i++) {
        data[i] = v[i];
    }
}


Vector::Vector(Vector&& v) noexcept {
    s = v.s;
    data = v.data;
    v.data = nullptr;
}


Vector& Vector::operator=(const Vector& v) {
    if (this != &v) {
        s = v.s;
        free(data);
        data = (float*)malloc(s * sizeof(float));
        memcpy(data, v.data, s * sizeof(float));
    }
    return *this;
}


Vector& Vector::operator=(Vector&& v) noexcept {
    free(data);
    s = v.s;
    data = v.data;
    v.data = nullptr;
    return *this;
}


Vector::~Vector() {
    free(data);
}


float& Vector::operator[](int i) {
    if (i > s)
        throw std::runtime_error("Overindexing Vector. Length: " + std::to_string(s + 1) + ", index: " + std::to_string(i));

    return data[i];
}


const float& Vector::operator[](int i) const {
    if (i > s)
        throw std::runtime_error("Overindexing Vector. Length: " + std::to_string(s + 1) + ", index: " + std::to_string(i));

    return data[i];
}


Vector Vector::operator+(const Vector& v) {
    if (v.s != s)
        throw std::runtime_error("Trying to add Vectors with different sizes: (" + std::to_string(s) + ", " + std::to_string(v.s) + ")");
    Vector r(s);
    for (int i = 0; i < v.s; i++) {
        r[i] = v[i] + data[i];
    }
    return r;
}


int Vector::argmax() const {
    int am = 0;
    float m = data[0];
    for (int i = 0; i < s; i++) {
        if (data[i] > m) {
            am = i;
            m = data[i];
        }
    }
    return am;
}

}