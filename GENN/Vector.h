#pragma once

namespace cpu {


class Vector {
public:
    int s;
    float* data;

    Vector() = default;
    Vector(int s);
    Vector(const Vector& v);
    Vector(Vector&& v) noexcept;
    Vector& operator= (const Vector& v);
    Vector& operator= (Vector&& v) noexcept;
    ~Vector();

    float& operator[] (int i);
    const float& operator[] (int i) const;
    Vector operator+ (const Vector& v);
    int argmax() const;
};

}
