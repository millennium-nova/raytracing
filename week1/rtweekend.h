#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

// using
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// 定数
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.14159926535897932385;

// ユーティリティ関数
inline double degrees_to_radians(double degrees) {
    return (degrees/180) * pi;
}

inline double random_double() {
    // [0,1) の実数乱数を返す
    return rand() / (RAND_MAX + 1.0); // < 1.0 を保証するために + 1.0 をする
}

inline double random_double(double min, double max) {
    // [min,max) の実数乱数を返す
    return min + (max-min)*random_double();
}

// 共通ヘッダー
#include "ray.h"
#include "vec3.h"

#endif