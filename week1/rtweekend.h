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

// 共通ヘッダー
#include "ray.h"
#include "vec3.h"

#endif