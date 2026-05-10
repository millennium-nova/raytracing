#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

// [0,1] の範囲の色を [0,255] の範囲に変換して出力
void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) { //static_cast<target_type>(arg)
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    auto scale = 1.0 / samples_per_pixel;

    r *= scale;
    g *= scale;
    b *= scale;
    
    
    
    
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

#endif