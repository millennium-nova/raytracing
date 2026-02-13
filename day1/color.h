#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

// [0,1] の範囲の色を [0,255] の範囲に変換して出力
void write_color(std::ostream &out, color pixel_color) { //static_cast<target_type>(arg)
    out << static_cast<int>(255.999 * pixel_color.x()) << ' '
        << static_cast<int>(255.999 * pixel_color.y()) << ' '
        << static_cast<int>(255.999 * pixel_color.z()) << '\n';
}

#endif