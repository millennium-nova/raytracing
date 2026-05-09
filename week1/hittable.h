#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

struct hit_record {
    point3 p; // レイと物体の交差点
    vec3 normal; // 交差点における法線ベクトル
    double t; // op = oc + t * ray_dir 
    bool front_face; // レイが物体の外側から入射しているかどうか

    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0; // 内積が負であれば、例は物体の外側にある
        normal = front_face ? outward_normal : -outward_normal; // 法線は常にレイと逆方向を向くようにする
    }
};

class hittable {
 public:
    virtual ~hittable() {}
    virtual bool hit(
        const ray& r, double t_min, double t_max, hit_record& rec
    ) const = 0; 
};


#endif