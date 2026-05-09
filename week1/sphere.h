#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere: public hittable {
    public:
        sphere() {}
        sphere(point3 center, double radius) : center(center), radius(radius) {}

        virtual bool hit(
            const ray& r, double t_min, double t_max, hit_record& rec
        ) const;
    
    public:
        point3 center;
        double radius;
};

bool sphere::hit(
    const ray& r, double t_min, double t_max, hit_record& rec
) const {
    vec3 oc = r.origin() - center;
    double a = r.direction().length_squared();
    double half_b = dot(r.direction(), oc);
    double c = oc.length_squared() - radius*radius;

    double discriminant = half_b*half_b - a*c;
    if (discriminant > 0) {
        auto root = sqrt(discriminant);
        auto t_near = (-half_b - root) / a;
        if (t_near < t_max && t_near > t_min) {
            rec.t = t_near;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            return true;
        }
        auto t_far = (-half_b + root) / a;
        if (t_far < t_max && t_far > t_min) {
            rec.t = t_far;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            return true;
        }
    }
    return false;
}

#endif