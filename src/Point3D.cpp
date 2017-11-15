//
// Created by Amaury Sabran on 11/15/17.
//

#include "Point3D.h"

int Point3D::get_i() {
    return i;
}

int Point3D::get_j() {
    return j;
}

int Point3D::get_k() {
    return k;
}

Point3D::Point3D(int a, int b, int c) {
    i=a;
    j=b;
    k=c;
}
