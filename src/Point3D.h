//
// Created by Amaury Sabran on 11/15/17.
//

#pragma once

#ifndef PROJECT_POINT3D_H
#define PROJECT_POINT3D_H


class Point3D {

private:
    int i;
    int j;
    int k;
public:
    int get_i();
    int get_j();
    int get_k();

    Point3D(int i,int j,int k);

};


#endif //PROJECT_POINT3D_H
