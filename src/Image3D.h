#pragma once

#include <opencv2/core/mat.hpp>
#include "Point3D.h"


class Image3D {
public:
	Image3D(int height,int width,int depth);
	Image3D(int height, int width, int depth, float *data);
    static Image3D zeros(int height,int width,int depth);
    static Image3D read_image(std::string folder, std::string filename, std::string extension, int depth);
    static std::string metrics(Image3D result, Image3D groundTruth);


    float* data;
	int height;
	int width;
	int depth;


    void set(int i,int j,int k,float v);
    float at(Point3D p);
    float at(int i, int j, int k);
    float at(int i, int j, int k, float defaultValue);

	Image3D conv3D(Image3D kernel);
    Image3D erode(int kHeight,int kWidth,int kDepth);
    Image3D dilate(int si,int sj,int sk);
    cv::Mat get_slice(int k);
    Image3D norm2();
    Image3D add(Image3D other);
    Image3D gradX();
    Image3D gradY();
    Image3D gradZ();
    Image3D gradNorm2();

    void mul(float x);

    float max();
};
