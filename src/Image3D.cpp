#include <stdexcept>
#include <string>
#include "Image3D.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <opencv/cv.hpp>
#include "Point3D.h"

using namespace std;
using namespace cv;
float *data;


Image3D::Image3D(int height, int width, int depth) {
    this->depth = depth;
    this->height = height;
    this->width = width;
    this->data = new float[depth * width * height]();
}

Image3D::Image3D(int height, int width, int depth, float *data) {
    this->depth = depth;
    this->height = height;
    this->width = width;
    this->data = data;
}

Image3D Image3D::read_image(string folder, string filename, string extension, int depth) {

    Mat gray_image = imread(folder + "/" + filename + "_0" + extension, IMREAD_GRAYSCALE);
    Mat image;
    gray_image.convertTo(image, CV_32F);
    int width = image.cols, height = image.rows;
    Image3D result = Image3D::Image3D(height, width, depth);
    for (int k = 0; k < depth; k++) {
        cout << "k " << k << endl;
        gray_image = imread(folder + "/" + filename + "_0" + extension, IMREAD_GRAYSCALE);
        gray_image.convertTo(image, CV_32F);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float x = image.at<float>(i, j) / 255.;
                result.set(i, j, k, x);
            }
        }
    }
    return result;
}

float Image3D::at(int i, int j, int k, bool zeroPadding) {
    if ((i >= height) or (i < 0) or (j >= width) or (j < 0) or (k >= depth) or (k < 0)) {
        if (zeroPadding) {
            return 0;
        } else {
            throw std::invalid_argument("tried to access out of bond voxel");
        }
    }
    return data[k * height * width + j * height + i];
};

float Image3D::at(int i, int j, int k) {
    if ((i >= height) or (i < 0) or (j >= width) or (j < 0) or (k >= depth) or (k < 0)) {

        throw std::invalid_argument("tried to access out of bond voxel"+ to_string(i) + ',' + to_string(j) + ',' + to_string(k));
    }
    return data[k * height * width + j * height + i];
};

void Image3D::set(int i, int j, int k, float v) {
    if ((i >= height) or (i < 0) or (j >= width) or (j < 0) or (k >= depth) or (k < 0)) {
        throw std::invalid_argument(
                "tried to access out of bond voxel" + to_string(i) + ',' + to_string(j) + ',' + to_string(k));
    }
    data[k * height * width + j * height + i] = v;
}

Mat Image3D::get_slice(int k) {
    Mat image = Mat(height, width, CV_32F);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image.at<float>(i, j) = at(i, j, k);
        }
    }
    return image;
}


Image3D Image3D::zeros(int height, int width, int depth) {
    Image3D res = Image3D(height, width, depth);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                res.set(i, j, k, 0);
            }
        }
    }
    return res;
}

Image3D Image3D::norm2() {
    Image3D res = Image3D(height, width, depth);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                res.set(i, j, k, at(i, j, k) * at(i, j, k));
            }
        }
    }
    return res;
}

Image3D Image3D::add(Image3D other) {
    if ((other.width != width) or (other.height != height) or (other.depth != depth)) {
        throw std::invalid_argument("shape mismatch in add");
    }

    Image3D res = Image3D(height, width, depth);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                res.set(i, j, k, at(i, j, k) + other.at(i, j, k));
            }
        }
    }
    return res;
}

Image3D Image3D::conv3D(Image3D kernel) {
    Image3D conv = zeros(height, width, depth);
    int offset_i = (kernel.height - 1) / 2;
    int offset_j = (kernel.width - 1) / 2;
    int offset_k = (kernel.depth - 1) / 2;

    for (int i = 0; i < height-kernel.height+1; i++) {
        for (int j = 0; j < width-kernel.width+1; j++) {
            for (int k = 0; k < depth-kernel.depth+1; k++) {
                float c = 0;
                for (int l = 0; l < kernel.height; l++) {
                    for (int m = 0; m < kernel.width; m++) {
                        for (int n = 0; n < kernel.depth; n++) {
                            c = c + at(i + l, j + m, k + n) * kernel.at(l, m, n);
                        }
                    }
                }
                conv.set(i + offset_i, j + offset_j, k + offset_k, c);
            }
        }
    }
    return conv;
}

float Image3D::max(){
    float current_max=0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                float x=at(i, j, k);
                current_max= x>current_max? x : current_max;
            }
        }
    }
    return current_max;
}

void Image3D::mul(float x){
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                float y=at(i, j, k);
                set(i,j,k,x*y);
            }
        }
    }
}

//Image3D Image3D::gradX() {
//    float kernelData[]={-1,0,1};
//    float kernelData2[]={1,2,1};
//    Image3D kernel = Image3D(3, 1, 1, kernelData);
//    return ((this->conv3D(kernel)).conv3D(Image3D(1,3,1,kernelData2))).conv3D(Image3D(1,1,3,kernelData2));
//}
//
//Image3D Image3D::gradY() {
//    float kernelData[]={-1,0,1};
//    float kernelData2[]={1,2,1};
//    Image3D kernel = Image3D(1, 3, 1, kernelData);
//    return ((this->conv3D(kernel)).conv3D(Image3D(3,1,1,kernelData2))).conv3D(Image3D(1,1,3,kernelData2));
//}
//
//Image3D Image3D::gradZ() {
//    float kernelData[]={-1,0,1};
//    float kernelData2[]={1,2,1};
//    Image3D kernel = Image3D(1, 1, 3, kernelData);
//    return ((this->conv3D(kernel)).conv3D(Image3D(1,3,1,kernelData2))).conv3D(Image3D(3,1,1,kernelData2));
//}

Image3D Image3D::gradX() {
    float kernelData[]={-1,0,1};
    Image3D kernel = Image3D(3, 1, 1, kernelData);
    return this->conv3D(kernel);
}

Image3D Image3D::gradY() {
    float kernelData[]={-1,0,1};
    Image3D kernel = Image3D(1, 3, 1, kernelData);
    return this->conv3D(kernel);
}

Image3D Image3D::gradZ() {
    float kernelData[]={-1,0,1};
    Image3D kernel = Image3D(1, 1, 3, kernelData);
    return this->conv3D(kernel);
}

Image3D Image3D::gradNorm2() {

    Image3D gradXnorm=this->gradX().norm2();
    Image3D gradYnorm=this->gradY().norm2();
    Image3D gradZnorm=this->gradZ().norm2();

    Image3D grad3D=(gradXnorm.add(gradYnorm)).add(gradZnorm);
    return grad3D;
}

float Image3D::at(Point3D p) {
    return this->at(p.get_i(),p.get_j(),p.get_k());
}

