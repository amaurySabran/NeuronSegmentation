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

float Image3D::at(int i, int j, int k, float defaultValue) {
    if ((i >= height) or (i < 0) or (j >= width) or (j < 0) or (k >= depth) or (k < 0)) {
        return defaultValue;
    }
    return data[k * height * width + j * height + i];
};

float Image3D::at(int i, int j, int k) {
    if ((i >= height) or (i < 0) or (j >= width) or (j < 0) or (k >= depth) or (k < 0)) {

        throw std::invalid_argument(
                "tried to access out of bond voxel" + to_string(i) + ',' + to_string(j) + ',' + to_string(k));
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

    for (int i = -offset_i; i < height - offset_i; i++) {
        for (int j = -offset_j; j < width - offset_j; j++) {
            for (int k = -offset_k; k < depth - offset_k; k++) {
                float c = 0;
                for (int l = 0; l < kernel.height; l++) {
                    for (int m = 0; m < kernel.width; m++) {
                        for (int n = 0; n < kernel.depth; n++) {
                            c = c + at(i + l, j + m, k + n, 0) * kernel.at(l, m, n);
                        }
                    }
                }
                conv.set(i + offset_i, j + offset_j, k + offset_k, c);
            }
        }
    }
    return conv;
}

float Image3D::max() {
    float current_max = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                float x = at(i, j, k);
                current_max = x > current_max ? x : current_max;
            }
        }
    }
    return current_max;
}

void Image3D::mul(float x) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                float y = at(i, j, k);
                set(i, j, k, x * y);
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
    float kernelData[] = {-1, 0, 1};
    Image3D kernel = Image3D(3, 1, 1, kernelData);
    return this->conv3D(kernel);
}

Image3D Image3D::gradY() {
    float kernelData[] = {-1, 0, 1};
    Image3D kernel = Image3D(1, 3, 1, kernelData);
    return this->conv3D(kernel);
}

Image3D Image3D::gradZ() {
    float kernelData[] = {-1, 0, 1};
    Image3D kernel = Image3D(1, 1, 3, kernelData);
    return this->conv3D(kernel);
}

Image3D Image3D::gradNorm2() {

    Image3D gradXnorm = this->gradX().norm2();
    Image3D gradYnorm = this->gradY().norm2();
    Image3D gradZnorm = this->gradZ().norm2();

    Image3D grad3D = (gradXnorm.add(gradYnorm)).add(gradZnorm);
    return grad3D;
}

float Image3D::at(Point3D p) {
    return this->at(p.get_i(), p.get_j(), p.get_k());
}

Image3D Image3D::erode(int kHeight, int kWidth, int kDepth) {
    Image3D result = zeros(height, width, depth);

    int offset_i = (kHeight - 1) / 2;
    int offset_j = (kWidth - 1) / 2;
    int offset_k = (kDepth - 1) / 2;

    for (int i = -offset_i; i < height - offset_i; i++) {
        for (int j = -offset_j; j < width - offset_j; j++) {
            for (int k = -offset_k; k < depth - offset_k; k++) {
                float min = 255;
                for (int l = 0; l < kHeight; l++) {
                    for (int m = 0; m < kWidth; m++) {
                        for (int n = 0; n < kDepth; n++) {
                            float z = at(i + l, j + m, k + n, 255);
                            //high default value so that borders are not taken into account
                            min = min < z ? min : z;
                        }
                    }
                }
                result.set(i + offset_i, j + offset_j, k + offset_k, min);
            }
        }
    }
    return result;
}

Image3D Image3D::dilate(int kHeight, int kWidth, int kDepth) {
    Image3D result = zeros(height, width, depth);

    int offset_i = (kHeight - 1) / 2;
    int offset_j = (kWidth - 1) / 2;
    int offset_k = (kDepth - 1) / 2;

    for (int i = -offset_i; i < height - offset_i; i++) {
        for (int j = -offset_j; j < width - offset_j; j++) {
            for (int k = -offset_k; k < depth - offset_k; k++) {
                float max = -1;
                for (int l = 0; l < kHeight; l++) {
                    for (int m = 0; m < kWidth; m++) {
                        for (int n = 0; n < kDepth; n++) {
                            float z = at(i + l, j + m, k + n, -1);
                            //high default value so that borders are not taken into account
                            max = max > z ? max : z;
                        }
                    }
                }
                result.set(i + offset_i, j + offset_j, k + offset_k, max);
            }
        }
    }
    return result;
}

string Image3D::metrics(Image3D result, Image3D groundTruth) {
    int TP=0;
    int TN=0;
    int FP=0;
    int FN=0;
    for (int i=0;i<groundTruth.height;i++){
        for (int j=0;j<groundTruth.width;j++){
            for (int k=0;k<groundTruth.depth;k++){
                if (result.at(i,j,k)>.5){
                    if (groundTruth.at(i,j,k)>.5){
                        TP+=1;
                    }
                    else{
                        FP+=1;
                    }
                }
                else{
                    if (groundTruth.at(i,j,k)>.5){
                        FN+=1;
                    }
                    else{
                        TN+=1;
                    }
                }

            }
        }
    }
    float precision = (TP+.0)/(TP+FP);
    float recall = (TP+.0)/(TP+FN);
    float accuracy = (TP+TN+.0)/(TP+TN+FP+FN);
    float iou= (TP+0.0)/(TP+FP+FN+0.0);
    string out;
    out+="Precision: "+to_string(precision)+'\n';
    out+="Recall: "+to_string(recall)+'\n';
    out+="Accuracy: "+to_string(accuracy)+'\n';
    out+="Intersection over Union: "+to_string(iou)+'\n';
    return out;
}

