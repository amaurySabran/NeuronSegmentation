//
// Created by Amaury Sabran on 10/25/17.
//

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <opencv/cv.hpp>
#include "Image3D.h"
#include "graphCuts3D.cpp"


using namespace std;
using namespace cv;

Mat addOverlay(Mat underlay, Mat overlay, int heatmap) {
    Mat result;
    Mat colorUnderlay;
    Mat colorOverlay;
    Mat colorUnderlay2;
    if (underlay.type() == 5) {
        cvtColor(underlay, colorUnderlay, CV_GRAY2BGR);
        colorUnderlay.convertTo(colorUnderlay2, CV_8U, 255);
    }
    else{
        colorUnderlay2=underlay;
    }

    Mat overlay2;
    overlay.convertTo(overlay2, CV_8U, 30);
    applyColorMap(overlay2, colorOverlay, heatmap);
    addWeighted(colorUnderlay2, 1, colorOverlay, 1, 0.0, result);
    return result;
}

int main(int argc, char *argv[]) {
    float alpha = 1.1;
    float beta = 2;
    int k = 80;
    // The strength of the connection in graph cuts is alpha/(1 +beta* grad2)
    int slice = 1;

    if (argc >= 2) {
        k = atoi(argv[1]);
        cout << "k :" << k << endl;
    }
    if (argc >= 3) {
        slice = atoi(argv[2]);
        cout << "slice :" << slice << endl;
    }
    if (argc >= 4) {
        alpha = atof(argv[3]);
        cout << "alpha :" << alpha << endl;
    }
    if (argc >= 5) {
        beta = atof(argv[4]);
        cout << "beta :" << beta << endl;
    }

    Image3D brainImage = Image3D::read_image("data/small_testing", "testing", ".jpg", k);
    cout<<"brain image max "<<brainImage.max();
    Image3D groundTruth = Image3D::read_image("data/small_testing_groundtruth", "testing_groundtruth", ".jpg", k);
    cout<<"groundtruth image max "<<groundTruth.max();
    Image3D confidence= Image3D::read_image("data/naive_confidence","confidence",".jpg",k);

    Image3D grad = brainImage.gradNorm2();
    grad.mul(1 / grad.max());

    imshow("brain image", brainImage.get_slice(slice));
    waitKey(0);

    Image3D result = getGraphCut(brainImage, grad, confidence, alpha, beta);

    Mat resultSlice = result.get_slice(slice);
    Mat trueResult = groundTruth.get_slice(slice);

    Mat overlay = addOverlay(brainImage.get_slice(slice), trueResult, COLORMAP_OCEAN);
    Mat overlay2 = addOverlay(overlay, resultSlice, COLORMAP_HOT);
    imshow("result (red: ours, blue: ground truth, purple: overlap)", overlay2);
    waitKey(0);
    cout << "Metrics :" << endl << Image3D::metrics(result, groundTruth) << endl;
}