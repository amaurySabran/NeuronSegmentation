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

int main(int argc, char *argv[]) {
    float alpha=1;
    float beta=1;
    int k=1;
    int slice=0;

    if (argc>=2){
        k=atoi(argv[1]);
        cout<<"k :"<<k<<endl;
    }
    if (argc>=3){
        slice=atoi(argv[2]);
        cout<<"slice :"<<slice<<endl;
    }
    if (argc>=4){
        alpha=atof(argv[3]);
        cout<<"alpha :"<<alpha<<endl;
    }
    if (argc>=5){
        beta=atof(argv[4]);
        cout<<"beta :"<<beta<<endl;
    }

    Image3D testing=Image3D::read_image("data/training","training",".jpg",k);
    Mat image = testing.get_slice(slice);
    cout<<"got the slice";
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    Image3D grad = testing.gradNorm2();

    cout<<"image max"<<testing.max()<<endl;
    cout<<"grad max"<<grad.max()<<endl;
    grad.mul(1/grad.max());
    Mat gradMat= grad.get_slice(slice);
    cout<<"grad mean "<<mean(gradMat)<<endl;

//    imshow("Display window",image);
//    waitKey(0);

    imshow("Display window",gradMat);
    waitKey(0);

    Image3D result= getGraphCut(testing,grad,testing,alpha,beta);

    Mat resultSlice=result.get_slice(slice);

    imshow("Display window",resultSlice);
    waitKey(0);

    cout<<"done";
}