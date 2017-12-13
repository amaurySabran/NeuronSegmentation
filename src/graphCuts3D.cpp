#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <opencv/cv.hpp>
#include "Image3D.h"
#include "maxflow/graph.h"

using namespace std;
using namespace cv;


float lambda(Point3D p, Point3D q, Image3D grad3D, float alpha, float beta) {
    //p et q sont les deux points dont on évalue la force de liaison

    //param�tres de g

    float gradp = grad3D.at(p);
    float gradq = grad3D.at(q);
    gradp = alpha / (1 + beta * gradp);
    gradq = alpha / (1+ beta* gradq);
    return (gradp + gradq) / 2;
}


Graph<float, float, float> buildGraph(Image3D I, Image3D neuronConfidence, Image3D grad3D, float alpha, float beta) {
    int h = I.height;
    int w = I.width;
    int d = I.depth;
    Graph<float, float, float> g(h * w * d, 6 * h * w * d);
    g.add_node(6 * h * w * d);

    for (int k = 0; k < I.depth; k++) {
        for (int i = 0; i < I.height; i++) {
            for (int j = 0; j < I.width; j++) {
                int pos = k * (h * w) + i * w + j;

                //edges avec puit et source

                g.add_tweights(pos, neuronConfidence.at(i, j, k), 1 - neuronConfidence.at(i, j, k));

                if ((k < I.depth - 1)) {
                    float l = lambda(Point3D(i, j, k), Point3D(i, j, k + 1), grad3D, alpha, beta);
                    g.add_edge(pos, pos + h * w, l, l);
                }

                if ((j < I.width - 1)) {
                    float l = lambda(Point3D(i, j + 1, k), Point3D(i, j, k), grad3D, alpha, beta);
                    g.add_edge(pos, pos + 1, l, l);
                }

                if ((i < I.height - 1)) {
                    float l = lambda(Point3D(i, j, k), Point3D(i + 1, j, k), grad3D, alpha, beta);
                    g.add_edge(pos, pos + w, l, l);
                }
            }
        }
    }
    return g;
}

Image3D getGraphCut(Image3D I, Image3D grad, Image3D neuronConfidence, float alpha, float beta) {
    Graph<float, float, float> g = buildGraph(I, neuronConfidence, grad, alpha, beta);
    float flow = g.maxflow();
    Image3D result(I.height, I.width, I.depth);
    int count = 0;
    for (int i = 0; i < I.height; i++) {
        for (int j = 0; j < I.width; j++) {
            for (int k = 0; k < I.depth; k++) {
                int pos = k * (I.height * I.width) + i * I.width + j;
                result.set(i, j, k, g.what_segment(pos));
            }
        }
    }
    return result;
}