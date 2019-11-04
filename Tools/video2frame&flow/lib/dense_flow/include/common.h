//
// Created by yjxiong on 11/18/15.
//

#ifndef DENSEFLOW_COMMON_H_H
#define DENSEFLOW_COMMON_H_H



#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <iostream>
using namespace cv;

void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
                        double lowerBound, double higherBound);
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color);

void encodeFlowMap(const Mat& flow_map_x, const Mat& flow_map_y,
                   vector<uchar>& encoded_x, vector<uchar>& encoded_y,
                   int bound, bool to_jpg=true);

inline void initializeMats(const Mat& frame,
                           Mat& capture_image, Mat& capture_gray,
                           Mat& prev_image, Mat& prev_gray){
    capture_image.create(frame.size(), CV_8UC3);
    capture_gray.create(frame.size(), CV_8UC1);

    prev_image.create(frame.size(), CV_8UC3);
    prev_gray.create(frame.size(), CV_8UC1);
}

void writeImages(vector<vector<uchar>> images, string name_temp);
//////////////////////output the first frame as "img_00000.png"
void writeImages2(vector<vector<uchar>> images, string name_temp);
//////////////////////over(4/4 [extract_flow_gpu.cpp;dense_flow_gpu.cpp;common.cpp])
void writeImages3(vector<Mat> images, string name_temp);
bool SaveMatBinary(const std::string& filename, const cv::Mat& output);
bool writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat);


//output the flow map as txt
void writeFlow_map(const Mat& flow_map_x, const Mat& flow_map_y);
void writeFlow_map2(string name_temp, const Mat& flow_y);
//over

#endif //DENSEFLOW_COMMON_H_H

