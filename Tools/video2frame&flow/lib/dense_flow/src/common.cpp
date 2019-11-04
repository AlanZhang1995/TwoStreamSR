//
// Created by yjxiong on 11/18/15.
//

#include "common.h"
#include <fstream>

bool writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat)
{
    if(!ofs.is_open()){
        return false;
    }
    if(out_mat.empty()){
        int s = 0;
        ofs.write((const char*)(&s), sizeof(int));
        return true;
    }
    int type = out_mat.type();
    ofs.write((const char*)(&out_mat.rows), sizeof(int));
    ofs.write((const char*)(&out_mat.cols), sizeof(int));
    ofs.write((const char*)(&type), sizeof(int));
    ofs.write((const char*)(out_mat.data), out_mat.elemSize() * out_mat.total());

    return true;
}


//! Save cv::Mat as binary
/*!
\param[in] filename filaname to save
\param[in] output cvmat to save
*/
bool SaveMatBinary(const std::string& filename, const cv::Mat& output){
    std::ofstream ofs(filename, std::ios::binary);
    return writeMatBinary(ofs, output);
}


void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
                               double lowerBound, double higherBound) {
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
            img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
        }
    }
#undef CAST
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

void encodeFlowMap(const Mat& flow_map_x, const Mat& flow_map_y,
                   vector<uchar>& encoded_x, vector<uchar>& encoded_y,
                   int bound, bool to_jpg){
    Mat flow_img_x(flow_map_x.size(), CV_8UC1);
    Mat flow_img_y(flow_map_y.size(), CV_8UC1);

    //output the flow map as txt
    //writeFlow_map(flow_map_x,flow_map_y);
    //over
    convertFlowToImage(flow_map_x, flow_map_y, flow_img_x, flow_img_y,
                       -bound, bound);

    if (to_jpg) {
        imencode(".png", flow_img_x, encoded_x);
        imencode(".png", flow_img_y, encoded_y);
    }else {
        encoded_x.resize(flow_img_x.total());
        encoded_y.resize(flow_img_y.total());
        memcpy(encoded_x.data(), flow_img_x.data, flow_img_x.total());
        memcpy(encoded_y.data(), flow_img_y.data, flow_img_y.total());
    }
}

void writeImages(vector<vector<uchar>> images, string name_temp){
    for (int i = 0; i < images.size(); ++i){
        char tmp[256];
        sprintf(tmp, "_%05d.png", i+1);
        FILE* fp;
        fp = fopen((name_temp + tmp).c_str(), "wb");
        fwrite( images[i].data(), 1, images[i].size(), fp);
        fclose(fp);
    }
}

//////////////////////output the first frame as "img_00000.png"
void writeImages2(vector<vector<uchar>> images, string name_temp){
    for (int i = 0; i < images.size(); i++){
        char tmp[256];
        sprintf(tmp, "_%05d.png", i );
        FILE* fp;
        fp = fopen((name_temp + tmp).c_str(), "wb");
        fwrite( images[i].data(), 1, images[i].size(), fp);
        fclose(fp);
    }
}
//////////////////////over(3/4 [extract_flow_gpu.cpp;dense_flow_gpu.cpp;common.h])

//////////////////////
void writeImages3(vector<Mat> images, string name_temp){
    int num=images.size();
    //printf("%d\n", num);
    //namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);  ////DEBUG1(1/2)
    for (int i = 0; i < num; i++){
        char tmp[256];
        //printf("%d\n", i);
        //imshow("MyWindow",images[i]);          ////DEBUG1(2/2)
        //waitKey(0);

        /*//method1-----'FileStorage' + output in order
        sprintf(tmp, "_%05d.xml", i+1 );
        FileStorage fs((name_temp + tmp).c_str(), FileStorage::WRITE);
        fs<<"flow"<<images[i];
        fs.release();
        */

        //method2-----3rd part implenmentation
        sprintf(tmp, "_%05d.mat", i+1 );
        //printf("%s\n", (name_temp + tmp).c_str());
        SaveMatBinary((name_temp + tmp).c_str(), images[i]);

        /*//method3-----'txt'+ output in reversed order
        sprintf(tmp, "_%05d.txt", i+1 );
        writeFlow_map2((name_temp + tmp).c_str(), images.back());
        images.pop_back();
        */
        
        
    }
}
//////////////////////


//////////////////////output the flow map as txt
void writeFlow_map(const Mat& flow_x, const Mat& flow_y){
    FILE* fp1,*fp2;
    fp1 = fopen("/home/alan/ApplyEyeMakeup/result/x.txt", "w");
    fp2 = fopen("/home/alan/ApplyEyeMakeup/result/y.txt", "w");
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            fprintf(fp1,"%f",x);
            if(j== flow_y.cols-1)
                fprintf(fp1,"\n");
            else
                fprintf(fp1," ");

            fprintf(fp2,"%f",y);
            if(j== flow_y.cols-1)
                fprintf(fp2,"\n");
            else
                fprintf(fp2," ");
        }
    }
    fclose(fp1);
    fclose(fp2);
}

void writeFlow_map2(string name_temp, const Mat& flow_y){
    FILE *fp2;
    //fp1 = fopen(name_temp, "w");
    fp2 = fopen(name_temp.c_str(), "w");
    for (int i = 0; i < flow_y.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            //float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            //fprintf(fp1,"%f",x);
            //if(j== flow_y.cols-1)
            //    fprintf(fp1,"\n");
            //else
            //    fprintf(fp1," ");

            fprintf(fp2,"%f",y);
            if(j== flow_y.cols-1)
                fprintf(fp2,"\n");
            else
                fprintf(fp2," ");
        }
    }
    //fclose(fp1);
    fclose(fp2);
}
//////////////////////over