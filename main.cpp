#include <iostream>
#include <vector>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "impl.h"

using namespace std;

int main(int argc, char** argv) {
  cv::Mat image;
  int h, w, ch;
  image =  cv::imread("../image.jpg");
  h = image.rows, w = image.cols, ch = image.channels();
  cv::imshow("原图",image);
  image.convertTo(image, CV_32FC3);
  float *d_img = (float*)image.data;

  vector<float> p = {132,174,105,170,165,167,147,190};
  vector<float> q = {132,170,105,170,165,167,147,190};
  float *d_p = p.data();
  float *d_q = q.data();
  cv::Mat result(h, w, CV_32FC3);
  result = 0;
  float *res = (float*)result.data;
  clock_t time0 = clock();
  mls_affine(d_img, h, w, d_p, d_q, p.size()/2, 1.0, 1.0, res);
  cout <<"time use in MLS is " <<1000*(clock() - time0)/(double)CLOCKS_PER_SEC <<"ms" << endl;
  result.convertTo(result, CV_8UC3);
  cv::imshow("affine mls",result);
  cv::imwrite("mls.jpg",result);
  
  cv::Mat result1(h, w, CV_32FC3);
  result1 = 0;
  float *res1 = (float*)result1.data;
  clock_t time1 = clock();
  mls_affine_inv(d_img, h, w, d_p, d_q, p.size()/2, 1.0, 1.0, res1);
  cout <<"time use in MLS is " <<1000*(clock() - time1)/(double)CLOCKS_PER_SEC <<"ms" << endl;
  result1.convertTo(result1, CV_8UC3);
  cv::imshow("affine mls inv",result1);
  cv::imwrite("mls_inv.jpg",result1);
  cv::waitKey();
  return 0;
}
