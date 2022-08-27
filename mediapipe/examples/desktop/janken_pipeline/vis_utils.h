
#pragma once
#include <opencv2/opencv.hpp>

#include "mediapipe/framework/formats/landmark.pb.h"

class VisUtility {
 public:
  static void CreateWhiteImage(const cv::Size &size, cv::Mat *output_image);
  static void Overlap(cv::Mat dst, cv::Mat src, int x, int y, int width,
                      int height);
  static void DrawNodePoints(const mediapipe::NormalizedLandmarkList &landmarks,
                             const cv::Mat &camera_frame_raw,
                             cv::Mat *output_frame_display_right);
  static void DrawFrameLines(const mediapipe::NormalizedLandmarkList &landmarks,
                             const cv::Mat &camera_frame_raw,
                             cv::Mat *output_frame_display_right);
  static void BlurImage(const cv::Size &gauss_kernel, const cv::Mat &src_image,
                        cv::Mat *dst_image);

 private:
  static const std::vector<std::vector<int>> connection_list_;
};
