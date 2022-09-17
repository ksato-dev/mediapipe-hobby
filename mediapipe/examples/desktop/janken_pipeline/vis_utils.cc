
#include "mediapipe/examples/desktop/janken_pipeline/vis_utils.h"

#include <opencv2/opencv.hpp>

const std::vector<std::vector<int>> VisUtility::connection_list_ = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},   {5, 6},   {6, 7},   {7, 8},
    {5, 9},   {9, 10},  {10, 11}, {11, 12}, {9, 13},  {13, 14}, {14, 15},
    {15, 16}, {13, 17}, {17, 18}, {18, 19}, {19, 20}, {0, 17},  {5, 1},
};

// 白画像を作る関数
void VisUtility::CreateAnyColorImage(const cv::Vec3b &color,
                                     const cv::Size &size,
                                     cv::Mat *output_image) {
  *output_image = cv::Mat::zeros(size, CV_8UC3);
  int cols = output_image->cols;
  int rows = output_image->rows;
  for (int j = 0; j < rows; j++) {
    for (int i = 0; i < cols; i++) {
      output_image->at<cv::Vec3b>(j, i)[0] = color[0];  // 青
      output_image->at<cv::Vec3b>(j, i)[1] = color[1];  // 緑
      output_image->at<cv::Vec3b>(j, i)[2] = color[2];  // 赤
    }
  }
}

// 画像を画像に貼り付ける関数
// ref: https://kougaku-navi.hatenablog.com/entry/20160108/p1
void VisUtility::Overlap(cv::Mat dst, cv::Mat src, int x, int y) {
  if (src.channels() == 4) cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);
  if (x >= dst.cols || y >= dst.rows) return;
  int w = (x >= 0) ? std::min(dst.cols - x, src.cols)
                   : std::min(std::max(src.cols + x, 0), dst.cols);
  int h = (y >= 0) ? std::min(dst.rows - y, src.rows)
                   : std::min(std::max(src.rows + y, 0), dst.rows);
  int u = (x >= 0) ? 0 : std::min(-x, src.cols - 1);
  int v = (y >= 0) ? 0 : std::min(-y, src.rows - 1);
  int px = std::max(x, 0);
  int py = std::max(y, 0);

  cv::Mat roi_dst = dst(cv::Rect(px, py, w, h));
  cv::Mat roi_resized = src(cv::Rect(u, v, w, h));
  roi_resized.copyTo(roi_dst);
}

void VisUtility::DrawNodePoints(
    const mediapipe::NormalizedLandmarkList &landmarks,
    const cv::Mat &camera_frame_raw, cv::Mat *output_frame_display_right) {
  for (int j = 0; j < landmarks.landmark_size(); j++) {
    auto &landmark = landmarks.landmark(j);
    int x = int(std::round(landmark.x() * camera_frame_raw.cols));
    int y = int(std::round(landmark.y() * camera_frame_raw.rows));
    // std::cout << "x, y = " << x << ", " << y
    //           << std::endl;
    cv::circle(*output_frame_display_right, cv::Point(x, y), 2,
               cv::Scalar(0, 0, 255), 4, cv::LINE_4);
  }
}

void VisUtility::DrawFrameLines(
    const mediapipe::NormalizedLandmarkList &landmarks,
    const cv::Mat &camera_frame_raw, cv::Mat *output_frame_display_right) {
  // index
  for (auto &conn : connection_list_) {
    auto &landmark1 = landmarks.landmark(conn[0]);
    int x1 = int(std::round(landmark1.x() * camera_frame_raw.cols));
    int y1 = int(std::round(landmark1.y() * camera_frame_raw.rows));

    auto &landmark2 = landmarks.landmark(conn[1]);
    int x2 = int(std::round(landmark2.x() * camera_frame_raw.cols));
    int y2 = int(std::round(landmark2.y() * camera_frame_raw.rows));

    cv::line(*output_frame_display_right, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(0, 255, 0), 4, cv::LINE_4);
  }
}

void VisUtility::BlurImage(const cv::Size &gauss_kernel,
                           const cv::Mat &src_image, cv::Mat *dst_image) {
  cv::Mat processed_image = src_image;
  for (int i = 0; i < 5; i++)
    cv::resize(processed_image, processed_image,
               cv::Size(processed_image.cols / 2, processed_image.rows / 2));

  cv::resize(processed_image, processed_image,
             cv::Size(src_image.cols, src_image.rows));
  cv::GaussianBlur(processed_image, processed_image, gauss_kernel, 1.0, 1.0);
  *dst_image = processed_image;
}

//! cite: https://qiita.com/koyayashi/items/ce620783a6cae726b4c1
void VisUtility::PutTranspPng(cv::Mat &dst, cv::Mat &src, int x, int y) {
  // std::cout << "channels:" << src.channels() << std::endl;
  // if (src.channels() == 4) cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);

  // 下記の処理は_pngがRGBAの4チャンネルMatである必要がある
  if (4 != src.channels()) {
    std::cout << "putTranspPng() invalid input" << std::endl;
    VisUtility::Overlap(dst, src, x, y);
    return;
  }
  // cv::Mat resized_img;
  // cv::resize(src, resized_img, cv::Size(width, height));

  int px = std::max(x, 0);
  int py = std::max(y, 0);

  // ---
  std::vector<cv::Mat> layers;
  cv::split(src, layers);

  // 貼り付ける画像（3チャンネル）
  // cv::Mat resized_img;
  cv::merge(layers.data(), 3, src);
  // copyToに使うmask（1チャンネル）
  cv::Mat mask = layers[3];

  cv::Mat roi_dst = dst(cv::Rect(px, py, src.cols, src.rows));

  src.copyTo(src, mask);
  for (int dst_y = 0; dst_y < dst.rows; dst_y++) {
    cv::Vec3b *dst_img_row = dst.ptr<cv::Vec3b>(dst_y);
    for (int dst_x = 0; dst_x < dst.cols; dst_x++) {
      auto flag_x = (px <= dst_x && dst_x < px + src.cols);
      auto flag_y = (py <= dst_y && dst_y < py + src.rows);
      if (flag_x && flag_y) {
        int mask_px = dst_x - px;
        int mask_py = dst_y - py;
        // std::cout << mask.type() << std::endl;
        // auto mask_value = mask.at<uint8_t>(mask_py, mask_px);
        // uint8_t mask_value = mask(Point(mask_py, mask_px));
        uint8_t *mask_img_row = mask.ptr<uint8_t>(mask_py);
        uint8_t mask_value = mask_img_row[mask_px];
        if (mask_value > 50) {
          // std::cout << (int)mask_value << std::endl;
          cv::Vec3b *src_img_row = src.ptr<cv::Vec3b>(mask_py);
          cv::Vec3b src_pix_data = src_img_row[mask_px];
          // dst_img_row[dst_x] = cv::Vec3b(243, 161, 130);
          dst_img_row[dst_x] = src_pix_data;
          // std::cout << "px, py: " << mask_px << ", " << mask_py << std::endl;
        }
        // std::cout << "px, py: " << mask_px << ", " << mask_py << std::endl;
      }
    }
  }
  // std::cout << "Done PutTransPng." << std::endl;
}