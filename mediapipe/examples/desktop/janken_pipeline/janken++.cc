// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>
#include <random>
#include <fstream>
#include <locale.h>
#include <wchar.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"

constexpr char kInputStream[] = "input_video";
// constexpr char kOutputStream[] = "output_video";
constexpr char kOutputStream[] = "landmarks";
constexpr char kWindowName[] = "Janken++";

const std::vector<std::vector<int>> kConnectionList = {
  {0, 1}, {1, 2}, {2, 3}, {3, 4},
  {0, 5}, {5, 6}, {6, 7}, {7, 8},
  {5, 9},
  {9, 10}, {10, 11}, {11, 12},
  {9, 13},
  {13, 14}, {14, 15}, {15, 16},
  {13, 17},
  {17, 18}, {18, 19}, {19, 20},
  {0, 17},
};

// 白画像を作る関数
void CreateWhiteImage(const cv::Size &size, cv::Mat *output_image) {
  *output_image = cv::Mat::zeros(size, CV_8UC3);
  int cols = output_image->cols;
  int rows = output_image->rows;
  for (int j = 0; j < rows; j++) {
    for (int i = 0; i < cols; i++) {
      output_image->at<cv::Vec3b>(j, i)[0] = 255; // 青
      output_image->at<cv::Vec3b>(j, i)[1] = 255; // 緑
      output_image->at<cv::Vec3b>(j, i)[2] = 255; // 赤
    }
  }
}

// 画像を画像に貼り付ける関数
// ref: https://kougaku-navi.hatenablog.com/entry/20160108/p1
void Overlap(cv::Mat dst, cv::Mat src, int x, int y, int width, int height) {
	cv::Mat resized_img;
	cv::resize(src, resized_img, cv::Size(width, height));

	if (x >= dst.cols || y >= dst.rows) return;
	int w = (x >= 0) ? std::min(dst.cols - x, resized_img.cols) : std::min(std::max(resized_img.cols + x, 0), dst.cols);
	int h = (y >= 0) ? std::min(dst.rows - y, resized_img.rows) : std::min(std::max(resized_img.rows + y, 0), dst.rows);
	int u = (x >= 0) ? 0 : std::min(-x, resized_img.cols - 1);
	int v = (y >= 0) ? 0 : std::min(-y, resized_img.rows - 1);
	int px = std::max(x, 0);
	int py = std::max(y, 0);

	cv::Mat roi_dst = dst(cv::Rect(px, py, w, h));
	cv::Mat roi_resized = resized_img(cv::Rect(u, v, w, h));
	roi_resized.copyTo(roi_dst);
}

void DrawNodePoints(const mediapipe::NormalizedLandmarkList &landmarks,
                    const cv::Mat &camera_frame_raw, cv::Mat *output_frame_display_right) {
  for (int j = 0; j < landmarks.landmark_size(); j++) {
    auto &landmark = landmarks.landmark(j);
    int x = int(std::round(landmark.x() * camera_frame_raw.cols));
    int y = int(std::round(landmark.y() * camera_frame_raw.rows));
    // std::cout << "x, y = " << x << ", " << y
    //           << std::endl;
    cv::circle(*output_frame_display_right, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), 4,
               cv::LINE_4);
  }
}

void DrawFrameLines(const mediapipe::NormalizedLandmarkList &landmarks,
                    const cv::Mat &camera_frame_raw, cv::Mat *output_frame_display_right) {
  // index
  for (auto &conn : kConnectionList) {
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

absl::Status Configure(const std::string &calculator_graph_config_file,
                       mediapipe::CalculatorGraphConfig *config) {
  // std::cout << "called Configure" << std::endl;
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      calculator_graph_config_file, &calculator_graph_config_contents));
  // std::cout << "Get calculator graph config contents: "
  //           << calculator_graph_config_contents << std::endl;
  mediapipe::CalculatorGraphConfig temp_config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);
  *config = temp_config;
  return absl::OkStatus();
}

absl::Status RunMPPGraph(
    const std::string &calculator_graph_config_file,
    // const cv::Mat &camera_frame_raw,
    std::vector<std::vector<cv::Point2i>> *left_eye_landmarks_list,
    std::vector<std::vector<cv::Point2i>> *right_eye_landmarks_list) {

  std::vector<std::string> kThemeMsgList(0);
#if 1
  // なぜか下記が実行できなくてキレそう（キレてる）
  // kThemeMsgList.push_back(std::string("に勝て！"));
  // kThemeMsgList.push_back(std::string("に負けろ！"));
  // kThemeMsgList.push_back(std::string("とあいこ！"));
  // kThemeMsgList.push_back(std::string("aaa"));  // 英語はいける。やっぱ日本語はクソ

  // 英語版
  kThemeMsgList.push_back(std::string(": Win this gesture!"));
  kThemeMsgList.push_back(std::string(": Lose this gesture!"));
  kThemeMsgList.push_back(std::string(": Draw this gesture!"));
#else
  // ref: https://code-database.com/knowledges/123
  std::ifstream ifs;
  ifs.open("mediapipe/examples/desktop/janken_pipeline/msg_list.txt", std::ios::in);
  std::string reading_line_buffer;
  setlocale(LC_CTYPE, "ja_JP.UTF-8");

  while(std::getline(ifs, reading_line_buffer)){
      size_t capacity = strlen(reading_line_buffer.c_str()) + 1;
      wchar_t *str2 = (wchar_t *)malloc(sizeof(wchar_t) * capacity);

      // char -> wchar_tの変換
      int result = mbstowcs(str2, reading_line_buffer.c_str(), capacity);

      std::cout << str2 << std::endl;
      // kThemeMsgList.push_back(reading_line_buffer);
  }
#endif

  std::random_device rnd; // 非決定的な乱数生成器
  std::mt19937_64 mt(rnd()); // メルセンヌ・ツイスタの32ビット版、引数は初期シード
  std::uniform_int_distribution<> rand_n(
      0, kThemeMsgList.size() - 1); // [0, n] 範囲の一様乱数

  // std::cout << "called RunMPPGraph" << std::endl;
  mediapipe::CalculatorGraphConfig config;
  Configure(calculator_graph_config_file, &config);

  std::vector<mediapipe::NormalizedLandmarkList> landmarks_list;

  std::vector<std::shared_ptr<AbstractHandGestureEstimator>> one_hand_estimator_list = {
    std::make_shared<GuGestureEstimator>(),
    std::make_shared<ChokiGestureEstimator>(),
    std::make_shared<PaGestureEstimator>(),
  };

  std::vector<std::shared_ptr<AbstractHandGestureEstimator>> two_hands_estimator_list = {
    std::make_shared<HeartGestureEstimator>()
  };

  mediapipe::CalculatorGraph graph;

  // 初回だけ初期化
  if (!graph.HasInputStream("input_video")) {
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    graph.ObserveOutputStream(
        kOutputStream,
        [&landmarks_list](
            const mediapipe::Packet &packet) -> ::mediapipe::Status {
          landmarks_list =
              packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
          // Do something.
          return mediapipe::OkStatus();
        });
  }
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  cv::VideoCapture capture;
  capture.open(0);

  bool grab_frames = true;

  while (grab_frames) {
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;

    if (camera_frame_raw.empty()) {
      // absl::AbortedError(absl::string_view("Image is empty."));
      MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
      return graph.WaitUntilDone();
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    // cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    // cv::imwrite("/tmp/temp.png", camera_frame);

    // Wrap Mat into an ImageFrame.
    // std::cout << "Wrap Mat into an ImageFrame." << std::endl;
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);

    // コピーしないと機能しない。
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    // std::cout << "Send image packet into the graph." << std::endl;
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us)));

    graph.WaitUntilIdle();

    cv::Mat landmark_image = cv::Mat::zeros(camera_frame_raw.size(), CV_8UC3);

    // PostProcess ---
    for (int i = 0; i < landmarks_list.size(); i++) {
      mediapipe::NormalizedLandmarkList &landmarks = landmarks_list[i];
      // draw frame-lines
      DrawFrameLines(landmarks, camera_frame_raw, &landmark_image);

      // draw node-points
      DrawNodePoints(landmarks, camera_frame_raw, &landmark_image);
    }

    cv::Mat output_frame_display_right;
    output_frame_display_right = cv::Mat::zeros(
        cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);

    // TODO: refactor
    GestureType pre_recognized_type = GestureType::UNKNOWN;
    if (landmarks_list.size() == 1) {
      // 片手
      cv::Mat gesture_image;
      for (auto &estimator : one_hand_estimator_list) {
        // std::cout << landmarks_list[0].landmark_size() << std::endl;
        estimator->Initialize();
        const GestureType recognized_type = estimator->Recognize(landmarks_list);
        if (recognized_type != GestureType::UNKNOWN) {
          if (recognized_type == GestureType::GU)
            gesture_image = cv::imread("mediapipe/resources/gu.png");
          else if (recognized_type == GestureType::CHOKI)
            gesture_image = cv::imread("mediapipe/resources/choki.png");
          else if (recognized_type == GestureType::PA)
            gesture_image = cv::imread("mediapipe/resources/pa.png");

          pre_recognized_type = recognized_type;
        }
      }
      if (gesture_image.empty()) {
        gesture_image = cv::Mat::zeros(
            cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);
      }
      else {
        cv::resize(gesture_image, gesture_image,
                   cv::Size(camera_frame_raw.rows, camera_frame_raw.rows));
      }
      Overlap(gesture_image, landmark_image, -3,
              camera_frame_raw.rows - 105,
              std::roundl(camera_frame_raw.cols * 0.2),
              std::roundl(camera_frame_raw.rows * 0.2));
      output_frame_display_right = gesture_image;
    }
    else if (landmarks_list.size() == 2) {
      // 両手
      cv::Mat gesture_image;
      for (auto &estimator : two_hands_estimator_list) {
        estimator->Initialize();
        const GestureType recognized_type = estimator->Recognize(landmarks_list);
        if (recognized_type != GestureType::UNKNOWN) {
          if (recognized_type == GestureType::HEART)
            gesture_image = cv::imread("mediapipe/resources/heart.png");

          pre_recognized_type = recognized_type;
        }
      }
      if (gesture_image.empty()) {
        gesture_image = cv::Mat::zeros(
            cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);
      }
      else {
        cv::resize(gesture_image, gesture_image,
                   cv::Size(camera_frame_raw.rows, camera_frame_raw.rows));
      }
      Overlap(gesture_image, landmark_image, -3,
              camera_frame_raw.rows - 105,
              std::roundl(camera_frame_raw.cols * 0.2),
              std::roundl(camera_frame_raw.rows * 0.2));
      output_frame_display_right = gesture_image;
    }
    landmarks_list = std::vector<mediapipe::NormalizedLandmarkList>();  // reset

    cv::Mat output_frame_display;

    // 左の表示 ---
    // cv::Mat output_frame_display_left = cv::Mat::zeros(camera_frame_raw.size(), CV_8UC3);
    // cv::Mat output_frame_display_left = cv::Mat::zeros(
    //     cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);
    // int cols = output_frame_display_left.cols;
    // int rows = output_frame_display_left.rows;
    // for (int j = 0; j < rows; j++) {
    //     for (int i = 0; i < cols; i++) {
    //         output_frame_display_left.at<cv::Vec3b>(j, i)[0] = 255; //青
    //         output_frame_display_left.at<cv::Vec3b>(j, i)[1] = 255; //緑
    //         output_frame_display_left.at<cv::Vec3b>(j, i)[2] = 255; //赤
    //     }
    // }

    cv::Mat output_frame_display_left;
    // CreateWhiteImage(cv::Size(camera_frame_raw.rows, camera_frame_raw.rows),
    //                  &output_frame_display_left);
    output_frame_display_left = cv::Mat::zeros(
        cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);

    cv::putText(output_frame_display_left, kThemeMsgList[rand_n(mt)], cv::Point(10, 30), 2, 1.0,
                cv::Scalar(0, 255, 0), 2, cv::LINE_4);
    // --- 左の表示

    cv::hconcat(output_frame_display_left, output_frame_display_right, output_frame_display);
    // --- PostProcess

    // cv::cvtColor(output_frame_display_right, output_frame_display_right, cv::COLOR_RGB2BGR);
    cv::imshow(kWindowName, output_frame_display);
    // Press any key to exit.
    const int pressed_key = cv::waitKey(1);
    if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
  }

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  const std::string calculator_graph_config_file = "mediapipe/graphs/janken_pipeline/hand_tracking_desktop_live.pbtxt";
  // const cv::Mat camera_frame_raw = cv::imread("C:\\resource\\image\\index.jpg");
  std::vector<std::vector<cv::Point2i>> *left_eye_landmarks_list;
  std::vector<std::vector<cv::Point2i>> *right_eye_landmarks_list;

  absl::Status run_status =
      RunMPPGraph(calculator_graph_config_file,
                  left_eye_landmarks_list, right_eye_landmarks_list);
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
