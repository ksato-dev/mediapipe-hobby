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
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// #include <locale.h>
// #include <wchar.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"
#include "mediapipe/examples/desktop/janken_pipeline/janken_judgement.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

// TODO: Convert parameters from local to global, & move parameters to external
// file.
constexpr char kInputStream[] = "input_video";
// constexpr char kOutputStream[] = "output_video";
constexpr char kOutputStream[] = "landmarks";
constexpr char kWindowName[] = "Janken++ (beta version)";

const double kCvWaitkeyEsc = 27;
const double kCvWaitkeySpase = 32;
const double kLimitTimeSec = 20.0;
const int kBufferSize = 23;

const std::vector<std::vector<int>> kConnectionList = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},   {0, 5},   {5, 6},   {6, 7},
    {7, 8},   {5, 9},   {9, 10},  {10, 11}, {11, 12}, {9, 13},  {13, 14},
    {14, 15}, {15, 16}, {13, 17}, {17, 18}, {18, 19}, {19, 20}, {0, 17},
};

std::map<JankenGestureType, cv::Mat> kGestureImageMap;

// ステータスバッファー更新
// TODO: Convert class ---
void InitializeGestureStatusBufferList(
    const int &buffer_size, std::vector<std::deque<int>> *status_buffer_list) {
  for (int i = 0; i < (int)(JankenGestureType::NUM_GESTURES); i++)
    status_buffer_list->push_back(std::deque<int>(buffer_size, 0));
}

void UpdateGestureStatusBufferList(
    const std::vector<int> &new_status_list,
    std::vector<std::deque<int>> *status_buffer_list) {
  const int buffer_size = status_buffer_list->at(0).size();
  for (int i = 0; i < status_buffer_list->size(); i++) {
    status_buffer_list->at(i).pop_front();
    status_buffer_list->at(i).push_back(new_status_list[i]);
  }
}

void CalculateStatistics(const std::vector<std::deque<int>> &status_buffer_list,
                         std::vector<float> *result_list) {
  const int buffer_size = status_buffer_list.at(0).size();
  for (int i = 0; i < status_buffer_list.size(); i++) {
    auto &status_buffer = status_buffer_list.at(i);
    float sum_status = 0.0;
    for (int j = 0; j < status_buffer.size(); j++) {
      sum_status += status_buffer.at(j);
    }
    const float avg_status = sum_status / (float)buffer_size;
    result_list->push_back(avg_status);
  }
}
// --- Convert class

// 白画像を作る関数
void CreateWhiteImage(const cv::Size &size, cv::Mat *output_image) {
  *output_image = cv::Mat::zeros(size, CV_8UC3);
  int cols = output_image->cols;
  int rows = output_image->rows;
  for (int j = 0; j < rows; j++) {
    for (int i = 0; i < cols; i++) {
      output_image->at<cv::Vec3b>(j, i)[0] = 255;  // 青
      output_image->at<cv::Vec3b>(j, i)[1] = 255;  // 緑
      output_image->at<cv::Vec3b>(j, i)[2] = 255;  // 赤
    }
  }
}

// 画像を画像に貼り付ける関数
// ref: https://kougaku-navi.hatenablog.com/entry/20160108/p1
void Overlap(cv::Mat dst, cv::Mat src, int x, int y, int width, int height) {
  cv::Mat resized_img;
  cv::resize(src, resized_img, cv::Size(width, height));

  if (x >= dst.cols || y >= dst.rows) return;
  int w = (x >= 0) ? std::min(dst.cols - x, resized_img.cols)
                   : std::min(std::max(resized_img.cols + x, 0), dst.cols);
  int h = (y >= 0) ? std::min(dst.rows - y, resized_img.rows)
                   : std::min(std::max(resized_img.rows + y, 0), dst.rows);
  int u = (x >= 0) ? 0 : std::min(-x, resized_img.cols - 1);
  int v = (y >= 0) ? 0 : std::min(-y, resized_img.rows - 1);
  int px = std::max(x, 0);
  int py = std::max(y, 0);

  cv::Mat roi_dst = dst(cv::Rect(px, py, w, h));
  cv::Mat roi_resized = resized_img(cv::Rect(u, v, w, h));
  roi_resized.copyTo(roi_dst);
}

void DrawNodePoints(const mediapipe::NormalizedLandmarkList &landmarks,
                    const cv::Mat &camera_frame_raw,
                    cv::Mat *output_frame_display_right) {
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

void DrawFrameLines(const mediapipe::NormalizedLandmarkList &landmarks,
                    const cv::Mat &camera_frame_raw,
                    cv::Mat *output_frame_display_right) {
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
  kGestureImageMap[JankenGestureType::GU] =
      cv::imread("mediapipe/resources/gu.png");
  kGestureImageMap[JankenGestureType::CHOKI] =
      cv::imread("mediapipe/resources/choki.png");
  kGestureImageMap[JankenGestureType::PA] =
      cv::imread("mediapipe/resources/pa.png");
  // kGestureImageMap[JankenGestureType::HEART] =
  // cv::imread("mediapipe/resources/heart.png");

  int win_cnt = 0;
  std::vector<std::deque<int>> status_buffer_list;
  InitializeGestureStatusBufferList(kBufferSize, &status_buffer_list);

  std::map<ResultType, cv::Mat> kOperationImageMap;
  // なぜか下記が実行できなくてキレそう（キレてる）
  // kOperationMsgList.push_back(std::string("に勝て！"));
  // kOperationMsgList.push_back(std::string("に負けろ！"));
  // kOperationMsgList.push_back(std::string("とあいこ！"));
  // kOperationMsgList.push_back(std::string("aaa"));  //
  // 英語はいける。やっぱ日本語はクソ

  // 英語版
  kOperationImageMap[ResultType::WIN] =
      cv::imread("mediapipe/resources/win_operation.png");
  kOperationImageMap[ResultType::LOSE] =
      cv::imread("mediapipe/resources/loss_operation.png");
  kOperationImageMap[ResultType::DRAW] =
      cv::imread("mediapipe/resources/draw_operation.png");

  const cv::Mat description_image =
      cv::imread("mediapipe/resources/description.png");
  const cv::Mat your_hand_image =
      cv::imread("mediapipe/resources/your_hand.png");

  std::random_device rnd;  // 非決定的な乱数生成器
  std::mt19937_64 mt(
      rnd());  // メルセンヌ・ツイスタの 64 ビット版、引数は初期シード
  std::uniform_int_distribution<> operation_rand_n(
      1, (int)(ResultType::NUM_RESULT_TYPES)-1);  // [1, n] 範囲の一様乱数,
                                                  // UNKNOWN はスキップ
  // std::uniform_int_distribution<> operation_rand_n(
  //     1, 1); // [1, n] 範囲の一様乱数, UNKNOWN はスキップ
  std::uniform_int_distribution<> opposite_gesture_rand_n(
      1, (int)(JankenGestureType::NUM_GESTURES)-1);  // [1, n] 範囲の一様乱数,
                                                     // UNKNOWMN はスキップ

  // std::cout << "called RunMPPGraph" << std::endl;
  mediapipe::CalculatorGraphConfig config;
  Configure(calculator_graph_config_file, &config);

  std::vector<mediapipe::NormalizedLandmarkList> landmarks_list;

  // TODO: Move to global field
  std::vector<std::shared_ptr<AbstractHandGestureEstimator>>
      one_hand_estimator_list = {
          std::make_shared<GuGestureEstimator>(),
          std::make_shared<ChokiGestureEstimator>(),
          std::make_shared<PaGestureEstimator>(),
      };

  // std::vector<std::shared_ptr<AbstractHandGestureEstimator>>
  // two_hands_estimator_list = {
  //   std::make_shared<HeartGestureEstimator>()
  // };

  mediapipe::CalculatorGraph graph;

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

  ResultType operation = ResultType(operation_rand_n(mt));
  JankenGestureType opposite_gesture =
      JankenGestureType(opposite_gesture_rand_n(mt));

  int time_since_resetting = 1000;  // 初期値がゼロだとはじめに〇が出てしまう。
  auto start_time = std::chrono::system_clock::now();

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

    // Extract landmarks when callback. ---
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us)));

    graph.WaitUntilIdle();

    // cv::Mat landmark_image = cv::Mat::zeros(camera_frame_raw.size(),
    // CV_8UC3);
    cv::Mat landmark_image = camera_frame_raw;
    // --- Extract landmarks when callback.

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
    std::vector<int> new_status_list((int)(JankenGestureType::NUM_GESTURES));

    auto current_recognized_type = JankenGestureType::UNKNOWN;
    if (landmarks_list.size() == 0) {
      Overlap(output_frame_display_right, description_image,
              (camera_frame_raw.rows - description_image.cols) / 2,
              (camera_frame_raw.rows - description_image.rows) / 2,
              description_image.cols, description_image.rows);
    } else {
      // else if (landmarks_list.size() == 1) {
      // 片手 -> 両手でもおｋにした。
      cv::Mat gesture_image;
      for (auto &estimator : one_hand_estimator_list) {
        // std::cout << landmarks_list[0].landmark_size() << std::endl;
        estimator->Initialize();
        const JankenGestureType recognized_type =
            estimator->Recognize(landmarks_list);
        if (recognized_type != JankenGestureType::UNKNOWN) {
          gesture_image = kGestureImageMap[recognized_type];
          current_recognized_type = recognized_type;
        }
      }
      if (gesture_image.empty()) {
        gesture_image = cv::Mat::zeros(
            cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);
      } else {
        cv::resize(gesture_image, gesture_image,
                   cv::Size(camera_frame_raw.rows, camera_frame_raw.rows));
      }

      // 背景を黒塗り
      output_frame_display_right = cv::Mat::zeros(
          cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);

      // 背景の上に手のランドマークを描画
      const float resize_ratio =
          ((float)camera_frame_raw.rows /
           landmark_image
               .cols);  // カメラの高さの合わせてランドマーク画像の幅をリサイズするための倍率
      const int resized_landmark_image_width =
          std::roundl(landmark_image.cols * resize_ratio);
      const int resized_landmark_image_height =
          std::roundl(landmark_image.rows * resize_ratio);
      Overlap(output_frame_display_right, landmark_image, 0,
              (camera_frame_raw.rows - resized_landmark_image_height) / 2,
              resized_landmark_image_width, resized_landmark_image_height);
    }
    // else if (landmarks_list.size() == 2) {
    //   // 両手
    //   cv::Mat gesture_image;
    //   for (auto &estimator : two_hands_estimator_list) {
    //     estimator->Initialize();
    //     const GestureType recognized_type =
    //     estimator->Recognize(landmarks_list); if (recognized_type !=
    //     GestureType::UNKNOWN) {
    //       gesture_image = kGestureImageMap[recognized_type];
    //       current_recognized_type = recognized_type;
    //     }
    //   }
    //   if (gesture_image.empty()) {
    //     gesture_image = cv::Mat::zeros(
    //         cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);
    //   }
    //   else {
    //     cv::resize(gesture_image, gesture_image,
    //                cv::Size(camera_frame_raw.rows, camera_frame_raw.rows));
    //   }
    //   Overlap(gesture_image, landmark_image, camera_frame_raw.cols - 300,
    //           camera_frame_raw.rows - 110,
    //           std::roundl(camera_frame_raw.cols * 0.22),
    //           std::roundl(camera_frame_raw.rows * 0.22));
    //   output_frame_display_right = gesture_image;
    // }

    // Write text.
    if (current_recognized_type != JankenGestureType::UNKNOWN) {
      cv::Mat gesture_image = kGestureImageMap[current_recognized_type];
      cv::resize(gesture_image, gesture_image,
                 cv::Size(your_hand_image.rows, your_hand_image.rows));

      cv::Mat overlap_image;
      cv::hconcat(your_hand_image, gesture_image, overlap_image);

      // 全体の横幅がカメラフレームの縦幅と同じなので注意。
      Overlap(output_frame_display_right, overlap_image,
              (camera_frame_raw.rows - overlap_image.cols) / 2, 0,
              overlap_image.cols, overlap_image.rows);
    }

    // Update all status-buffer.
    new_status_list[(int)(current_recognized_type)]++;
    // for (auto &new_status : new_status_list)
    //   std::cout << new_status << " ";
    // std::cout << std::endl;

    cv::Mat output_frame_display_left = cv::Mat::zeros(
        cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);

    if (time_since_resetting < 10) {
      // 合否表示注はバッファをクリアしておく。
      status_buffer_list = std::vector<std::deque<int>>();
      InitializeGestureStatusBufferList(kBufferSize, &status_buffer_list);

      cv::circle(
          output_frame_display_left,
          cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2), 100,
          cv::Scalar(0, 255, 0), 5, cv::LINE_4);
      time_since_resetting++;
    } else {
      // 合否結果を表示しているときは更新しない。
      UpdateGestureStatusBufferList(new_status_list, &status_buffer_list);

      std::vector<float> result_list;
      CalculateStatistics(status_buffer_list, &result_list);
      JankenGestureType candidate_of_gesture_type = JankenGestureType::UNKNOWN;
      float max_score = 0;
      for (int i = 0; i < result_list.size(); i++) {
        if (max_score < result_list[i]) {
          candidate_of_gesture_type = JankenGestureType(i);
          max_score = result_list[i];
        }
      }
      // std::cout << "GestureType: " << (int)(candidate_of_gesture_type) <<
      // std::endl;

      // Judgement
      const ResultType current_result_type = JankenJudgement::JudgeNormalJanken(
          candidate_of_gesture_type, opposite_gesture);

      // std::cout << "Current result, Your gesture, Opposite gesture,
      // Operation: "
      //           << (int)(current_result_type) << ", "
      //           << (int)(candidate_of_gesture_type) << ", "
      //           << (int)(opposite_gesture) << ", " << (int)(operation)
      //           << std::endl;

      // スコアを更新するタイミングでは次のお題を表示しない。
      const bool flag_for_update = (current_result_type == operation);
      if (flag_for_update && 0.5 < max_score) {
        win_cnt++;

        // 相手の次の手は今のと重複しないようにする。
        // JankenGestureType pre_oppo_gesture = candidate_of_gesture_type;

        JankenGestureType next_correct_gesture_type = candidate_of_gesture_type;
        while (candidate_of_gesture_type == next_correct_gesture_type) {
          JankenGestureType next_opposite_gesture =
              JankenGestureType(opposite_gesture_rand_n(mt));
          ResultType next_operation = ResultType(operation_rand_n(mt));
          ResultType next_result_type = JankenJudgement::JudgeNormalJanken(
              candidate_of_gesture_type, next_opposite_gesture);

          const bool next_flag_for_update =
              (next_result_type == next_operation);

          if (!next_flag_for_update) {
            operation = next_operation;
            opposite_gesture = next_opposite_gesture;
            break;
          }
        }

      // ResultType next_result_type = current_result_type;
      // while (candidate_of_gesture_type == opposite_gesture &&
      //        current_result_type == next_result_type) {
      //   opposite_gesture = JankenGestureType(opposite_gesture_rand_n(mt));
      //   next_result_type = JankenJudgement::JudgeNormalJanken(
      //       candidate_of_gesture_type, opposite_gesture);
      // }

      time_since_resetting = 0;
    }
    else {
      // CreateWhiteImage(cv::Size(camera_frame_raw.rows,
      // camera_frame_raw.rows),
      //                  &output_frame_display_left);
      output_frame_display_left = kGestureImageMap[opposite_gesture];
      cv::resize(output_frame_display_left, output_frame_display_left,
                 cv::Size(camera_frame_raw.rows, camera_frame_raw.rows));
      auto &ope_image = kOperationImageMap[operation];

      // 全体の横幅がカメラフレームの縦幅と同じなので注意。
      Overlap(output_frame_display_left, ope_image,
              (camera_frame_raw.rows - ope_image.cols) / 2, 0, ope_image.cols,
              ope_image.rows);
    }
    landmarks_list = std::vector<mediapipe::NormalizedLandmarkList>();  // reset
  }

  // delay
  // std::this_thread::sleep_for(std::chrono::milliseconds(500));

  const auto current_time = std::chrono::system_clock::now();
  const double time_sec =
      (double)(std::chrono::duration_cast<std::chrono::microseconds>(
                   current_time - start_time)
                   .count() /
               1000.0) /
      1000.0;
  const double time_left_sec = kLimitTimeSec - std::round(time_sec * 10) / 10;
  // std::cout << time_sec << std::endl;
  // std::cout << kLimitTimeSec << std::endl;
  std::stringstream ss;
  ss << std::setprecision(4) << time_left_sec;

  cv::putText(
      output_frame_display_left, std::string("Limit: ") + ss.str(),
      cv::Point(camera_frame_raw.rows - 170, camera_frame_raw.rows - 20), 2,
      0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_4);

  // display score
  cv::putText(output_frame_display_left,
              std::string("Score: ") + std::to_string(win_cnt),
              cv::Point(20, camera_frame_raw.rows - 20), 2, 0.8,
              cv::Scalar(0, 255, 0), 2, cv::LINE_4);

  // cv::circle(*output_frame_display_right, cv::Point(x, y), 2, cv::Scalar(0,
  // 0, 255), 4,
  //            cv::LINE_4);
  // --- 左の表示

  cv::Mat output_frame_display;
  cv::hconcat(output_frame_display_left, output_frame_display_right,
              output_frame_display);
  // --- PostProcess

  // cv::cvtColor(output_frame_display_right, output_frame_display_right,
  // cv::COLOR_RGB2BGR);
  cv::imshow(kWindowName, output_frame_display);
  // Press any key to exit.

  if (time_left_sec <= 0) {
    const cv::Mat result_image =
        cv::Mat::zeros(output_frame_display_left.size(), CV_8UC3);
    cv::putText(
        result_image, std::string("Your Score: ") + std::to_string(win_cnt),
        // cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2),
        cv::Point(30, camera_frame_raw.rows / 2 - 35), 2, 1.5,
        cv::Scalar(0, 255, 0), 2, cv::LINE_4);
    cv::putText(
        result_image, std::string("Restart: <Space>"),
        // cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2),
        cv::Point(30, camera_frame_raw.rows / 2 + 20), 2, 1.5,
        cv::Scalar(0, 255, 0), 2, cv::LINE_4);
    cv::putText(
        result_image, std::string("Finish: <ESC>"),
        // cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2),
        cv::Point(30, camera_frame_raw.rows / 2 + 75), 2, 1.5,
        cv::Scalar(0, 255, 0), 2, cv::LINE_4);

    const cv::Mat instruction_image =
        cv::Mat::zeros(output_frame_display_left.size(), CV_8UC3);
    cv::hconcat(result_image, instruction_image, output_frame_display);
    cv::imshow(kWindowName, output_frame_display);
    const int pressed_key = cv::waitKey(0);
    if (pressed_key == kCvWaitkeyEsc) {
      // terminate
      grab_frames = false;
    } else if (pressed_key == kCvWaitkeySpase) {
      // restart
      start_time = std::chrono::system_clock::now();
      win_cnt = 0;
      continue;
    }
  } else {
    const int pressed_key = cv::waitKey(1);
    if (pressed_key == kCvWaitkeyEsc) {
      // terminate
      grab_frames = false;
    }
  }
}

MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
return graph.WaitUntilDone();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  const std::string calculator_graph_config_file =
      "mediapipe/graphs/janken_pipeline/hand_tracking_desktop_live.pbtxt";
  // const cv::Mat camera_frame_raw =
  // cv::imread("C:\\resource\\image\\index.jpg");
  std::vector<std::vector<cv::Point2i>> *left_eye_landmarks_list;
  std::vector<std::vector<cv::Point2i>> *right_eye_landmarks_list;

  absl::Status run_status =
      RunMPPGraph(calculator_graph_config_file, left_eye_landmarks_list,
                  right_eye_landmarks_list);
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
