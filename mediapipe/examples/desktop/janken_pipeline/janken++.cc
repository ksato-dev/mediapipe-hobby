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
#include "mediapipe/examples/desktop/janken_pipeline/status_buffer_processor.h"
#include "mediapipe/examples/desktop/janken_pipeline/vis_utils.h"
#include "mediapipe/examples/desktop/janken_pipeline/post_processor.h"

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

const int kCvWaitkeyEsc = 27;
const int kCvWaitkeySpase = 32;
const double kLimitTimeSec = 20.0;
const int kBufferSize = 23;

std::map<JankenGestureType, cv::Mat> kGestureImageMap;

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

absl::Status RunMPPGraph(const std::string &calculator_graph_config_file) {
  kGestureImageMap[JankenGestureType::GU] =
      cv::imread("mediapipe/resources/gu.png");
  kGestureImageMap[JankenGestureType::CHOKI] =
      cv::imread("mediapipe/resources/choki.png");
  kGestureImageMap[JankenGestureType::PA] =
      cv::imread("mediapipe/resources/pa.png");
  kGestureImageMap[JankenGestureType::HEART] =
      cv::imread("mediapipe/resources/heart.png");

  int win_cnt = 0;
  std::vector<StatusBuffer> status_buffer_list;
  StatusBufferProcessor::Initialize(kBufferSize, &status_buffer_list);

  std::map<ResultType, cv::Mat> kOperationImageMap;
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
      1,
      (int)(JankenGestureType::NUM_GESTURES)-2);  // [1, n-1] 範囲の一様乱数,
                                                  // UNKNOWMN, HEART はスキップ

  // std::cout << "called RunMPPGraph" << std::endl;
  mediapipe::CalculatorGraphConfig config;
  Configure(calculator_graph_config_file, &config);

  std::vector<mediapipe::NormalizedLandmarkList> landmarks_list;

  // TODO: Move to global field
  std::vector<std::shared_ptr<AbstractHandGestureEstimator>>
      hand_estimator_list = {
          std::make_shared<GuGestureEstimator>(),
          std::make_shared<ChokiGestureEstimator>(),
          std::make_shared<PaGestureEstimator>(),
          std::make_shared<HeartGestureEstimator>(),
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
  capture.open(
      0);  // TODO:
           // 複数カメラ接続している時に自動空いているカメラを使いに行かせる。
  // TODO: camera size を固定にするか自動にするか決める。

  bool grab_frames = true;

  ResultType operation = ResultType(operation_rand_n(mt));
  JankenGestureType opposite_gesture =
      JankenGestureType(opposite_gesture_rand_n(mt));

  int num_frames_since_resetting = 1000;  // 初期値がゼロだとはじめに〇が出てしまう。
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
    // --- Extract landmarks when callback.

    // PostProcess ---
    cv::Mat landmark_image;
    VisUtility::BlurImage(cv::Size(11, 11), camera_frame_raw, &landmark_image);

    for (int i = 0; i < landmarks_list.size(); i++) {
      mediapipe::NormalizedLandmarkList &landmarks = landmarks_list[i];
      // draw frame-lines
      VisUtility::DrawFrameLines(landmarks, landmark_image, &landmark_image);

      // draw node-points
      VisUtility::DrawNodePoints(landmarks, landmark_image, &landmark_image);
    }

    cv::Mat output_frame_display_right;
    output_frame_display_right = cv::Mat::zeros(
        cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);

    std::vector<bool> new_status_list((int)(JankenGestureType::NUM_GESTURES));

    auto current_recognized_type = JankenGestureType::UNKNOWN;
    if (landmarks_list.size() == 0) {
      VisUtility::Overlap(output_frame_display_right, description_image,
                          (camera_frame_raw.rows - description_image.cols) / 2,
                          (camera_frame_raw.rows - description_image.rows) / 2,
                          description_image.cols, description_image.rows);
    } else {
      // else if (landmarks_list.size() == 1) {
      // 片手 -> 両手でもおｋにした。
      cv::Mat gesture_image;
      for (auto &estimator : hand_estimator_list) {
        // std::cout << landmarks_list[0].landmark_size() << std::endl;
        estimator->Initialize();
        const JankenGestureType temp_recognized_type =
            estimator->Recognize(landmarks_list);
        if (temp_recognized_type != JankenGestureType::UNKNOWN) {
          gesture_image = kGestureImageMap[temp_recognized_type];
          current_recognized_type = temp_recognized_type;
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
      // カメラの高さの合わせてランドマーク画像の幅をリサイズするための倍率
      const float resize_ratio =
          ((float)camera_frame_raw.rows / landmark_image.cols);
      const int resized_landmark_image_width =
          std::roundl(landmark_image.cols * resize_ratio);
      const int resized_landmark_image_height =
          std::roundl(landmark_image.rows * resize_ratio);
      VisUtility::Overlap(
          output_frame_display_right, landmark_image, 0,
          (camera_frame_raw.rows - resized_landmark_image_height) / 2,
          resized_landmark_image_width, resized_landmark_image_height);
    }

    // Write text.
    if (current_recognized_type != JankenGestureType::UNKNOWN) {
      cv::Mat gesture_image = kGestureImageMap[current_recognized_type];
      cv::resize(gesture_image, gesture_image,
                 cv::Size(your_hand_image.rows, your_hand_image.rows));

      cv::Mat overlap_image;
      cv::hconcat(your_hand_image, gesture_image, overlap_image);

      // 全体の横幅がカメラフレームの縦幅と同じなので注意。
      VisUtility::Overlap(output_frame_display_right, overlap_image,
                          (camera_frame_raw.rows - overlap_image.cols) / 2, 0,
                          overlap_image.cols, overlap_image.rows);
    }
    cv::putText(
        output_frame_display_right, std::string("Finish: <ESC>"),
        cv::Point(camera_frame_raw.rows - 140, camera_frame_raw.rows - 10), 1,
        1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_4);

    // Update all status-buffer.
    new_status_list[(int)(current_recognized_type)] = true;

    cv::Mat output_frame_display_left = cv::Mat::zeros(
        cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);

    if (num_frames_since_resetting < 10) {
      // 合否表示注はバッファをクリアしておく。
      status_buffer_list = std::vector<StatusBuffer>();
      StatusBufferProcessor::Initialize(kBufferSize, &status_buffer_list);

      cv::circle(
          output_frame_display_left,
          cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2), 100,
          cv::Scalar(0, 255, 0), 5, cv::LINE_4);
      num_frames_since_resetting++;
    } else {
      // 合否結果を表示しているときは更新しない。
      StatusBufferProcessor::Update(new_status_list, &status_buffer_list);

      std::vector<float> score_list;
      StatusBufferProcessor::CalculateStatistics(status_buffer_list,
                                                 &score_list);
      JankenGestureType candidate_of_gesture_type = JankenGestureType::UNKNOWN;
      float max_score = 0;
      for (int i = 0; i < score_list.size(); i++) {
        if (max_score < score_list[i]) {
          candidate_of_gesture_type = JankenGestureType(i);
          max_score = score_list[i];
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

        num_frames_since_resetting = 0;
      } else {
        // CreateWhiteImage(cv::Size(camera_frame_raw.rows,
        // camera_frame_raw.rows),
        //                  &output_frame_display_left);
        output_frame_display_left = kGestureImageMap[opposite_gesture];
        cv::resize(output_frame_display_left, output_frame_display_left,
                   cv::Size(camera_frame_raw.rows, camera_frame_raw.rows));
        auto &ope_image = kOperationImageMap[operation];

        // 全体の横幅がカメラフレームの縦幅と同じなので注意。
        VisUtility::Overlap(output_frame_display_left, ope_image,
                            (camera_frame_raw.rows - ope_image.cols) / 2, 0,
                            ope_image.cols, ope_image.rows);
      }
      landmarks_list =
          std::vector<mediapipe::NormalizedLandmarkList>();  // reset
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
        0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_4);

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

  absl::Status run_status = RunMPPGraph(calculator_graph_config_file);

  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
