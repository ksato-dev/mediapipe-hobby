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
// #include "mediapipe/examples/desktop/janken_pipeline/janken_judgement.h"
// #include "mediapipe/examples/desktop/janken_pipeline/status_buffer_processor.h"
// #include "mediapipe/examples/desktop/janken_pipeline/vis_utils.h"
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

  // std::random_device rnd;  // 非決定的な乱数生成器
  // std::mt19937_64 mt(
  //     rnd());  // メルセンヌ・ツイスタの 64 ビット版、引数は初期シード
  // std::uniform_int_distribution<> operation_rand_n(
  //     1, (int)(ResultType::NUM_RESULT_TYPES)-1);  // [1, n] 範囲の一様乱数,
  //                                                 // UNKNOWN はスキップ
  // // std::uniform_int_distribution<> operation_rand_n(
  // //     1, 1); // [1, n] 範囲の一様乱数, UNKNOWN はスキップ
  // std::uniform_int_distribution<> opposite_gesture_rand_n(
  //     1,
  //     (int)(JankenGestureType::NUM_GESTURES)-2);  // [1, n-1] 範囲の一様乱数,
  //                                                 // UNKNOWMN, HEART はスキップ

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

  // ResultType operation = ResultType(operation_rand_n(mt));
  // JankenGestureType opposite_gesture =
  //     JankenGestureType(opposite_gesture_rand_n(mt));

  int num_frames_since_resetting = 1000;  // 初期値がゼロだとはじめに〇が出てしまう。
  auto start_time = std::chrono::system_clock::now();

  auto &post_processor = PostProcessor();

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

    cv::Mat output_image;
    post_processor.Execute(camera_frame_raw, &landmarks_list, &output_image,
                           &start_time, &grab_frames);
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
