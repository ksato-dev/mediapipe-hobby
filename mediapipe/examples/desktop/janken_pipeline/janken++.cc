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
// #include
// "mediapipe/examples/desktop/janken_pipeline/status_buffer_processor.h"
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

const int kCvWaitkeyEsc = 27;
const int kCvWaitkeySpase = 32;
const double kLimitTimeSec = 20.0;

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
  int win_cnt = 0;
  std::vector<StatusBuffer> status_buffer_list;
  StatusBufferProcessor::Initialize(&status_buffer_list);

  // std::cout << "called RunMPPGraph" << std::endl;
  mediapipe::CalculatorGraphConfig config;
  Configure(calculator_graph_config_file, &config);

  std::vector<mediapipe::NormalizedLandmarkList> landmarks_list;

  // TODO: Move to global field
  // std::vector<std::shared_ptr<AbstractHandGestureEstimator>>
  // two_hands_estimator_list = {
  //   std::make_shared<HeartGestureEstimator>()
  // };

  cv::VideoCapture capture;
  // 空いてるカメラ探しに行く処理 ---
  const int limit_cams = 10;
  for (int i = 0; i < limit_cams; i++) {
    capture.open(i);
    cv::Mat temp;
    capture >> temp;
    if (temp.empty()) {
      std::cout << "This camera is used, maybe." << std::endl;
      capture.release();
    } else {
      break;
    }
  }
  // --- 空いてるカメラ探しに行く処理

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

  bool grab_frames = true;

  // ResultType operation = ResultType(operation_rand_n(mt));
  // GestureType opposite_gesture =
  //     GestureType(opposite_gesture_rand_n(mt));

  int num_frames_since_resetting =
      1000;  // 初期値がゼロだとはじめに〇が出てしまう。
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
