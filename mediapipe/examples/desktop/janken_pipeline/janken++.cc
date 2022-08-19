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

constexpr char kInputStream[] = "input_video";
// constexpr char kOutputStream[] = "output_video";
constexpr char kOutputStream[] = "landmarks";
constexpr char kWindowName[] = "Janken++";

// mediapipe::CalculatorGraphConfig config;
// mediapipe::CalculatorGraph graph;
const std::vector<int> kLeftIdxList = {362, 398, 382, 384, 381, 385, 380, 386,
                                       374, 387, 373, 388, 390, 466, 249, 263};
const std::vector<int> kRightIdxList = {133, 173, 155, 157, 154, 158, 153, 159,
                                        145, 160, 144, 161, 163, 246, 7,   33};

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
  // std::cout << "called RunMPPGraph" << std::endl;
  mediapipe::CalculatorGraphConfig config;
  Configure(calculator_graph_config_file, &config);

  std::vector<mediapipe::NormalizedLandmarkList> landmarks_list;

  static mediapipe::CalculatorGraph graph;

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

  // std::cout << "Start grabbing and processing frames." << std::endl;

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
    // std::cout << "exec once." << std::endl;
    // continue;
    // cv::Mat camera_frame_raw = cv::imread(camera_frame_raw_path);
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

    cv::Mat output_frame_mat = cv::Mat::zeros(camera_frame_raw.size(), CV_8UC3);
    // camera_frame_raw.copyTo(output_frame_mat);

    for (int i = 0; i < landmarks_list.size(); i++) {
      mediapipe::NormalizedLandmarkList &landmarks = landmarks_list[i];
      // std::cout << "num_of_landmarks:" << landmarks.landmark_size()
      //           << std::endl;

      // const int num_refined_landmarks = 478;

      for (int j = 0; j < landmarks.landmark_size(); j++) {
        auto &landmark = landmarks.landmark(j);
        int x = int(std::round(landmark.x() * camera_frame_raw.cols));
        int y = int(std::round(landmark.y() * camera_frame_raw.rows));
        // std::cout << "x, y = " << x << ", " << y
        //           << std::endl;
        cv::circle(output_frame_mat, cv::Point(x, y), 2, cv::Scalar(0,255,0), 4, cv::LINE_4);
      }

      break;  // 片方の手だけ利用する。
    }
    landmarks_list = std::vector<mediapipe::NormalizedLandmarkList>();  // reset

    // cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    cv::imshow(kWindowName, output_frame_mat);
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
