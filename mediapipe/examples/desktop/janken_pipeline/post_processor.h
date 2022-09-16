#pragma once
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"
#include "mediapipe/examples/desktop/janken_pipeline/janken_judgement.h"
#include "mediapipe/examples/desktop/janken_pipeline/status_buffer_processor.h"
#include "mediapipe/framework/formats/landmark.pb.h"


class PostProcessor {
 public:
  PostProcessor();
  void Execute(const cv::Mat &camera_frame_raw,
               std::vector<mediapipe::NormalizedLandmarkList> *landmarks_list,
               cv::Mat *output_frame_for_display,
               std::chrono::system_clock::time_point *start_time_,
               bool *grab_frames);

 private:
  std::string k_window_name_;
  double k_limit_time_sec_;
  int k_cv_waitkey_esc_;
  int k_cv_waitkey_spase_;

  cv::Mat k_description_image_;
  cv::Mat k_your_hand_image_;

  std::vector<std::shared_ptr<AbstractHandGestureEstimator>>
      k_hand_estimator_list_;

  std::map<GestureType, cv::Mat> k_gesture_image_map_;
  std::map<ResultType, cv::Mat> k_janken_operation_image_map_;
  std::map<GestureType, cv::Mat> k_imitation_operation_image_map_;
  std::map<GestureType, cv::Mat> k_imitation_operation_image_map_;

  double k_th_score_;
  std::mt19937_64
      k_mt_;  // メルセンヌ・ツイスタの 64 ビット版、引数は初期シード
  std::uniform_int_distribution<>
      k_janken_operation_rand_n_;  // [1, n] 範囲の一様乱数, UNKNOWN はスキップ
  std::uniform_int_distribution<> k_opposite_gesture_rand_n_;

  ResultType operation_;
  GestureType opposite_gesture_;
  RuleType rule_;

  // GestureType をひっかけてルールタイプを取り出す。
  std::map<GestureType, RuleType> k_gesture_and_rule_map_;

  std::vector<StatusBuffer> status_buffer_list_;
  int time_since_resetting_;  // 初期値がゼロだとはじめに〇が出てしまう。
  int win_cnt_;
  unsigned long long num_frames_since_resetting_ = 0;
};
