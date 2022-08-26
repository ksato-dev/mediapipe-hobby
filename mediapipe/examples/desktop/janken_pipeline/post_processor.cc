
#include "mediapipe/examples/desktop/janken_pipeline/post_processor.h"

#include <random>

#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"
#include "mediapipe/examples/desktop/janken_pipeline/janken_judgement.h"
#include "mediapipe/examples/desktop/janken_pipeline/status_buffer_processor.h"
#include "mediapipe/examples/desktop/janken_pipeline/vis_utils.h"


// TODO: 機能追加：1. ハートのポーズ出題, 2. 呪術廻戦のポーズ認識・出題
PostProcessor::PostProcessor() {
  k_limit_time_sec_ = 20.0;
  k_window_name_ = "Gesture++ (beta version)";
  k_cv_waitkey_esc_ = 27;
  k_cv_waitkey_spase_ = 32;
  k_buffer_size_ = 17;

  k_description_image_ = cv::imread("mediapipe/resources/description.png");
  k_your_hand_image_ = cv::imread("mediapipe/resources/your_hand.png");

  k_hand_estimator_list_ = {
      std::make_shared<GuGestureEstimator>(),
      std::make_shared<ChokiGestureEstimator>(),
      std::make_shared<PaGestureEstimator>(),
      std::make_shared<HeartGestureEstimator>(),
      std::make_shared<The103GestureEstimator>(),
      std::make_shared<RyoikiTenkaiGestureEstimator>(),
  };

  k_gesture_image_map_;
  k_gesture_image_map_[GestureType::GU] =
      cv::imread("mediapipe/resources/gu.png");
  k_gesture_image_map_[GestureType::CHOKI] =
      cv::imread("mediapipe/resources/choki.png");
  k_gesture_image_map_[GestureType::PA] =
      cv::imread("mediapipe/resources/pa.png");
  k_gesture_image_map_[GestureType::HEART] =
      cv::imread("mediapipe/resources/heart.png");
  k_gesture_image_map_[GestureType::THE_103] =
      cv::imread("mediapipe/resources/103.png");
  k_gesture_image_map_[GestureType::RYOIKI_TENKAI] =
      cv::imread("mediapipe/resources/ryoiki_tenkai.jpg");

  k_operation_image_map_[ResultType::WIN] =
      cv::imread("mediapipe/resources/win_operation.png");
  k_operation_image_map_[ResultType::LOSE] =
      cv::imread("mediapipe/resources/loss_operation.png");
  k_operation_image_map_[ResultType::DRAW] =
      cv::imread("mediapipe/resources/draw_operation.png");

  k_gesture_and_rule_map_[GestureType::GU] = RuleType::JANKEN;
  k_gesture_and_rule_map_[GestureType::CHOKI] = RuleType::JANKEN;
  k_gesture_and_rule_map_[GestureType::PA] = RuleType::JANKEN;
  k_gesture_and_rule_map_[GestureType::HEART] = RuleType::IMITATION;
  k_gesture_and_rule_map_[GestureType::THE_103] = RuleType::IMITATION;
  k_gesture_and_rule_map_[GestureType::RYOIKI_TENKAI] = RuleType::IMITATION;

  k_th_score_ = 0.75;

  std::random_device rnd;  // 非決定的な乱数生成器
  k_mt_ = std::mt19937_64(rnd());

  k_opposite_gesture_rand_n_ = std::uniform_int_distribution<>(
      (int)(GestureType::UNKNOWN) + 1, (int)(GestureType::NUM_GESTURES)-1);
  // [1, n - 1] 範囲の一様乱数, UNKNOWN, NUM_GESTURES はスキップ

  k_janken_operation_rand_n_ =
      std::uniform_int_distribution<>(1, (int)(ResultType::NUM_RESULT_TYPES)-1);
  // [1, n - 1] 範囲の一様乱数, UNKNOWN, NUM_RESULT_TYPES はスキップ

  // まずジェスチャーを乱数で決めて、その後マップからルールタイプを参照し、それに応じてオペレーションを決定する。
  opposite_gesture_ = GestureType(k_opposite_gesture_rand_n_(k_mt_));
  // opposite_gesture_ = GestureType(k_opposite_gesture_rand_n_(k_mt_));
  rule_ = k_gesture_and_rule_map_[opposite_gesture_];

  if (rule_ == RuleType::JANKEN) {
    operation_ = ResultType(k_janken_operation_rand_n_(k_mt_));
  } else {
    operation_ = ResultType::UNKNOWN;
  }
}

// TODO: refactor a code below
void PostProcessor::Execute(
    const cv::Mat &camera_frame_raw,
    std::vector<mediapipe::NormalizedLandmarkList> *landmarks_list,
    cv::Mat *output_frame_for_display,
    std::chrono::system_clock::time_point *start_time, bool *grab_frames) {
#if 0
  cv::Mat landmark_image = camera_frame_raw;
#else
  cv::Mat landmark_image;
  VisUtility::BlurImage(cv::Size(11, 11), camera_frame_raw, &landmark_image);
#endif

  for (int i = 0; i < landmarks_list->size(); i++) {
    mediapipe::NormalizedLandmarkList &landmarks = landmarks_list->at(i);
    // draw frame-lines
    VisUtility::DrawFrameLines(landmarks, landmark_image, &landmark_image);

    // draw node-points
    VisUtility::DrawNodePoints(landmarks, landmark_image, &landmark_image);
  }

  cv::Mat output_frame_display_right;
  output_frame_display_right = cv::Mat::zeros(
      cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);

  std::vector<bool> new_status_list((int)(GestureType::NUM_GESTURES) - 1);

  auto current_recognized_type = GestureType::UNKNOWN;
  if (landmarks_list->size() == 0) {
    VisUtility::Overlap(output_frame_display_right, k_description_image_,
                        (camera_frame_raw.rows - k_description_image_.cols) / 2 + 45,
                        (camera_frame_raw.rows - k_description_image_.rows) / 2,
                        k_description_image_.cols * 0.8, k_description_image_.rows * 0.8);
  } else {
    // else if (landmarks_list.size() == 1) {
    // 片手 -> 両手でもおｋにした。
    cv::Mat gesture_image;
    for (auto &estimator : k_hand_estimator_list_) {
      // std::cout << landmarks_list[0].landmark_size() << std::endl;
      estimator->Initialize();
      const GestureType temp_recognized_type =
          estimator->Recognize(*landmarks_list);
      if (temp_recognized_type != GestureType::UNKNOWN) {
        gesture_image = k_gesture_image_map_[temp_recognized_type];
        current_recognized_type = temp_recognized_type;  // ジェスチャー仮確定
      }
    }
    if (gesture_image.empty()) {
      gesture_image = cv::Mat::zeros(
          cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);
    } else {
      // std::cout << "resize1" << std::endl;
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
  if (current_recognized_type != GestureType::UNKNOWN) {
    cv::Mat gesture_image = k_gesture_image_map_[current_recognized_type];
    // std::cout << "resize2" << std::endl;
    const float resize_ratio =
        ((float)k_your_hand_image_.rows / gesture_image.rows);
    const int resized_gesture_image_width =
        std::roundl(gesture_image.cols * resize_ratio);
    const int resized_gesture_image_height =
        std::roundl(gesture_image.rows * resize_ratio);
    cv::resize(gesture_image, gesture_image,
               cv::Size(resized_gesture_image_width,
                        resized_gesture_image_height));

    cv::Mat overlap_image;
    cv::hconcat(k_your_hand_image_, gesture_image, overlap_image);

    // 全体の横幅がカメラフレームの縦幅と同じなので注意。
    VisUtility::Overlap(output_frame_display_right, overlap_image,
                        (camera_frame_raw.rows - overlap_image.cols) / 2, 0,
                        overlap_image.cols, overlap_image.rows);
  }
  cv::putText(
      output_frame_display_right, std::string("Quit: <ESC>"),
      cv::Point(camera_frame_raw.rows - 130, camera_frame_raw.rows - 10), 1,
      1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_4);

  // Update all status-buffer.
  new_status_list[(int)(current_recognized_type)] = true;

  cv::Mat output_frame_display_left = cv::Mat::zeros(
      cv::Size(camera_frame_raw.rows, camera_frame_raw.rows), CV_8UC3);

  if (num_frames_since_resetting < 10) {
    // 合否表示注はバッファをクリアしておく。
    status_buffer_list_ = std::vector<StatusBuffer>();
    StatusBufferProcessor::Initialize(k_buffer_size_, &status_buffer_list_);

    cv::circle(output_frame_display_left,
               cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2),
               100, cv::Scalar(0, 255, 0), 5, cv::LINE_4);
    num_frames_since_resetting++;
  } else {
    // ジェスチャー確定処理 ---

    // 合否結果を表示しているときは更新しない。
    // for (auto status : new_status_list) std::cout << (int)status << " ";
    // std::cout << std::endl;

    StatusBufferProcessor::Update(new_status_list, &status_buffer_list_);

    std::vector<float> score_list;
    StatusBufferProcessor::CalculateStatistics(status_buffer_list_,
                                               &score_list);
    GestureType candidate_of_gesture_type = GestureType::UNKNOWN;
    float max_score = 0;
    for (int i = 0; i < score_list.size(); i++) {
      if (max_score < score_list[i]) {
        candidate_of_gesture_type = GestureType(i);
        max_score = score_list[i];
      }
    }
    // std::cout << status_buffer_list_.size() << " " << score_list.size() << " " << (int)candidate_of_gesture_type << std::endl;
    // std::cout << (int)GestureType::RYOIKI_TENKAI << " " << (int)GestureType::NUM_GESTURES << std::endl;

    // Judgement
    // --- ジェスチャー確定処理

    // スコアを更新するタイミングでは次のお題を表示しない。
    bool flag_for_update;
    if (rule_ == RuleType::JANKEN) {
      const ResultType current_result_type = JankenJudgement::JudgeNormalJanken(
          candidate_of_gesture_type, opposite_gesture_);
      flag_for_update = (current_result_type == operation_);
    }
    else if (rule_ == RuleType::IMITATION)
      flag_for_update = (candidate_of_gesture_type == opposite_gesture_);

    if (flag_for_update && k_th_score_ < max_score) {
      win_cnt_++;

      // 相手の次の手は今のと重複しないようにする。
      GestureType next_correct_gesture_type = candidate_of_gesture_type;
      while (candidate_of_gesture_type == next_correct_gesture_type) {
        GestureType next_opposite_gesture =
            GestureType(k_opposite_gesture_rand_n_(k_mt_));
        RuleType next_rule = k_gesture_and_rule_map_[next_opposite_gesture];

        ResultType next_operation;
        ResultType next_result_type;
        bool next_flag_for_update;

        if (next_rule == RuleType::JANKEN) {
          next_operation =
              ResultType(k_janken_operation_rand_n_(k_mt_));
          next_result_type = JankenJudgement::JudgeNormalJanken(
              candidate_of_gesture_type, next_opposite_gesture);
          next_flag_for_update = (next_result_type == next_operation);
        }
        else if (next_rule == RuleType::IMITATION) {
          next_operation =
              ResultType::UNKNOWN;
          next_flag_for_update = (candidate_of_gesture_type == next_opposite_gesture);
        }

        if (!next_flag_for_update) {
          operation_ = next_operation;
          opposite_gesture_ = next_opposite_gesture;
          rule_ = next_rule;
          break;
        }
      }

      num_frames_since_resetting = 0;
    } else {
      // CreateWhiteImage(cv::Size(camera_frame_raw.rows,
      // camera_frame_raw.rows),
      //                  &output_frame_display_left);
      output_frame_display_left = k_gesture_image_map_[opposite_gesture_];
      // std::cout << output_frame_display_left.empty() << std::endl;
      // std::cout << "resize3" << std::endl;
      cv::resize(output_frame_display_left, output_frame_display_left,
                 cv::Size(camera_frame_raw.rows, camera_frame_raw.rows));
      cv::Mat ope_image;

      if (rule_ == RuleType::JANKEN)
        ope_image = k_operation_image_map_[operation_];
      else if (rule_ == RuleType::IMITATION)
        ope_image = cv::imread("mediapipe/resources/imitation_operation.png");

      // 全体の横幅がカメラフレームの縦幅と同じなので注意。
      VisUtility::Overlap(output_frame_display_left, ope_image,
                          (camera_frame_raw.rows - ope_image.cols) / 2, 0,
                          ope_image.cols, ope_image.rows);
    }
    *landmarks_list =
        std::vector<mediapipe::NormalizedLandmarkList>();  // reset
  }

  // delay
  // std::this_thread::sleep_for(std::chrono::milliseconds(500));

  const auto current_time = std::chrono::system_clock::now();
  const double time_sec =
      (double)(std::chrono::duration_cast<std::chrono::microseconds>(
                   current_time - *start_time)
                   .count() /
               1000.0) /
      1000.0;
  const double time_left_sec =
      k_limit_time_sec_ - std::round(time_sec * 10) / 10;
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
              std::string("Score: ") + std::to_string(win_cnt_),
              cv::Point(20, camera_frame_raw.rows - 20), 2, 0.8,
              cv::Scalar(0, 255, 0), 2, cv::LINE_4);

  // cv::circle(*output_frame_display_right, cv::Point(x, y), 2, cv::Scalar(0,
  // 0, 255), 4,
  //            cv::LINE_4);
  // --- 左の表示

  cv::hconcat(output_frame_display_left, output_frame_display_right,
              *output_frame_for_display);
  // --- PostProcess

  // cv::cvtColor(output_frame_display_right, output_frame_display_right,
  // cv::COLOR_RGB2BGR);
  cv::imshow(k_window_name_, *output_frame_for_display);
  // Press any key to exit.

  if (time_left_sec <= 0) {
    const cv::Mat result_image =
        cv::Mat::zeros(output_frame_display_left.size(), CV_8UC3);
    cv::putText(
        result_image, std::string("Your Score: ") + std::to_string(win_cnt_),
        // cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2),
        cv::Point(30, camera_frame_raw.rows / 2 - 35), 2, 1.5,
        cv::Scalar(0, 255, 0), 2, cv::LINE_4);
    cv::putText(
        result_image, std::string("Restart: <Space>"),
        // cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2),
        cv::Point(30, camera_frame_raw.rows / 2 + 20), 2, 1.5,
        cv::Scalar(0, 255, 0), 2, cv::LINE_4);
    cv::putText(
        result_image, std::string("Quit: <ESC>"),
        // cv::Point(camera_frame_raw.rows / 2, camera_frame_raw.rows / 2),
        cv::Point(30, camera_frame_raw.rows / 2 + 75), 2, 1.5,
        cv::Scalar(0, 255, 0), 2, cv::LINE_4);

    const cv::Mat instruction_image =
        cv::Mat::zeros(output_frame_display_left.size(), CV_8UC3);
    cv::hconcat(result_image, instruction_image, *output_frame_for_display);
    cv::imshow(k_window_name_, *output_frame_for_display);
    const int pressed_key = cv::waitKey(0);
    if (pressed_key == k_cv_waitkey_esc_) {
      // terminate
      *grab_frames = false;
    } else if (pressed_key == k_cv_waitkey_spase_) {
      // restart
      *start_time = std::chrono::system_clock::now();
      win_cnt_ = 0;
      status_buffer_list_ = std::vector<StatusBuffer>();
      StatusBufferProcessor::Initialize(k_buffer_size_, &status_buffer_list_);
    }
  } else {
    const int pressed_key = cv::waitKey(1);
    if (pressed_key == k_cv_waitkey_esc_) {
      // terminate
      *grab_frames = false;
    }
  }
}