#pragma once

#include <deque>
#include <vector>

#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"

typedef std::deque<bool> StatusBuffer;

class StatusBufferProcessor {
  // ステータスバッファー更新クラス
 public:
  static void Initialize(std::vector<StatusBuffer> *status_buffer_list);

  static void Update(const std::vector<bool> &new_status_list,
                     std::vector<StatusBuffer> *status_buffer_list);

  static void CalculateStatistics(
      const std::vector<StatusBuffer> &status_buffer_list,
      std::vector<float> *result_list);

 private:
  static const int k_buffer_size_;
};
