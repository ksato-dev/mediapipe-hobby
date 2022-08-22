
#include "mediapipe/examples/desktop/janken_pipeline/status_buffer_processor.h"

void StatusBufferProcessor::InitializeGestureStatusBufferList(
    const int &buffer_size, std::vector<StatusBuffer> *status_buffer_list) {
  for (int i = 0; i < (int)(JankenGestureType::NUM_GESTURES); i++)
    status_buffer_list->push_back(StatusBuffer(buffer_size, 0));
}

void StatusBufferProcessor::UpdateGestureStatusBufferList(
    const std::vector<bool> &new_status_list,
    std::vector<StatusBuffer> *status_buffer_list) {
  const int buffer_size = status_buffer_list->at(0).size();
  for (int i = 0; i < status_buffer_list->size(); i++) {
    status_buffer_list->at(i).pop_front();
    status_buffer_list->at(i).push_back(new_status_list[i]);
  }
}

void StatusBufferProcessor::CalculateStatistics(
    const std::vector<StatusBuffer> &status_buffer_list,
    std::vector<float> *result_list) {
  const int buffer_size = status_buffer_list.at(0).size();
  for (int i = 0; i < status_buffer_list.size(); i++) {
    auto &status_buffer = status_buffer_list.at(i);
    float sum_status = 0.0;
    for (int j = 0; j < status_buffer.size(); j++) {
      sum_status += (float)(status_buffer.at(j));
    }
    const float avg_status = sum_status / (float)buffer_size;
    result_list->push_back(avg_status);
  }
}