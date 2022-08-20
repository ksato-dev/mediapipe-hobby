
#include "mediapipe/examples/desktop/janken_pipeline/janken_judgement.h"

const ResultType JankenJudgement::JudgeNormalJanken(
    const JankenGestureType &your_gesture,
    const JankenGestureType &opposite_gesture) {
  ResultType ret_result_type = ResultType::UNKNOWN;
  if (your_gesture != JankenGestureType::UNKNOWN) {
    if (your_gesture == opposite_gesture)
      ret_result_type = ResultType::DRAW;
    else if (your_gesture == JankenGestureType(1) &&
                 opposite_gesture == JankenGestureType(3) ||
             your_gesture == JankenGestureType(2) &&
                 opposite_gesture == JankenGestureType(1) ||
             your_gesture == JankenGestureType(3) &&
                 opposite_gesture == JankenGestureType(2))
      ret_result_type = ResultType::LOSE;
    else
      ret_result_type = ResultType::WIN;
  }
  return ret_result_type;
}
