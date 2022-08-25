
#include "mediapipe/examples/desktop/janken_pipeline/janken_judgement.h"

const ResultType JankenJudgement::JudgeNormalJanken(
    const GestureType &your_gesture,
    const GestureType &opposite_gesture) {
  ResultType ret_result_type = ResultType::UNKNOWN;
  if (your_gesture != GestureType::UNKNOWN) {
    if (your_gesture == opposite_gesture)
      ret_result_type = ResultType::DRAW;
    else if (your_gesture == GestureType(1) &&
                 opposite_gesture == GestureType(3) ||
             your_gesture == GestureType(2) &&
                 opposite_gesture == GestureType(1) ||
             your_gesture == GestureType(3) &&
                 opposite_gesture == GestureType(2))
      ret_result_type = ResultType::LOSE;
    else
      ret_result_type = ResultType::WIN;
  }
  return ret_result_type;
}
