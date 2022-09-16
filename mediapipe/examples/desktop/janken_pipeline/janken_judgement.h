#pragma once

#include <string>

#include "mediapipe/examples/desktop/janken_pipeline/common.h"
#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"

class JankenJudgement {
 public:
  // const bool Judge(const GestureType &gesture_type);
  static const ResultType JudgeNormalJanken(
      const GestureType &your_gesture,
      const GestureType &opposite_gesture);
};
