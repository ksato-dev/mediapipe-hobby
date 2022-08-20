#pragma once

#include <string>
#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"

// cite: https://github.com/ksato-dev/JankenExercise
enum ResultType {
  UNKNOWN,
  WIN,
  LOSE,
  DRAW,
  NUM_RESULT_TYPES,
};

class JankenJudgement {
 public:

  // const bool Judge(const GestureType &gesture_type);
  static const ResultType JudgeNormalJanken(
      const JankenGestureType &your_gesture,
      const JankenGestureType &opposite_gesture);

 private:
  // const ResultType JudgeExpandedJanken(const GestureType &gesture_type, );  // ただタイプの一致を見ればよいのでいらない？

  // std::string last_operation_msg;
  // std::string last_operation_msg;
};
