
#pragma once

#include <map>
#include <utility>
#include <vector>

#include "mediapipe/examples/desktop/janken_pipeline/common.h"
#include "mediapipe/framework/formats/landmark.pb.h"

// TODO: 公式のコードを見て、左右判定をできるようにする。

class AbstractHandGestureEstimator {
 public:
  // AbstractHandGestureEstimator();
  virtual ~AbstractHandGestureEstimator(){};
  virtual void Initialize() = 0;
  virtual const GestureType Recognize(
      const std::vector<mediapipe::NormalizedLandmarkList>
          &hand_landmarks_list) = 0;
};

// One Hand
class OneHandGestureEstimator : public AbstractHandGestureEstimator {
 public:
  // OneHandGestureEstimator();
  virtual void Initialize();

  virtual const GestureType
  // Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
  //               &hand_landmarks_list);
  Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
                &hand_landmarks_list) {
    return GestureType::UNKNOWN;
  };

 protected:
  // TODO: implement
  void SetHandType(const HandType &hand_type);
  void TransformLandmarks(
      const std::vector<mediapipe::NormalizedLandmarkList> &hand_landmarks_list,
      std::vector<mediapipe::NormalizedLandmarkList>
          *transformed_hand_landmarks_list);

  HandType hand_type_;  // default: left
};

class GuGestureEstimator : public OneHandGestureEstimator {
 public:
  // GuGestureEstimator(){};
  virtual void Initialize(){};
  virtual const GestureType Recognize(
      const std::vector<mediapipe::NormalizedLandmarkList>
          &hand_landmarks_list);
};

class ChokiGestureEstimator : public OneHandGestureEstimator {
 public:
  // ChokiGestureEstimator(){};
  virtual void Initialize(){};
  virtual const GestureType Recognize(
      const std::vector<mediapipe::NormalizedLandmarkList>
          &hand_landmarks_list);
};

class PaGestureEstimator : public OneHandGestureEstimator {
 public:
  // PaGestureEstimator(){};
  virtual void Initialize(){};
  virtual const GestureType Recognize(
      const std::vector<mediapipe::NormalizedLandmarkList>
          &hand_landmarks_list);
};

class RyoikiTenkaiGestureEstimator : public OneHandGestureEstimator {
 public:
  // PaGestureEstimator(){};
  virtual void Initialize(){};
  virtual const GestureType Recognize(
      const std::vector<mediapipe::NormalizedLandmarkList>
          &hand_landmarks_list);
};
// --- One Hand

// Two Hands ---
class TwoHandsGestureEstimator : public AbstractHandGestureEstimator {
 public:
  // TwoHandsGestureEstimator(){};
  // virtual ~TwoHandsGestureEstimator(){};
  virtual void Initialize(){};
  virtual const GestureType Recognize(
      const std::vector<mediapipe::NormalizedLandmarkList>
          &hand_landmarks_list) {
    return GestureType::UNKNOWN;
  };
};

class HeartGestureEstimator : public TwoHandsGestureEstimator {
 public:
  // TwoHandsGestureEstimator(){};
  // virtual ~TwoHandsGestureEstimator(){};
  virtual void Initialize(){};
  virtual const GestureType Recognize(
      const std::vector<mediapipe::NormalizedLandmarkList>
          &hand_landmarks_list);
};

class The103GestureEstimator : public TwoHandsGestureEstimator {
 public:
  // TwoHandsGestureEstimator(){};
  // virtual ~TwoHandsGestureEstimator(){};
  virtual void Initialize(){};
  virtual const GestureType Recognize(
      const std::vector<mediapipe::NormalizedLandmarkList>
          &hand_landmarks_list);
};

// --- Two Hands
