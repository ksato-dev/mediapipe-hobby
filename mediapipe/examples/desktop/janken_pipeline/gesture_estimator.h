
#pragma once

#include <map>
#include <utility>
#include <vector>

#include "mediapipe/framework/formats/landmark.pb.h"


enum HandType {
  LEFT_HAND = 0,
  RIGHT_HAND = 1,
};

enum class RuleType { UNKNOWN, JANKEN, IMITATION, NUM_RULES };

enum class GestureType {
  UNKNOWN,
  GU,     // JANKEN
  CHOKI,  // JANKEN
  PA,     // JANKEN
  HEART,
  THE_103,
  RYOIKI_TENKAI,
  NUM_GESTURES
};

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
