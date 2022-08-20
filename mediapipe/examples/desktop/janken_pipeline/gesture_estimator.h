
#pragma once

#include <vector>
#include "mediapipe/framework/formats/landmark.pb.h"

enum HandType {
    LEFT_HAND = 0,
    RIGHT_HAND = 1,
};

// enum class GestureType {
//     UNKNOWN = 0,
//     GU = 1,
//     CHOKI = 2,
//     PA = 3,
//     HEART = 4,
//     NUM_GESTURES = 5,
// };

enum class JankenGestureType {
    UNKNOWN = 0,
    GU = 1,
    CHOKI = 2,
    PA = 3,
    NUM_GESTURES = 4,
};

class AbstractHandGestureEstimator {
  public:
    // AbstractHandGestureEstimator();
    virtual ~AbstractHandGestureEstimator(){};
    virtual void Initialize() = 0;
    virtual const JankenGestureType
    Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
                  &hand_landmarks_list) = 0;
};

// One Hand
class OneHandGestureEstimator : public AbstractHandGestureEstimator {
  public:
    // OneHandGestureEstimator();
    virtual void Initialize();

    virtual const JankenGestureType
    // Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
    //               &hand_landmarks_list);
    Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
                  &hand_landmarks_list){return JankenGestureType::UNKNOWN;};
  protected:
    // TODO: implement
    void SetHandType(const HandType &hand_type);
    void
    TransformLandmarks(const std::vector<mediapipe::NormalizedLandmarkList>
                           &hand_landmarks_list,
                       std::vector<mediapipe::NormalizedLandmarkList>
                           *transformed_hand_landmarks_list);

    HandType hand_type_;  // default: left
};

class GuGestureEstimator : public OneHandGestureEstimator {
  public:
    // GuGestureEstimator(){};
    virtual void Initialize(){};
    virtual const JankenGestureType
    Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
                  &hand_landmarks_list);
};

class ChokiGestureEstimator : public OneHandGestureEstimator {
  public:
    // ChokiGestureEstimator(){};
    virtual void Initialize(){};
    virtual const JankenGestureType
    Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
                  &hand_landmarks_list);
};

class PaGestureEstimator : public OneHandGestureEstimator {
  public:
    // PaGestureEstimator(){};
    virtual void Initialize(){};
    virtual const JankenGestureType
    Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
                  &hand_landmarks_list);
};
// --- One Hand

// Two Hands ---
class TwoHandsGestureEstimator : public AbstractHandGestureEstimator {
  public:
    // TwoHandsGestureEstimator(){};
    // virtual ~TwoHandsGestureEstimator(){};
    virtual void Initialize(){};
    virtual const JankenGestureType
    Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
                  &hand_landmarks_list){return JankenGestureType::UNKNOWN;};
};

// class HeartGestureEstimator : public TwoHandsGestureEstimator {
//   public:
//     // TwoHandsGestureEstimator(){};
//     // virtual ~TwoHandsGestureEstimator(){};
//     virtual void Initialize(){};
//     virtual const JankenGestureType
//     Recognize(const std::vector<mediapipe::NormalizedLandmarkList>
//                   &hand_landmarks_list);
// };
// --- Two Hands
