
#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"

void OneHandGestureEstimator::Initialize() {
    hand_type_ = HandType::LEFT_HAND;
};

const GestureType GuGestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList>
        &hand_landmarks_list) {
    // auto &landms = hand_landmarks_list[int(hand_type_)];
    auto &landms = hand_landmarks_list[0];
    if (landms.landmark_size() != 21) return GestureType::UNKNOWN;

    const bool index_nodes_status =
        landms.landmark(6).y() < landms.landmark(7).y() &&
        landms.landmark(6).y() < landms.landmark(8).y();

    const bool middle_nodes_status =
        landms.landmark(10).y() < landms.landmark(11).y() &&
        landms.landmark(10).y() < landms.landmark(12).y();

    const bool ring_nodes_status =
        landms.landmark(14).y() < landms.landmark(15).y() &&
        landms.landmark(14).y() < landms.landmark(16).y();

    const bool pinky_nodes_status =
        landms.landmark(18).y() < landms.landmark(19).y() &&
        landms.landmark(18).y() < landms.landmark(20).y();

    GestureType ret_type = GestureType::UNKNOWN;
    if (index_nodes_status && middle_nodes_status && ring_nodes_status && pinky_nodes_status)
        ret_type = GestureType::GU;

    return ret_type;
}

const GestureType ChokiGestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList>
        &hand_landmarks_list) {

    auto &landms = hand_landmarks_list[0];
    if (landms.landmark_size() != 21) return GestureType::UNKNOWN;

    const bool index_nodes_status =
        landms.landmark(5).y() > landms.landmark(6).y() &&
        landms.landmark(6).y() > landms.landmark(7).y() &&
        landms.landmark(7).y() > landms.landmark(8).y();

    const float eps = 0.08;
    const bool index_and_middle_nodes_status =
        abs(landms.landmark(8).x() - landms.landmark(12).x()) > eps;

    const bool middle_nodes_status =
        landms.landmark(9).y() > landms.landmark(10).y() &&
        landms.landmark(10).y() > landms.landmark(11).y() &&
        landms.landmark(11).y() > landms.landmark(12).y();

    const bool ring_nodes_status =
        landms.landmark(14).y() < landms.landmark(15).y() &&
        landms.landmark(14).y() < landms.landmark(16).y();

    const bool pinky_nodes_status =
        landms.landmark(18).y() < landms.landmark(19).y() &&
        landms.landmark(18).y() < landms.landmark(20).y();

    GestureType ret_type = GestureType::UNKNOWN;
    if (index_nodes_status && index_and_middle_nodes_status &&
        middle_nodes_status && ring_nodes_status && pinky_nodes_status)
        ret_type = GestureType::CHOKI;

    return ret_type;
}

const GestureType PaGestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList>
        &hand_landmarks_list) {

    auto &landms = hand_landmarks_list[0];
    if (landms.landmark_size() != 21) return GestureType::UNKNOWN;

    const bool index_nodes_status =
        landms.landmark(5).y() > landms.landmark(6).y() &&
        landms.landmark(6).y() > landms.landmark(7).y() &&
        landms.landmark(7).y() > landms.landmark(8).y();

    const bool middle_nodes_status =
        landms.landmark(9).y() > landms.landmark(10).y() &&
        landms.landmark(10).y() > landms.landmark(11).y() &&
        landms.landmark(11).y() > landms.landmark(12).y();

    const bool ring_nodes_status =
        landms.landmark(13).y() > landms.landmark(14).y() &&
        landms.landmark(14).y() > landms.landmark(15).y() &&
        landms.landmark(15).y() > landms.landmark(16).y();

    const bool pinky_nodes_status =
        landms.landmark(17).y() > landms.landmark(18).y() &&
        landms.landmark(18).y() > landms.landmark(19).y() &&
        landms.landmark(19).y() > landms.landmark(20).y();

    GestureType ret_type = GestureType::UNKNOWN;
    if (index_nodes_status && middle_nodes_status && ring_nodes_status && pinky_nodes_status)
        ret_type = GestureType::PA;

    return ret_type;
}

const GestureType RyoikiTenkaiGestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList>
        &hand_landmarks_list) {

    // TODO: 右か左かの判定を行う。

    auto &landms = hand_landmarks_list[0];
    if (landms.landmark_size() != 21) return GestureType::UNKNOWN;

    const bool index_nodes_status =
        landms.landmark(5).y() > landms.landmark(6).y() &&
        landms.landmark(6).y() > landms.landmark(7).y() &&
        landms.landmark(7).y() > landms.landmark(8).y();

    const float eps = 0.08;
    const bool index_and_middle_nodes_status =
        abs(landms.landmark(8).x() - landms.landmark(12).x()) < eps;

    const bool middle_nodes_status =
        landms.landmark(9).y() > landms.landmark(10).y() &&
        landms.landmark(10).y() > landms.landmark(11).y() &&
        landms.landmark(11).y() > landms.landmark(12).y();

    const bool ring_nodes_status =
        landms.landmark(14).y() < landms.landmark(15).y() &&
        landms.landmark(14).y() < landms.landmark(16).y();

    const bool pinky_nodes_status =
        landms.landmark(18).y() < landms.landmark(19).y() &&
        landms.landmark(18).y() < landms.landmark(20).y();

    GestureType ret_type = GestureType::UNKNOWN;
    if (index_nodes_status && index_and_middle_nodes_status &&
        middle_nodes_status && ring_nodes_status && pinky_nodes_status)
        ret_type = GestureType::RYOIKI_TENKAI;

    return ret_type;
}

const GestureType HeartGestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList> &hand_landmarks_list) {
    if (hand_landmarks_list.size() != 2) return GestureType::UNKNOWN;
    auto &landms1 = hand_landmarks_list[0];
    auto &landms2 = hand_landmarks_list[1];

    const float eps = 0.10;

    const bool thumb_nodes_status1 =
        abs(landms1.landmark(4).x() - landms2.landmark(4).x()) < eps;
    const bool thumb_nodes_status2 =
        (landms1.landmark(8).y() < landms1.landmark(4).y()) &&
        (landms2.landmark(8).y() < landms2.landmark(4).y()) &&
        (landms1.landmark(12).y() < landms1.landmark(4).y()) &&
        (landms2.landmark(12).y() < landms2.landmark(4).y()) &&
        (landms1.landmark(16).y() < landms1.landmark(4).y()) &&
        (landms2.landmark(16).y() < landms2.landmark(4).y()) &&
        (landms1.landmark(20).y() < landms1.landmark(4).y()) &&
        (landms2.landmark(20).y() < landms2.landmark(4).y());

    const bool index_nodes_status1 =
        abs(landms1.landmark(8).x() - landms2.landmark(8).x()) < eps;
    const bool index_nodes_status2 =
        (landms1.landmark(7).y() < landms1.landmark(8).y()) &&
        (landms2.landmark(7).y() < landms2.landmark(8).y());

    const bool middle_nodes_status1 =
        abs(landms1.landmark(12).x() - landms2.landmark(12).x()) < eps;
    const bool middle_nodes_status2 =
        (landms1.landmark(11).y() < landms1.landmark(12).y()) &&
        (landms2.landmark(11).y() < landms2.landmark(12).y());

    // ---
    const bool ring_nodes_status1 =
        abs(landms1.landmark(16).x() - landms2.landmark(16).x()) < eps;
    const bool ring_nodes_status2 =
        (landms1.landmark(15).y() < landms1.landmark(16).y()) &&
        (landms2.landmark(15).y() < landms2.landmark(16).y());

    // ---
    const bool pinky_nodes_status1 =
        abs(landms1.landmark(20).x() - landms2.landmark(20).x()) < eps;
    // 第１関節と第２関節
    const bool pinky_nodes_status2 =
        (landms1.landmark(19).y() < landms1.landmark(20).y()) &&
        (landms2.landmark(19).y() < landms2.landmark(20).y());

    GestureType ret_type = GestureType::UNKNOWN;
    if ((thumb_nodes_status1 && thumb_nodes_status2) &&
        ((index_nodes_status1 && index_nodes_status2) &&
         (middle_nodes_status1 && middle_nodes_status2) &&
         (ring_nodes_status1 && ring_nodes_status2) &&
         (pinky_nodes_status1 && pinky_nodes_status2)))
        ret_type = GestureType::HEART;

    return ret_type;
}

const GestureType The103GestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList> &hand_landmarks_list) {
    if (hand_landmarks_list.size() != 2) return GestureType::UNKNOWN;
    mediapipe::NormalizedLandmarkList landms1 = hand_landmarks_list[0];
    mediapipe::NormalizedLandmarkList landms2 = hand_landmarks_list[1];

    bool landms1_is_left = true;
    for (int i = 0; i < landms1.landmark_size(); i++) {
        const float point1_x = landms1.landmark(i).x();
        const float point2_x = landms2.landmark(i).x();
        if (point1_x > point2_x) landms1_is_left = false;
    }

    if (!landms1_is_left) {
        // std::cout << "landms1 is right" << std::endl;
        mediapipe::NormalizedLandmarkList temp = landms1;
        landms1 = landms2;
        landms2 = temp;
    }

    // right-hand
    const bool index_nodes_status1 =
        landms1.landmark(5).y() > landms1.landmark(6).y() &&
        landms1.landmark(6).y() > landms1.landmark(7).y() &&
        landms1.landmark(7).y() > landms1.landmark(8).y();

    const bool middle_nodes_status1 =
        landms1.landmark(10).y() < landms1.landmark(11).y() &&
        landms1.landmark(10).y() < landms1.landmark(12).y();

    const bool ring_nodes_status1 =
        landms1.landmark(14).y() < landms1.landmark(15).y() &&
        landms1.landmark(14).y() < landms1.landmark(16).y();

    const bool pinky_nodes_status1 =
        landms1.landmark(18).y() < landms1.landmark(19).y() &&
        landms1.landmark(18).y() < landms1.landmark(20).y();

    // ---

    // left-hand
    const float eps = 0.08;
    const bool thumb_and_index_nodes_status2 =
        abs(landms2.landmark(4).y() - landms2.landmark(8).y()) < eps;

    const bool thumb_nodes_status2 =
        (landms2.landmark(4).x() < landms2.landmark(2).x()) &&
        (landms2.landmark(3).x() < landms2.landmark(2).x());

    const bool index_nodes_status2 =
        (landms2.landmark(6).x() < landms2.landmark(5).x()) &&
        (landms2.landmark(7).x() < landms2.landmark(5).x()) &&
        (landms2.landmark(8).x() < landms2.landmark(5).x());

    const bool middle_nodes_status2 =
        landms2.landmark(9).y() > landms2.landmark(10).y() &&
        landms2.landmark(10).y() > landms2.landmark(11).y() &&
        landms2.landmark(11).y() > landms2.landmark(12).y();

    const bool ring_nodes_status2 =
        landms2.landmark(13).y() > landms2.landmark(14).y() &&
        landms2.landmark(14).y() > landms2.landmark(15).y() &&
        landms2.landmark(15).y() > landms2.landmark(16).y();

    const bool pinky_nodes_status2 =
        landms2.landmark(17).y() > landms2.landmark(18).y() &&
        landms2.landmark(18).y() > landms2.landmark(19).y() &&
        landms2.landmark(19).y() > landms2.landmark(20).y();

    GestureType ret_type = GestureType::UNKNOWN;
    if ((index_nodes_status1 && middle_nodes_status1 && ring_nodes_status1 &&
         pinky_nodes_status1) &&
        (thumb_and_index_nodes_status2 && thumb_nodes_status2 &&
         index_nodes_status2 && middle_nodes_status2 && ring_nodes_status2 &&
         pinky_nodes_status2))
        ret_type = GestureType::THE_103;

    return ret_type;
}