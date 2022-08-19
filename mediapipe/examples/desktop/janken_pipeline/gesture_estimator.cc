
#include "mediapipe/examples/desktop/janken_pipeline/gesture_estimator.h"

void OneHandGestureEstimator::Initialize() {
    hand_type_ = HandType::LEFT_HAND;
};

const GestureType GuGestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList>
        &hand_landmarks_list) {
    // std::cout << "num_of_landmarks_list:" << hand_landmarks_list.size()
    //           << std::endl;

    // mediapipe::NormalizedLandmarkList landms = hand_landmarks_list[int(hand_type_)];
    // auto &landms = hand_landmarks_list[int(hand_type_)];
    auto &landms = hand_landmarks_list[0];
    if (landms.landmark_size() != 21) return GestureType::UNKNOWN;

    // std::cout << "num_of_landmarks:" << landms.landmark_size()
    //           << std::endl;

    // std::cout << "1" << std::endl;
    const bool index_nodes_status =
        landms.landmark(6).y() < landms.landmark(7).y() &&
        landms.landmark(6).y() < landms.landmark(8).y();

    // std::cout << "2" << std::endl;
    const bool middle_nodes_status =
        landms.landmark(10).y() < landms.landmark(11).y() &&
        landms.landmark(10).y() < landms.landmark(12).y();

    // std::cout << "3" << std::endl;
    const bool ring_nodes_status =
        landms.landmark(14).y() < landms.landmark(15).y() &&
        landms.landmark(14).y() < landms.landmark(16).y();

    // std::cout << "4" << std::endl;
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
    // std::cout << "num_of_landmarks_list:" << hand_landmarks_list.size()
    //           << std::endl;

    // mediapipe::NormalizedLandmarkList landms = hand_landmarks_list[int(hand_type_)];
    // auto &landms = hand_landmarks_list[int(hand_type_)];
    auto &landms = hand_landmarks_list[0];
    if (landms.landmark_size() != 21) return GestureType::UNKNOWN;

    // std::cout << "num_of_landmarks:" << landms.landmark_size()
    //           << std::endl;

    // std::cout << "1" << std::endl;
    const bool index_nodes_status =
        landms.landmark(5).y() > landms.landmark(6).y() &&
        landms.landmark(6).y() > landms.landmark(7).y() &&
        landms.landmark(7).y() > landms.landmark(8).y();

    // std::cout << "2" << std::endl;
    const bool middle_nodes_status =
        landms.landmark(9).y() > landms.landmark(10).y() &&
        landms.landmark(10).y() > landms.landmark(11).y() &&
        landms.landmark(11).y() > landms.landmark(12).y();

    // std::cout << "3" << std::endl;
    const bool ring_nodes_status =
        landms.landmark(14).y() < landms.landmark(15).y() &&
        landms.landmark(14).y() < landms.landmark(16).y();

    // std::cout << "4" << std::endl;
    const bool pinky_nodes_status =
        landms.landmark(18).y() < landms.landmark(19).y() &&
        landms.landmark(18).y() < landms.landmark(20).y();

    GestureType ret_type = GestureType::UNKNOWN;
    if (index_nodes_status && middle_nodes_status && ring_nodes_status && pinky_nodes_status)
        ret_type = GestureType::CHOKI;

    return ret_type;
}

const GestureType PaGestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList>
        &hand_landmarks_list) {
    // std::cout << "num_of_landmarks_list:" << hand_landmarks_list.size()
    //           << std::endl;

    // mediapipe::NormalizedLandmarkList landms = hand_landmarks_list[int(hand_type_)];
    // auto &landms = hand_landmarks_list[int(hand_type_)];
    auto &landms = hand_landmarks_list[0];
    if (landms.landmark_size() != 21) return GestureType::UNKNOWN;

    // std::cout << "num_of_landmarks:" << landms.landmark_size()
    //           << std::endl;

    // std::cout << "1" << std::endl;
    const bool index_nodes_status =
        landms.landmark(5).y() > landms.landmark(6).y() &&
        landms.landmark(6).y() > landms.landmark(7).y() &&
        landms.landmark(7).y() > landms.landmark(8).y();

    // std::cout << "2" << std::endl;
    const bool middle_nodes_status =
        landms.landmark(9).y() > landms.landmark(10).y() &&
        landms.landmark(10).y() > landms.landmark(11).y() &&
        landms.landmark(11).y() > landms.landmark(12).y();

    // std::cout << "3" << std::endl;
    const bool ring_nodes_status =
        landms.landmark(13).y() > landms.landmark(14).y() &&
        landms.landmark(14).y() > landms.landmark(15).y() &&
        landms.landmark(15).y() > landms.landmark(16).y();

    // std::cout << "4" << std::endl;
    const bool pinky_nodes_status =
        landms.landmark(17).y() > landms.landmark(18).y() &&
        landms.landmark(18).y() > landms.landmark(19).y() &&
        landms.landmark(19).y() > landms.landmark(20).y();

    GestureType ret_type = GestureType::UNKNOWN;
    if (index_nodes_status && middle_nodes_status && ring_nodes_status && pinky_nodes_status)
        ret_type = GestureType::PA;

    return ret_type;
}

const GestureType HeartGestureEstimator::Recognize(
    const std::vector<mediapipe::NormalizedLandmarkList> &hand_landmarks_list) {
    if (hand_landmarks_list.size() != 2) return GestureType::UNKNOWN;
    auto &landms1 = hand_landmarks_list[0];
    auto &landms2 = hand_landmarks_list[1];

    const float eps = 0.05;

    const bool thumb_nodes_status1 =
        abs(landms1.landmark(4).x() - landms2.landmark(4).x()) < eps;
    const bool thumb_nodes_status2 =
        (landms1.landmark(3).y() < landms1.landmark(4).y()) &&
        (landms2.landmark(3).y() < landms2.landmark(4).y());

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
        // if (thumb_nodes_status && pinky_nodes_status)
        ret_type = GestureType::HEART;

    return ret_type;
}
