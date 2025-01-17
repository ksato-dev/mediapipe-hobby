# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

# package(default_visibility = ["//mediapipe/examples:__subpackages__"
# ])
package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "janken++",
    srcs = ["janken++.cc"],

    visibility = ["//visibility:public"],

    # linkopts = ["-/SUBSYSTEM:WINDOWS"],
    deps = [
        # "//mediapipe/examples/desktop:janken++",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",

        "//mediapipe/examples/desktop/janken_pipeline:gesture_estimator",
        "//mediapipe/examples/desktop/janken_pipeline:janken_judgement",
        "//mediapipe/examples/desktop/janken_pipeline:status_buffer_processor",
        "//mediapipe/examples/desktop/janken_pipeline:vis_utils",
        "//mediapipe/examples/desktop/janken_pipeline:post_processor",

        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "gesture_estimator",
    srcs = ["gesture_estimator.cc"],
    hdrs = ["gesture_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "janken_judgement",
    srcs = ["janken_judgement.cc"],
    hdrs = ["janken_judgement.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "status_buffer_processor",
    srcs = ["status_buffer_processor.cc"],
    hdrs = ["status_buffer_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "vis_utils",
    srcs = ["vis_utils.cc"],
    hdrs = ["vis_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
    ],
    alwayslink = 1,
)

cc_library(
    name = "post_processor",
    srcs = ["post_processor.cc"],
    hdrs = ["post_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/examples/desktop/janken_pipeline:gesture_estimator",
        "//mediapipe/examples/desktop/janken_pipeline:janken_judgement",
        "//mediapipe/examples/desktop/janken_pipeline:status_buffer_processor",
        "//mediapipe/examples/desktop/janken_pipeline:vis_utils",

        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
    ],
    alwayslink = 1,
)
