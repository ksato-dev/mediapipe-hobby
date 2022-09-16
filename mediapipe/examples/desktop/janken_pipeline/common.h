#pragma once

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

// cite: https://github.com/ksato-dev/JankenExercise
enum class ResultType {
  UNKNOWN,
  WIN,
  LOSE,
  DRAW,
  NUM_RESULT_TYPES,
};

enum class ScoreRank {
  BAD,
  GOOD,
  EXCELLENT,
  NUM_SCORE_RANKS,
};