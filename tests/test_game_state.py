"""测试游戏状态管理模块

覆盖: 事件检测、LLM决策触发判断、缓存管理
从 mahjong-ai 迁移，适配 quehun 的 game_engine.state 模块。
"""

import pytest
from game_engine.state import (
    GameStateManager, GamePhase, EventType, GameState, build_game_state,
)


def make_state(hand_tiles, discard_pool=None):
    """快速构造 GameState"""
    return GameState(
        hand_tiles=list(hand_tiles),
        discard_pool=discard_pool or [[], [], [], []],
    )


class TestEventDetection:
    """事件检测测试"""

    def test_first_recognition(self):
        """首次识别应返回 UNKNOWN_CHANGE"""
        mgr = GameStateManager()
        state = make_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
                            "1z", "1z", "6z", "7z"])
        event = mgr.update(state)
        assert event == EventType.UNKNOWN_CHANGE

    def test_drew_tile_13_to_14(self):
        """摸牌: 13张 -> 14张"""
        mgr = GameStateManager()
        prev = make_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
                           "1z", "1z", "6z", "7z"])
        mgr.update(prev)

        curr = make_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
                           "1z", "1z", "6z", "7z", "5z"])
        event = mgr.update(curr)
        assert event == EventType.DREW_TILE

    def test_discarded_14_to_13(self):
        """出牌: 14张 -> 13张"""
        mgr = GameStateManager()
        prev = make_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
                           "1z", "1z", "6z", "7z", "5z"])
        mgr.update(prev)

        curr = make_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
                           "1z", "1z", "6z", "7z"])
        event = mgr.update(curr)
        assert event == EventType.DISCARDED
        assert mgr.state.last_discard == "5z"

    def test_no_change_same_hand(self):
        """手牌未变化 = NO_CHANGE"""
        mgr = GameStateManager()
        hand = make_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
                           "1z", "1z", "6z", "7z"])
        mgr.update(hand)
        event = mgr.update(hand)
        assert event == EventType.NO_CHANGE

    def test_empty_hand_returns_no_change(self):
        """空手牌返回 NO_CHANGE"""
        mgr = GameStateManager()
        empty = GameState(hand_tiles=[])
        event = mgr.update(empty)
        assert event == EventType.NO_CHANGE


class TestLLMDecisionTrigger:
    """LLM 决策触发条件测试"""

    def test_no_cache_triggers_decision(self):
        """无缓存时触发决策"""
        mgr = GameStateManager()
        assert mgr.needs_llm_decision() is True

    def test_drew_tile_triggers_decision(self):
        """摸牌后触发决策"""
        mgr = GameStateManager()
        prev = make_state(["1m", "2m", "3m"] * 4 + ["1z"])
        mgr.update(prev)
        mgr.cache_decision({"discard": "1z"})

        curr = make_state(["1m", "2m", "3m"] * 4 + ["1z", "2z"])
        mgr.update(curr)
        assert mgr.needs_llm_decision() is True

    def test_discarded_triggers_decision(self):
        """出牌后触发决策"""
        mgr = GameStateManager()
        prev = make_state(["1m", "2m", "3m"] * 4 + ["1z", "2z"])
        mgr.update(prev)
        mgr.cache_decision({"discard": "2z"})

        curr = make_state(["1m", "2m", "3m"] * 4 + ["1z"])
        mgr.update(curr)
        assert mgr.needs_llm_decision() is True


class TestFindTileDiffs:
    """牌差计算测试"""

    def test_find_new_tile(self):
        mgr = GameStateManager()
        prev = ["1m", "2m", "3m"]
        curr = ["1m", "2m", "3m", "4m"]
        result = mgr._find_new_tile(prev, curr)
        assert result == "4m"

    def test_find_removed_tile(self):
        mgr = GameStateManager()
        prev = ["1m", "2m", "3m", "4m"]
        curr = ["1m", "2m", "3m"]
        result = mgr._find_removed_tile(prev, curr)
        assert result == "4m"

    def test_find_removed_with_duplicate(self):
        """有一对相同牌，移除一张后仍能找到"""
        mgr = GameStateManager()
        prev = ["1m", "1m", "2m", "3m"]
        curr = ["1m", "2m", "3m"]
        result = mgr._find_removed_tile(prev, curr)
        assert result == "1m"


class TestGameStateManagerReset:
    """状态重置测试"""

    def test_reset_clears_cache(self):
        mgr = GameStateManager()
        state = make_state(["1m", "2m", "3m"])
        mgr.update(state)
        mgr.cache_decision({"discard": "3m"})
        assert mgr.get_cached_decision() is not None

        mgr.reset()
        assert mgr.get_cached_decision() is None
        assert mgr.state is None
