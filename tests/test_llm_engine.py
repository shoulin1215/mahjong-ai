"""测试 LLM 决策引擎 - 4层JSON解析与容错

从 mahjong-ai 迁移，适配 quehun 的 llm_advisor.advisor 模块。
"""

import pytest
from llm_advisor.advisor import LLMAdvisor
from game_engine.state import GameState


def make_game_state(hand_tiles=None):
    """快速构造 GameState"""
    if hand_tiles is None:
        hand_tiles = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z", "5m"]
    state = GameState(hand_tiles=hand_tiles)
    # 简单计算向听（不依赖完整 build_game_state，只测解析逻辑）
    state.shanten = 2
    state.best_discard = "5m"
    return state


class TestJSONParsing:
    """JSON 响应解析测试（4层策略）"""

    def setup_method(self):
        self.advisor = LLMAdvisor.__new__(LLMAdvisor)  # 不调用 __init__（避免读环境变量）

    def test_parse_valid_json_direct(self):
        """策略1: 直接返回合法 JSON"""
        raw = '{"discard": "5p", "reason": "效率低", "alternative": "9s"}'
        state = make_game_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z", "5m"])
        result = self.advisor._parse_response(raw, state)
        assert result.discard == "5p"
        assert result.confidence == 0.9

    def test_parse_codeblock_json(self):
        """策略2: ```json ... ``` 包裹的 JSON"""
        raw = '''```json
{"discard": "1m", "reason": "保留好形", "alternative": "7z"}
```'''
        state = make_game_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z"])
        result = self.advisor._parse_response(raw, state)
        assert result.discard == "1m"

    def test_parse_json_with_prose(self):
        """策略3: JSON 混在自然语言中（正则提取花括号块）"""
        raw = '根据当前手牌，我建议：\n{"discard": "1z", "reason": "字牌价值低", "alternative": "6z"}\n以上是建议。'
        state = make_game_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z"])
        result = self.advisor._parse_response(raw, state)
        assert result.discard == "1z"

    def test_fallback_tile_code(self):
        """策略4: 正则提取牌编码"""
        raw = "建议打出 5p，因为效率最低"
        state = make_game_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z", "5m"])
        result = self.advisor._parse_response(raw, state)
        assert result.discard == "5p"
        assert result.confidence == 0.8

    def test_fallback_chinese(self):
        """策略5: 中文牌名匹配"""
        raw = "建议打出五饼，因为效率最低"
        state = make_game_state(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z", "5m"])
        result = self.advisor._parse_response(raw, state)
        assert result.discard == "5p"
        assert result.confidence == 0.7

    def test_algorithm_fallback(self):
        """全部失败时使用算法推荐"""
        raw = "I cannot determine the best discard."
        state = make_game_state()
        result = self.advisor._parse_response(raw, state)
        assert result.discard == state.best_discard

    def test_empty_response_uses_algorithm(self):
        """空响应使用算法推荐"""
        state = make_game_state()
        result = self.advisor._parse_response("", state)
        assert result.discard is not None

    def test_compatible_with_discard_reason(self):
        """兼容 mahjong-ai 的 discard_reason 字段"""
        raw = '{"discard": "5m", "discard_reason": "孤张万字"}'
        state = make_game_state()
        result = self.advisor._parse_response(raw, state)
        assert result.discard == "5m"
        assert "孤张万字" in result.reason


class TestExtractTileCode:
    """牌编码正则提取测试"""

    def setup_method(self):
        self.advisor = LLMAdvisor.__new__(LLMAdvisor)

    def test_extract_from_hand(self):
        hand = ["1m", "5p", "9s", "3z"]
        assert self.advisor._extract_tile_code("建议打 5p", hand) == "5p"

    def test_not_in_hand(self):
        hand = ["1m", "2m", "3m"]
        assert self.advisor._extract_tile_code("建议打 9s", hand) is None

    def test_multiple_matches_picks_first_in_hand(self):
        hand = ["3p", "5m"]
        result = self.advisor._extract_tile_code("3p or 5m", hand)
        assert result in hand
