"""测试提示词模板

从 mahjong-ai 迁移，适配 quehun 的 llm_advisor.prompt 模块。
"""

from llm_advisor.prompt import (
    build_system_prompt,
    build_user_prompt,
    build_danger_context,
    tiles_to_chinese,
    TILE_NAMES,
    WIND_NAMES,
)
from game_engine.state import GameState


class TestSystemPrompt:
    """系统提示词测试"""

    def test_contains_required_sections(self):
        """系统提示词包含必要的策略指导"""
        prompt = build_system_prompt()
        assert "JSON" in prompt
        assert "discard" in prompt

    def test_contains_output_format(self):
        """包含 JSON 输出格式模板"""
        prompt = build_system_prompt()
        assert '"discard"' in prompt


class TestBuildUserPrompt:
    """用户提示词构建测试"""

    def test_basic_hand(self):
        """基本手牌信息"""
        state = GameState(hand_tiles=["1m", "2m", "3m", "1z"])
        state.shanten = 3
        prompt = build_user_prompt(state)
        assert "手牌" in prompt
        assert "一万" in prompt

    def test_shanten_display(self):
        """向听数显示"""
        state = GameState(hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z"])
        state.shanten = 2
        prompt = build_user_prompt(state)
        assert "2向听" in prompt

    def test_shanten_after_discard_display(self):
        """14张时显示打出后向听"""
        state = GameState(hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z", "5m"])
        state.shanten = 2
        state.shanten_after_discard = 1
        prompt = build_user_prompt(state)
        assert "打出后1向听" in prompt

    def test_danger_context(self):
        """危险度上下文"""
        state = GameState(hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "6z", "7z", "5m"])
        state.shanten = 2
        state.danger_ranking = [
            ("1z", {"danger_level": 1, "reason": "字牌较安全"}),
        ]
        ctx = build_danger_context(state)
        assert "危险度" in ctx
        assert "东" in ctx  # 1z = 东


class TestTileNames:
    """牌名映射测试"""

    def test_all_number_tiles(self):
        for suit in ["m", "p", "s"]:
            for i in range(1, 10):
                key = f"{i}{suit}"
                assert key in TILE_NAMES

    def test_all_honor_tiles(self):
        for i in range(1, 8):
            key = f"{i}z"
            assert key in TILE_NAMES

    def test_wind_names(self):
        assert WIND_NAMES["1z"] == "东"
        assert WIND_NAMES["2z"] == "南"


class TestTilesToChinese:
    """牌列表转中文测试"""

    def test_basic_conversion(self):
        result = tiles_to_chinese(["1m", "2m", "3m"])
        assert "一万" in result
        assert "二万" in result

    def test_mixed_suits(self):
        result = tiles_to_chinese(["1m", "1p", "1s", "5z"])
        assert "一万" in result
        assert "一饼" in result
        assert "一索" in result
        assert "中" in result
