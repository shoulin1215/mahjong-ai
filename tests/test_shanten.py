"""测试向听数计算引擎"""

import pytest
from game_engine.shanten import (
    tile_to_index, tiles_to_counts, calc_shanten,
    calc_effective_tiles, best_discard, index_to_tile,
)


class TestTileConversion:
    """牌编码转换测试"""

    def test_man_tiles(self):
        assert tile_to_index("1m") == 0
        assert tile_to_index("9m") == 8

    def test_pin_tiles(self):
        assert tile_to_index("1p") == 9
        assert tile_to_index("9p") == 17

    def test_sou_tiles(self):
        assert tile_to_index("1s") == 18
        assert tile_to_index("9s") == 26

    def test_honor_tiles(self):
        assert tile_to_index("1z") == 27  # 东
        assert tile_to_index("7z") == 33  # 白

    def test_index_to_tile_roundtrip(self):
        for i in range(34):
            tile = index_to_tile(i)
            assert tile_to_index(tile) == i


class TestShantenCalculation:
    """向听数计算测试"""

    def test_already_won(self):
        """和牌 = -1"""
        tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "1p", "1s", "1s"]
        counts = tiles_to_counts(tiles)
        assert calc_shanten(counts) == -1

    def test_tenpai(self):
        """听牌 = 0"""
        tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "1p", "1s"]
        counts = tiles_to_counts(tiles)
        assert calc_shanten(counts) == 0

    def test_typical_hand(self):
        """普通手牌向听数"""
        tiles = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "1z", "5m"]
        counts = tiles_to_counts(tiles)
        shanten = calc_shanten(counts)
        assert 0 <= shanten <= 6

    def test_chiitoi_tenpai(self):
        """七对子听牌"""
        tiles = ["1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "5m", "6m", "6m", "7m"]
        counts = tiles_to_counts(tiles)
        assert calc_shanten(counts) == 0


class TestEffectiveTiles:
    """有效进张测试"""

    def test_tenpai_has_effective(self):
        """听牌时有有效进张"""
        tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "1p", "1s"]
        effective = calc_effective_tiles(tiles)
        assert len(effective) > 0

    def test_high_shanten_has_effective(self):
        """高向听也有有效进张"""
        tiles = ["1m", "5m", "9m", "1p", "5p", "9p", "1s", "5s", "9s", "1z", "3z", "5z", "7z"]
        effective = calc_effective_tiles(tiles)
        assert len(effective) > 0


class TestBestDiscard:
    """最佳出牌测试"""

    def test_14_tiles_returns_best(self):
        """14张手牌返回最佳出牌"""
        tiles = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "1z", "5m", "9m"]
        best, shanten_after, effective = best_discard(tiles)
        assert best in tiles
        assert isinstance(shanten_after, int)
