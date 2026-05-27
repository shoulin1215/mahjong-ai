# game_engine/shanten.py
# 向听数计算 —— 日麻规则（标准形 + 七对子 + 国士无双）

from itertools import combinations

# ==================== 牌型编码 ====================

# 国士无双需要的 13 种牌（一九字）
KOKUSHI_TILES = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]


def tile_to_index(tile: str) -> int:
    """将牌名转换为 0-33 的整数索引"""
    suit_map = {'m': 0, 'p': 9, 's': 18, 'z': 27}
    suit = tile[-1]
    num = int(tile[:-1])
    return suit_map[suit] + num - 1


def tiles_to_counts(tiles: list[str]) -> list[int]:
    """将手牌列表转换为 34 元素的计数数组"""
    counts = [0] * 34
    for t in tiles:
        try:
            counts[tile_to_index(t)] += 1
        except (KeyError, ValueError, IndexError):
            pass
    return counts


def index_to_tile(idx: int) -> str:
    """将索引转换回牌名"""
    suits = ['m', 'p', 's']
    if idx < 27:
        suit = suits[idx // 9]
        num = idx % 9 + 1
        return f"{num}{suit}"
    else:
        return f"{idx - 27 + 1}z"


# ==================== 向听数计算核心 ====================

def calc_shanten(counts: list[int]) -> int:
    """
    综合计算向听数，取三种牌型中的最小值：
    - 标准形（4 面子 1 雀头）
    - 七对子
    - 国士无双

    向听数 = -1 表示已和牌。
    """
    best = _calc_shanten_standard(counts)
    best = min(best, _calc_shanten_chiitoi(counts))
    best = min(best, _calc_shanten_kokushi(counts))
    return best


def _calc_shanten_standard(counts: list[int]) -> int:
    """标准形（4 面子 1 雀头）向听数"""
    best = 8

    for head in range(34):
        if counts[head] < 2:
            continue
        counts[head] -= 2
        s = _calc_mentsu(counts, 0, 0, 0)
        best = min(best, s - 1)  # 有雀头，减1
        counts[head] += 2

    # 无雀头
    s = _calc_mentsu(counts, 0, 0, 0)
    best = min(best, s)

    return best


def _calc_shanten_chiitoi(counts: list[int]) -> int:
    """七对子向听数 = 6 - 对子数"""
    pairs = sum(1 for c in counts if c >= 2)
    # 如果有 7 个对子，向听 = -1（和牌）
    return 6 - pairs


def _calc_shanten_kokushi(counts: list[int]) -> int:
    """国士无双向听数 = 13 - 幺九种数 - (是否有幺九对子 ? 1 : 0)"""
    unique_yaokyuu = sum(1 for i in KOKUSHI_TILES if counts[i] >= 1)
    has_pair = any(counts[i] >= 2 for i in KOKUSHI_TILES)
    return 13 - unique_yaokyuu - (1 if has_pair else 0)


def _calc_mentsu(counts: list[int], idx: int, mentsu: int, taatsu: int) -> int:
    """
    递归计算面子/搭子组合的最小向听数。
    mentsu: 已完成面子数
    taatsu: 已完成搭子数（包括对子）
    """
    # 跳过空牌
    while idx < 34 and counts[idx] == 0:
        idx += 1

    if idx >= 34:
        total = mentsu + taatsu
        if total > 4:
            taatsu = 4 - mentsu
        return 8 - 2 * mentsu - taatsu

    best = 8 - 2 * mentsu - taatsu

    # 尝试刻子（3张相同）
    if counts[idx] >= 3:
        counts[idx] -= 3
        best = min(best, _calc_mentsu(counts, idx, mentsu + 1, taatsu))
        counts[idx] += 3

    # 尝试顺子（只对数牌）
    suit = idx // 9
    num = idx % 9
    if suit < 3 and num <= 6 and counts[idx + 1] > 0 and counts[idx + 2] > 0:
        counts[idx] -= 1
        counts[idx + 1] -= 1
        counts[idx + 2] -= 1
        best = min(best, _calc_mentsu(counts, idx, mentsu + 1, taatsu))
        counts[idx] += 1
        counts[idx + 1] += 1
        counts[idx + 2] += 1

    # 尝试对子搭子
    if counts[idx] >= 2:
        counts[idx] -= 2
        best = min(best, _calc_mentsu(counts, idx + 1, mentsu, taatsu + 1))
        counts[idx] += 2

    # 尝试两面/嵌张搭子
    if suit < 3 and num <= 7 and counts[idx + 1] > 0:
        counts[idx] -= 1
        counts[idx + 1] -= 1
        best = min(best, _calc_mentsu(counts, idx + 1, mentsu, taatsu + 1))
        counts[idx] += 1
        counts[idx + 1] += 1

    if suit < 3 and num <= 6 and counts[idx + 2] > 0:
        counts[idx] -= 1
        counts[idx + 2] -= 1
        best = min(best, _calc_mentsu(counts, idx + 1, mentsu, taatsu + 1))
        counts[idx] += 1
        counts[idx + 2] += 1

    # 孤张（不用作搭子）
    counts[idx] -= 1
    best = min(best, _calc_mentsu(counts, idx, mentsu, taatsu))
    counts[idx] += 1

    return best


# ==================== 有效进张计算 ====================

def calc_effective_tiles(tiles: list[str]) -> list[str]:
    """
    计算有效进张（摸哪张牌能减少向听数）。
    注意：此函数假设手牌为 13 张（摸牌前）。
    """
    counts = tiles_to_counts(tiles)
    current_shanten = calc_shanten(counts)

    effective = []
    for i in range(34):
        if counts[i] >= 4:
            continue
        counts[i] += 1
        new_shanten = calc_shanten(counts)
        if new_shanten < current_shanten:
            effective.append(index_to_tile(i))
        counts[i] -= 1

    return effective


# ==================== 14张手牌出牌建议 ====================

def best_discard(tiles: list[str]) -> tuple[str, int, list[str]]:
    """
    对 14 张手牌，找出打出哪张后向听数最小。

    Returns:
        (最佳出牌, 出牌后向听数, 出牌后的有效进张)
    """
    best_tile = tiles[0]
    best_shanten = 9
    best_effective: list[str] = []

    seen = set()
    for i, tile in enumerate(tiles):
        if tile in seen:
            continue
        seen.add(tile)

        remaining = tiles[:i] + tiles[i+1:]
        shanten = calc_shanten(tiles_to_counts(remaining))
        effective = calc_effective_tiles(remaining) if shanten >= 0 else []

        if shanten < best_shanten or (shanten == best_shanten and len(effective) > len(best_effective)):
            best_shanten = shanten
            best_tile = tile
            best_effective = effective

    return best_tile, best_shanten, best_effective
