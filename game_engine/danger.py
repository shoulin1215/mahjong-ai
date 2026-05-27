# game_engine/danger.py
# 危险牌评估：根据场上弃牌判断安全度

SUIT_MAP = {'m': 0, 'p': 9, 's': 18, 'z': 27}


def assess_danger(
    tile: str,
    discard_pool: list[list[str]],
    round_wind: str = '1z',
    seat_wind: str = '1z'
) -> dict:
    """
    评估一张牌的危险度。

    打点评估维度：
    - 字牌宣言数：已打出越多越安全
    - 筋牌：对方打过某张牌，则该牌两面搭子接受度下降
    - 无关系牌（字牌末节）
    - 现物：对方已打出的牌 = 完全安全

    Returns:
        {
            "tile": "5m",
            "danger_level": 0-5,  # 0=极安全, 5=极危险
            "reason": "...",
            "safe_players": [0,1,2]  # 相对安全的玩家索引
        }
    """
    safe_players = []
    danger_scores = [0, 0, 0, 0]  # 对四家各自的危险度

    # 索引 0=自家（不计入），1-3=其他三家
    for i in range(1, 4):
        pool = discard_pool[i] if i < len(discard_pool) else []

        # 规则1：现物（已打出过该牌）= 完全安全
        if tile in pool:
            safe_players.append(i)
            continue

        # 规则2：字牌评估
        if tile.endswith('z'):
            tile_num = int(tile[0])
            # 役牌（中发白，风牌）危险
            if tile_num >= 5:  # 中发白
                danger_scores[i] += 3
            elif tile == round_wind or tile == seat_wind:
                danger_scores[i] += 3
            else:
                danger_scores[i] += 1
            continue

        # 规则3：数牌危险度（中张 > 端张）
        num = int(tile[0])
        suit = tile[1]

        # 中张（4-6）危险度最高，端张（1-2, 8-9）相对安全
        if 4 <= num <= 6:
            danger_scores[i] += 3
        elif 3 <= num <= 7:
            danger_scores[i] += 2
        else:
            danger_scores[i] += 1

        # 规则4：筋牌检测（简化版）
        # 如果对方打了 n，则 n-3 和 n+3 相对安全（两面搭子消失）
        for p_tile in pool:
            if not p_tile.endswith(suit):
                continue
            p_num = int(p_tile[0])
            if abs(p_num - num) == 3:
                danger_scores[i] = max(0, danger_scores[i] - 1)

    avg_danger = sum(danger_scores[1:]) / 3 if len(danger_scores) > 1 else 0
    danger_level = min(5, int(avg_danger))

    return {
        "tile": tile,
        "danger_level": danger_level,
        "reason": _danger_reason(danger_level, safe_players),
        "safe_players": safe_players
    }


def _danger_reason(level: int, safe_players: list) -> str:
    reasons = {
        0: "现物/极安全，可放心打出",
        1: "端张字牌，危险度低",
        2: "筋牌或边张，相对安全",
        3: "中张数牌，有一定风险",
        4: "危险中张，建议评估进攻价值",
        5: "高危险牌，非必要不打"
    }
    base = reasons.get(level, "")
    if safe_players:
        base += f"（对玩家 {safe_players} 是现物）"
    return base


def rank_discards_by_safety(
    candidates: list[str],
    discard_pool: list[list[str]]
) -> list[tuple[str, dict]]:
    """
    对候选出牌按安全度排序（从最安全到最危险）
    Returns: [(tile, danger_info), ...]
    """
    ranked = []
    for tile in candidates:
        info = assess_danger(tile, discard_pool)
        ranked.append((tile, info))
    ranked.sort(key=lambda x: x[1]['danger_level'])
    return ranked
