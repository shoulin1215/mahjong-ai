# llm_advisor/prompt.py
# Prompt 构造引擎

from game_engine.state import GameState

TILE_NAMES = {
    '1m': '一万', '2m': '二万', '3m': '三万', '4m': '四万', '5m': '五万',
    '6m': '六万', '7m': '七万', '8m': '八万', '9m': '九万',
    '1p': '一饼', '2p': '二饼', '3p': '三饼', '4p': '四饼', '5p': '五饼',
    '6p': '六饼', '7p': '七饼', '8p': '八饼', '9p': '九饼',
    '1s': '一索', '2s': '二索', '3s': '三索', '4s': '四索', '5s': '五索',
    '6s': '六索', '7s': '七索', '8s': '八索', '9s': '九索',
    '1z': '东', '2z': '南', '3z': '西', '4z': '北',
    '5z': '中', '6z': '发', '7z': '白'
}

WIND_NAMES = {'1z': '东', '2z': '南', '3z': '西', '4z': '北'}


def tiles_to_chinese(tiles: list[str]) -> str:
    return '、'.join(TILE_NAMES.get(t, t) for t in tiles)


def build_system_prompt() -> str:
    return """你是一位专业的日本麻将（雀魂）AI顾问，精通日麻规则、战术和概率计算。

你的任务：根据玩家当前的手牌状态，给出**最优出牌建议**。

回答格式要求（必须严格遵守）：
1. 首先输出 JSON 格式的决策，用 ```json ``` 包裹
2. JSON 结构：{"discard": "牌编码", "reason": "理由", "alternative": "备选牌编码或null"}
3. 牌编码格式：万子=1m~9m，饼子=1p~9p，索子=1s~9s，字牌=1z~7z
4. JSON 之后可以用自然语言补充详细分析

示例回复：
```json
{"discard": "5m", "reason": "孤张万字，对进张无贡献", "alternative": "9s"}
```
5m是万字孤张，打出后保留9s的索子搭子更为灵活。

你已收到程序计算的向听数、有效进张、危险度分析作为参考，请综合判断给出最终建议。"""


def build_user_prompt(state: GameState) -> str:
    d = state.to_prompt_dict()

    hand_str = tiles_to_chinese(d['hand'])
    # 14张时：shanten 是当前向听，shanten_after_discard 是打出后向听
    if d.get('shanten_after_discard') is not None:
        shanten_str = f"{d['shanten']}向听（打出后{d['shanten_after_discard']}向听）"
    else:
        shanten_str = "听牌" if d['shanten'] == 0 else f"{d['shanten']}向听"
    effective_str = tiles_to_chinese(d['effective_tiles']) if d['effective_tiles'] else "无"
    best_algo_str = TILE_NAMES.get(d['best_discard_by_algorithm'], d['best_discard_by_algorithm'] or '未知')

    self_discards = tiles_to_chinese(d['discard_pool_self']) if d['discard_pool_self'] else "无"

    others_discard_lines = []
    for i, pool in enumerate(d['discard_pool_others'], 1):
        player_name = ['右家', '对家', '左家'][i - 1]
        tiles_str = tiles_to_chinese(pool) if pool else "无"
        others_discard_lines.append(f"  {player_name}：{tiles_str}")
    others_discards = '\n'.join(others_discard_lines) if others_discard_lines else "  无数据"

    doras_str = tiles_to_chinese(d['doras']) if d['doras'] else "未知"
    round_wind = WIND_NAMES.get(d['round_wind'], d['round_wind'])
    seat_wind = WIND_NAMES.get(d['seat_wind'], d['seat_wind'])

    prompt = f"""【当前局面】
场风：{round_wind}风  自风：{seat_wind}风  宝牌：{doras_str}

【手牌】（共{d['hand_count']}张）
{hand_str}

【算法分析】
- 向听数：{shanten_str}
- 有效进张：{effective_str}
- 算法推荐出牌：{best_algo_str}

【自家弃牌】
{self_discards}

【其他玩家弃牌】
{others_discards}

请给出你的出牌建议，说明选择理由。"""

    return prompt


def build_danger_context(state: GameState) -> str:
    """生成危险度上下文（追加到 prompt）"""
    if not state.danger_ranking:
        return ""

    lines = ["【危险度参考（从低到高）】"]
    for tile, info in state.danger_ranking[:5]:  # 只显示前5个
        name = TILE_NAMES.get(tile, tile)
        level_str = "★" * info['danger_level'] + "☆" * (5 - info['danger_level'])
        lines.append(f"  {name}：{level_str}  {info['reason']}")

    return '\n'.join(lines)
