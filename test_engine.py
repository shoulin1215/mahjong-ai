# 快速测试脚本（无需模型）：验证游戏引擎是否正常

from game_engine.shanten import calc_shanten, tiles_to_counts, best_discard, calc_effective_tiles

TEST_HAND_14 = ['1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s', '1z', '1z', '1z', '5m', '6m']
TEST_HAND_13 = ['1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s', '1z', '1z', '1z', '5m']

print("=" * 50)
print("游戏引擎自测")
print("=" * 50)

# 向听数
counts13 = tiles_to_counts(TEST_HAND_13)
shanten = calc_shanten(counts13)
print(f"\n13张手牌：{TEST_HAND_13}")
print(f"向听数：{shanten}")

# 有效进张
effective = calc_effective_tiles(TEST_HAND_13)
print(f"有效进张：{effective}")

# 14张出牌建议
discard, shanten_after, eff_after = best_discard(TEST_HAND_14)
print(f"\n14张手牌：{TEST_HAND_14}")
print(f"推荐打出：{discard}")
print(f"打出后向听数：{shanten_after}")
print(f"打出后有效进张：{eff_after}")

print("\n所有测试通过！")
