# game_engine/state.py
# 游戏状态构建与管理
#
# 职责：
# 1. 将 DetectionResult 构建为 GameState（向听/进张/危险度）
# 2. 追踪牌局状态变化，检测事件（摸牌/出牌/副露）
# 3. 控制 LLM 请求时机（缓存 + 事件触发，避免重复计算）

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .shanten import calc_shanten, tiles_to_counts, calc_effective_tiles, best_discard
from .danger import rank_discards_by_safety

logger = logging.getLogger(__name__)


# ============================================
# 游戏阶段 & 事件类型
# ============================================

class GamePhase(Enum):
    """游戏阶段"""
    IDLE = "idle"
    DRAWING = "drawing"        # 摸牌 (14张)
    DISCARDING = "discarding"  # 出牌 (13张)
    WAITING_CALL = "waiting_call"
    ENDED = "ended"


class EventType(Enum):
    """事件类型"""
    NO_CHANGE = "no_change"
    DREW_TILE = "drew_tile"       # 摸牌 13->14
    DISCARDED = "discarded"       # 出牌 14->13
    MELD = "meld"                 # 副露
    UNKNOWN_CHANGE = "unknown_change"


# ============================================
# GameState 数据结构
# ============================================

@dataclass
class GameState:
    # 手牌
    hand_tiles: list[str] = field(default_factory=list)

    # 弃牌池（四家，0=自家）
    discard_pool: list[list[str]] = field(default_factory=lambda: [[], [], [], []])

    # 场风/自风
    round_wind: str = '1z'   # 东
    seat_wind: str = '1z'    # 东

    # 宝牌
    doras: list[str] = field(default_factory=list)

    # 计算结果（构建时填充）
    shanten: int = 8
    shanten_after_discard: Optional[int] = None  # 14张时打出最佳牌后的向听数
    effective_tiles: list[str] = field(default_factory=list)
    best_discard: Optional[str] = None
    danger_ranking: list[tuple] = field(default_factory=list)

    # 事件追踪（从 mahjong-ai 迁移）
    phase: GamePhase = GamePhase.IDLE
    last_event: EventType = EventType.NO_CHANGE
    last_event_time: float = 0.0
    last_discard: str = ""
    total_recognitions: int = 0
    total_llm_calls: int = 0

    def to_prompt_dict(self) -> dict:
        """转换为 LLM Prompt 友好的结构"""
        return {
            "hand": self.hand_tiles,
            "hand_count": len(self.hand_tiles),
            "shanten": self.shanten,
            "shanten_after_discard": self.shanten_after_discard,
            "effective_tiles": self.effective_tiles,
            "best_discard_by_algorithm": self.best_discard,
            "discard_pool_self": self.discard_pool[0],
            "discard_pool_others": self.discard_pool[1:],
            "doras": self.doras,
            "round_wind": self.round_wind,
            "seat_wind": self.seat_wind,
        }


# ============================================
# 构建函数
# ============================================

def build_game_state(detection) -> GameState:
    """
    根据 DetectionResult 构建完整的 GameState。
    """
    state = GameState(
        hand_tiles=detection.hand_tiles,
        discard_pool=detection.discard_pool,
    )

    tiles = state.hand_tiles
    if not tiles:
        return state

    # 计算向听数
    counts = tiles_to_counts(tiles)
    state.shanten = calc_shanten(counts)

    # 14 张牌：计算最佳出牌
    if len(tiles) == 14:
        bd, shanten_after, effective = best_discard(tiles)
        state.best_discard = bd
        state.shanten_after_discard = shanten_after
        state.effective_tiles = effective
        state.phase = GamePhase.DRAWING

        # 危险度排名（只对候选出牌）
        candidates = list(set(tiles))
        state.danger_ranking = rank_discards_by_safety(candidates, state.discard_pool)

    # 13 张牌：计算有效进张
    elif len(tiles) == 13:
        state.effective_tiles = calc_effective_tiles(tiles)
        state.phase = GamePhase.DISCARDING

    return state


# ============================================
# GameStateManager - 事件检测 & LLM 节流
# （从 mahjong-ai/game_state.py 迁移）
# ============================================

class GameStateManager:
    """游戏状态管理器

    核心职责：
    1. 对比前后两次识别结果，判断发生了什么事件
    2. 控制何时需要调用 LLM（避免重复请求）
    3. 追踪牌局进度
    """

    def __init__(
        self,
        cache_ttl: float = 2.0,
        on_state_change: Optional[Callable] = None,
    ):
        self.cache_ttl = cache_ttl
        self.on_state_change = on_state_change
        self.state: Optional[GameState] = None

        # 前一次手牌（用于事件检测）
        self._prev_hand: List[str] = []

        # LLM 结果缓存
        self._cached_decision: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0.0

    def update(self, game_state: GameState) -> EventType:
        """更新状态并返回事件类型

        Args:
            game_state: 当前帧的 GameState（由 build_game_state 构建）

        Returns:
            EventType 事件类型
        """
        if game_state is not None:
            game_state.total_recognitions += 1

        if not game_state or not game_state.hand_tiles:
            return EventType.NO_CHANGE

        prev_hand = self._prev_hand
        curr_hand = game_state.hand_tiles
        self._prev_hand = list(curr_hand)

        self.state = game_state

        event = self._detect_event(prev_hand, curr_hand)
        game_state.last_event = event
        game_state.last_event_time = time.time()

        if event != EventType.NO_CHANGE and self.on_state_change:
            try:
                self.on_state_change(event, game_state)
            except Exception as e:
                logger.error(f"状态回调异常: {e}")

        return event

    def _detect_event(
        self, prev_hand: List[str], curr_hand: List[str]
    ) -> EventType:
        """通过对比前后手牌推断事件"""
        if not prev_hand:
            return EventType.UNKNOWN_CHANGE if curr_hand else EventType.NO_CHANGE

        prev_count = len(prev_hand)
        curr_count = len(curr_hand)

        # 13 -> 14: 摸牌
        if prev_count == 13 and curr_count == 14:
            new_tile = self._find_new_tile(prev_hand, curr_hand)
            if new_tile:
                self.state.last_discard = ""
                logger.info(f"[事件] 摸牌: {new_tile}")
                return EventType.DREW_TILE

        # 14 -> 13: 出牌
        if prev_count == 14 and curr_count == 13:
            removed = self._find_removed_tile(prev_hand, curr_hand)
            if removed:
                self.state.last_discard = removed
                logger.info(f"[事件] 出牌: {removed}")
                return EventType.DISCARDED

        # 数量不变但内容变了 -> 可能是副露或识别错误
        if prev_count == curr_count and prev_hand != curr_hand:
            logger.debug("[事件] 手牌内容变化（可能副露/误识别）")
            return EventType.UNKNOWN_CHANGE

        return EventType.NO_CHANGE

    @staticmethod
    def _find_new_tile(prev_tiles: List[str], curr_tiles: List[str]) -> Optional[str]:
        """找出新增的牌"""
        prev_cnt = Counter(prev_tiles)
        curr_cnt = Counter(curr_tiles)
        diff = curr_cnt - prev_cnt
        if diff:
            return list(diff.keys())[0]
        return None

    @staticmethod
    def _find_removed_tile(prev_tiles: List[str], curr_tiles: List[str]) -> Optional[str]:
        """找出被移除的牌"""
        prev_cnt = Counter(prev_tiles)
        curr_cnt = Counter(curr_tiles)
        diff = prev_cnt - curr_cnt
        if diff:
            return list(diff.keys())[0]
        return None

    def needs_llm_decision(self) -> bool:
        """判断是否需要重新调用 LLM 决策

        条件满足其一即触发：
        1. 缓存过期
        2. 发生了新的事件（摸牌/出牌后需要新建议）
        """
        if self._cached_decision is None:
            return True

        cache_age = time.time() - self._cache_time
        if cache_age > self.cache_ttl:
            logger.debug(f"LLM缓存过期 ({cache_age:.1f}s > {self.cache_ttl}s)")
            return True

        # 有新事件时强制刷新
        if self.state and self.state.last_event in (
            EventType.DREW_TILE,
            EventType.DISCARDED,
            EventType.MELD,
        ):
            return True

        return False

    def cache_decision(self, decision: Dict[str, Any]):
        """缓存 LLM 决策结果"""
        self._cached_decision = decision
        self._cache_time = time.time()
        if self.state:
            self.state.total_llm_calls += 1

    def get_cached_decision(self) -> Optional[Dict[str, Any]]:
        """获取缓存的决策结果"""
        return self._cached_decision

    def reset(self):
        """重置状态（新一局开始）"""
        self.state = None
        self._prev_hand = []
        self._cached_decision = None
        self._cache_time = 0.0
        logger.info("游戏状态已重置")
