# game_engine/__init__.py
from .shanten import calc_shanten, calc_effective_tiles, best_discard
from .danger import assess_danger, rank_discards_by_safety
from .state import (
    GameState, build_game_state,
    GamePhase, EventType, GameStateManager,
)
