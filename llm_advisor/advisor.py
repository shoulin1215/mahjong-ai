# llm_advisor/advisor.py
# 大模型推理路由器：支持 Ollama（本地）和 OpenAI/Claude（云端）
#
# 合并了 mahjong-ai 的 4 层 JSON 解析 + 重试机制 + 连接池复用

import json as _json
import logging
import os
import re
import time
from typing import Optional

import httpx
from dotenv import load_dotenv

from game_engine.state import GameState
from vision_service.models import AdviceResult
from .prompt import build_system_prompt, build_user_prompt, build_danger_context, TILE_NAMES

load_dotenv()
logger = logging.getLogger(__name__)

# 反向映射：中文牌名 -> 编码
TILE_REVERSE = {v: k for k, v in TILE_NAMES.items()}


class LLMAdvisor:
    """
    大模型推理适配器，支持：
    - ollama：本地 Qwen2.5 等（默认）
    - openai：GPT-4o / DeepSeek / GLM / 通义千问等 OpenAI 兼容接口
    - claude：Claude 3.5 Sonnet

    改进（从 mahjong-ai 合并）：
    - 4 层 JSON 解析容错（直接JSON → 代码块 → 正则花括号 → 文本提取）
    - 超时重试 + 指数退避
    - 持久化 httpx 连接池
    """

    MAX_RETRIES = 2

    def __init__(self):
        self.backend = os.getenv('LLM_BACKEND', 'ollama')

        # Ollama
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2.5:9b')

        # OpenAI / OpenAI-Compatible（DeepSeek, GLM, 通义千问等）
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o')

        # Claude
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY', '')

        self.timeout = float(os.getenv('LLM_TIMEOUT', '30'))

        # 持久化 HTTP 客户端（连接复用）
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(f"LLMAdvisor 初始化，后端: {self.backend}, OpenAI base: {self.openai_base_url}")

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建持久化的 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """关闭 HTTP 客户端连接"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_advice(self, state: GameState) -> Optional[AdviceResult]:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(state)
        danger_ctx = build_danger_context(state)
        if danger_ctx:
            user_prompt += '\n\n' + danger_ctx

        # 重试机制（从 mahjong-ai 合并）
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if self.backend == 'ollama':
                    raw_text = await self._call_ollama(system_prompt, user_prompt)
                elif self.backend == 'openai':
                    raw_text = await self._call_openai(system_prompt, user_prompt)
                elif self.backend == 'claude':
                    raw_text = await self._call_claude(system_prompt, user_prompt)
                else:
                    logger.error(f"未知 LLM 后端: {self.backend}")
                    return self._fallback_advice(state)

                if not raw_text:
                    return self._fallback_advice(state)

                return self._parse_response(raw_text, state)

            except Exception as e:
                last_error = e
                logger.warning(f"LLM 第{attempt+1}次调用失败: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避
                continue

        logger.error(f"LLM 所有重试均失败: {last_error}")
        return self._fallback_advice(state)

    # ==================== Ollama ====================

    async def _call_ollama(self, system: str, user: str) -> str:
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512
            }
        }

        client = await self._get_client()
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get('message', {}).get('content', '')

    # ==================== OpenAI / OpenAI-Compatible ====================

    async def _call_openai(self, system: str, user: str) -> str:
        """OpenAI 及其兼容接口（DeepSeek / GLM / 通义千问 / Moonshot 等）"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 未配置")

        base_url = self.openai_base_url.rstrip('/')
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.3,
            "max_tokens": 512
        }

        client = await self._get_client()
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']

    # ==================== Claude ====================

    async def _call_claude(self, system: str, user: str) -> str:
        if not self.claude_api_key:
            raise ValueError("ANTHROPIC_API_KEY 未配置")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 512,
            "system": system,
            "messages": [{"role": "user", "content": user}]
        }

        client = await self._get_client()
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data['content'][0]['text']

    # ==================== 4 层响应解析（从 mahjong-ai 合并增强） ====================

    def _parse_response(self, text: str, state: GameState) -> AdviceResult:
        """
        从 LLM 回复文本中提取出推荐出牌。
        4 层策略：直接JSON → 代码块 → 正则花括号 → 编码/中文兜底。
        """
        discard = None
        alternative = None
        reason = text.strip()
        confidence = 0.5

        # 1. 尝试直接解析（文本以 { 开头）
        text_stripped = text.strip()
        if text_stripped.startswith("{") and text_stripped.endswith("}"):
            json_data = self._try_parse_json(text_stripped)
            if json_data:
                discard, alternative, reason, confidence = self._extract_from_json(
                    json_data, state, confidence=0.9
                )

        # 2. 提取 ```json ... ``` 代码块
        if not discard:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text_stripped, re.DOTALL)
            if match:
                json_data = self._try_parse_json(match.group(1).strip())
                if json_data:
                    discard, alternative, reason, confidence = self._extract_from_json(
                        json_data, state, confidence=0.9
                    )

        # 3. 正则提取包含 "discard" 的花括号块
        if not discard:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text_stripped)
            if match:
                json_data = self._try_parse_json(match.group(0))
                if json_data:
                    discard, alternative, reason, confidence = self._extract_from_json(
                        json_data, state, confidence=0.9
                    )

        # 4. 正则提取牌编码（如 "5m", "1z"）
        if not discard:
            discard = self._extract_tile_code(text, state.hand_tiles)
            if discard:
                confidence = 0.8

        # 5. 中文牌名匹配（最后兜底）
        if not discard:
            discard = self._extract_by_chinese(text, state.hand_tiles)
            if discard:
                confidence = 0.7

        # 如果都没匹配到，使用算法推荐
        if not discard:
            discard = state.best_discard

        return AdviceResult(
            discard=discard,
            reason=reason,
            confidence=confidence,
            alternative=alternative or (state.best_discard if state.best_discard != discard else None)
        )

    def _try_parse_json(self, text: str) -> Optional[dict]:
        """尝试解析 JSON，失败返回 None"""
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            return None

    def _extract_from_json(
        self, json_data: dict, state: GameState, confidence: float = 0.9
    ) -> tuple:
        """从 JSON dict 中提取 discard/alternative/reason

        Returns:
            (discard, alternative, reason, confidence)
        """
        discard = None
        alternative = None
        reason = ""
        conf = confidence

        discard_candidate = json_data.get("discard", "")
        if discard_candidate and discard_candidate in state.hand_tiles:
            discard = discard_candidate

        alt_candidate = json_data.get("alternative")
        if alt_candidate and alt_candidate in state.hand_tiles and alt_candidate != discard:
            alternative = alt_candidate

        # 兼容 mahjong-ai 的字段名
        json_reason = json_data.get("reason", "") or json_data.get("discard_reason", "")
        if json_reason:
            reason = json_reason

        return discard, alternative, reason, conf

    def _extract_tile_code(self, text: str, hand: list[str]) -> Optional[str]:
        """从文本中正则提取牌编码（1m~9m, 1p~9p, 1s~9s, 1z~7z）"""
        matches = re.findall(r'\b([1-9][mps]|[1-7]z)\b', text)
        for code in matches:
            if code in hand:
                return code
        return None

    def _extract_by_chinese(self, text: str, hand: list[str]) -> Optional[str]:
        """从文本中匹配中文牌名（兜底）"""
        for cn_name, code in TILE_REVERSE.items():
            pattern = rf'(?:打出|弃|打|推荐|建议)[^。\n]{{0,10}}{re.escape(cn_name)}'
            if re.search(pattern, text) and code in hand:
                return code
        return None

    def _fallback_advice(self, state: GameState) -> AdviceResult:
        """LLM 失败时的算法兜底"""
        discard = state.best_discard or (state.hand_tiles[0] if state.hand_tiles else None)
        return AdviceResult(
            discard=discard,
            reason=f"（LLM 暂不可用）算法推荐：向听数 {state.shanten}，"
                   f"有效进张 {len(state.effective_tiles)} 张",
            confidence=0.6
        )
