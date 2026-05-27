// content.js - 注入到雀魂页面
// 负责接收建议并在页面上显示浮层

(function () {
  'use strict';

  // ==================== 浮层创建 ====================

  let overlay = null;
  let isDragging = false;
  let dragOffsetX = 0, dragOffsetY = 0;

  function createOverlay() {
    if (overlay) return;

    overlay = document.createElement('div');
    overlay.id = 'mahjong-ai-overlay';
    overlay.innerHTML = `
      <div class="mai-header">
        <span class="mai-title">雀魂AI</span>
        <div class="mai-controls">
          <button class="mai-btn mai-minimize" title="最小化">—</button>
          <button class="mai-btn mai-close" title="关闭">✕</button>
        </div>
      </div>
      <div class="mai-body">
        <div class="mai-status">等待识别...</div>
        <div class="mai-advice hidden">
          <div class="mai-recommend">
            <span class="mai-label">推荐出牌</span>
            <span class="mai-tile" id="mai-tile-recommend"></span>
          </div>
          <div class="mai-reason" id="mai-reason"></div>
          <div class="mai-hand">
            <span class="mai-label">当前手牌</span>
            <div class="mai-tiles-row" id="mai-hand-tiles"></div>
          </div>
          <div class="mai-shanten">
            <span class="mai-label">向听数</span>
            <span id="mai-shanten-num"></span>
          </div>
        </div>
      </div>
      <div class="mai-footer">
        <span id="mai-timestamp"></span>
      </div>
    `;

    document.body.appendChild(overlay);
    bindOverlayEvents();
  }

  function bindOverlayEvents() {
    const header = overlay.querySelector('.mai-header');
    const closeBtn = overlay.querySelector('.mai-close');
    const minimizeBtn = overlay.querySelector('.mai-minimize');
    const body = overlay.querySelector('.mai-body');

    // 拖拽
    header.addEventListener('mousedown', (e) => {
      isDragging = true;
      dragOffsetX = e.clientX - overlay.offsetLeft;
      dragOffsetY = e.clientY - overlay.offsetTop;
    });
    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      overlay.style.left = (e.clientX - dragOffsetX) + 'px';
      overlay.style.top = (e.clientY - dragOffsetY) + 'px';
    });
    document.addEventListener('mouseup', () => { isDragging = false; });

    // 关闭/最小化
    closeBtn.addEventListener('click', () => { overlay.style.display = 'none'; });
    minimizeBtn.addEventListener('click', () => {
      body.classList.toggle('hidden');
    });
  }

  // ==================== 消息监听 ====================

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'SHOW_ADVICE') {
      showAdvice(msg.payload);
    } else if (msg.type === 'HIDE_OVERLAY') {
      if (overlay) overlay.style.display = 'none';
    } else if (msg.type === 'SHOW_OVERLAY') {
      if (overlay) overlay.style.display = 'block';
      else createOverlay();
    }
  });

  // ==================== 展示建议 ====================

  function showAdvice(data) {
    if (!overlay) createOverlay();
    overlay.style.display = 'block';

    const statusEl = overlay.querySelector('.mai-status');
    const adviceEl = overlay.querySelector('.mai-advice');

    if (!data || data.error) {
      statusEl.textContent = data?.error || '识别失败，请重试';
      adviceEl.classList.add('hidden');
      return;
    }

    statusEl.textContent = '';
    adviceEl.classList.remove('hidden');

    // 推荐出牌
    const recommendEl = document.getElementById('mai-tile-recommend');
    if (data.advice?.discard) {
      recommendEl.textContent = formatTile(data.advice.discard);
      recommendEl.dataset.tile = data.advice.discard;
    }

    // 推理理由
    const reasonEl = document.getElementById('mai-reason');
    reasonEl.textContent = data.advice?.reason || '';

    // 手牌展示（花色分色）
    const handEl = document.getElementById('mai-hand-tiles');
    handEl.innerHTML = '';
    (data.hand_tiles || []).forEach(tile => {
      const span = document.createElement('span');
      span.className = 'mai-tile-item';
      if (data.advice?.discard === tile) span.classList.add('mai-tile-highlight');
      // 花色分色
      const suit = getTileSuit(tile);
      span.classList.add(`mai-tile-${suit}`);
      span.textContent = formatTile(tile);
      span.title = tile;
      handEl.appendChild(span);
    });

    // 向听数
    const shantenEl = document.getElementById('mai-shanten-num');
    if (data.shanten !== undefined) {
      shantenEl.textContent = data.shanten === 0 ? '听牌！' : data.shanten;
      shantenEl.style.color = data.shanten === 0 ? '#e24b4a' : 'inherit';
    }

    // 时间戳
    document.getElementById('mai-timestamp').textContent =
      new Date().toLocaleTimeString('zh-CN');
  }

  // ==================== 牌面展示转换 ====================

  const TILE_DISPLAY = {
    // 万子
    '1m': '1万', '2m': '2万', '3m': '3万', '4m': '4万', '5m': '5万',
    '6m': '6万', '7m': '7万', '8m': '8万', '9m': '9万',
    // 饼子
    '1p': '1饼', '2p': '2饼', '3p': '3饼', '4p': '4饼', '5p': '5饼',
    '6p': '6饼', '7p': '7饼', '8p': '8饼', '9p': '9饼',
    // 索子
    '1s': '1索', '2s': '2索', '3s': '3索', '4s': '4索', '5s': '5索',
    '6s': '6索', '7s': '7索', '8s': '8索', '9s': '9索',
    // 字牌
    '1z': '东', '2z': '南', '3z': '西', '4z': '北',
    '5z': '中', '6z': '发', '7z': '白'
  };

  function formatTile(tile) {
    return TILE_DISPLAY[tile] || tile;
  }

  function getTileSuit(tile) {
    if (tile.endsWith('m')) return 'man';    // 万字
    if (tile.endsWith('p')) return 'pin';    // 筒子
    if (tile.endsWith('s')) return 'sou';    // 索子
    return 'honor';                          // 字牌
  }

  // ==================== 初始化 ====================

  // 页面加载完成后创建浮层（默认隐藏，等待 background 指令）
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createOverlay);
  } else {
    createOverlay();
  }

})();
