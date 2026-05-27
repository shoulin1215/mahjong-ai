// background.js - Service Worker
// 负责定时截图并发送给 Vision Service

const CONFIG = {
  CAPTURE_INTERVAL_MS: 1500,       // 截图间隔（毫秒）
  VISION_SERVICE_URL: 'http://localhost:8000/analyze',
  ENABLED: false,
  MAHJONG_URLS: [
    'mahjongsoul.game.yo-star.com',
    'game.mahjongsoul.com'
  ]
};

let captureTimer = null;
let lastHandHash = '';  // 用于检测手牌变化，避免重复推理
let mahjongTabId = null; // 固定追踪的雀魂标签页 ID

// ==================== 消息监听 ====================

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'START_CAPTURE') {
    startCapture();
    sendResponse({ ok: true });
  } else if (msg.type === 'STOP_CAPTURE') {
    stopCapture();
    sendResponse({ ok: true });
  } else if (msg.type === 'GET_STATUS') {
    sendResponse({ enabled: CONFIG.ENABLED, tabId: mahjongTabId });
  } else if (msg.type === 'UPDATE_CONFIG') {
    Object.assign(CONFIG, msg.payload);
    sendResponse({ ok: true });
  }
  return true;  // 保持消息通道异步开放
});

// ==================== 标签页追踪 ====================

/**
 * 查找雀魂标签页并缓存 tabId。
 * 如果缓存的标签已关闭，自动重新查找。
 */
async function findMahjongTab() {
  // 优先使用缓存的 tabId
  if (mahjongTabId !== null) {
    try {
      const tab = await chrome.tabs.get(mahjongTabId);
      if (tab && !tab.discarded) return tab;
    } catch {
      // 标签已关闭，清除缓存
      mahjongTabId = null;
    }
  }

  // 查找所有包含雀魂 URL 的标签（不限制为 active）
  const tabs = await chrome.tabs.query({});
  const mjTab = tabs.find(t =>
    t.url && CONFIG.MAHJONG_URLS.some(u => t.url.includes(u))
  );

  if (mjTab) {
    mahjongTabId = mjTab.id;
    console.log('[雀魂AI] 已锁定雀魂标签页:', mjTab.id, mjTab.url);
    return mjTab;
  }
  return null;
}

// 监听标签关闭事件，清除缓存
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === mahjongTabId) {
    console.log('[雀魂AI] 雀魂标签页已关闭');
    mahjongTabId = null;
  }
});

// ==================== 截图逻辑 ====================

function startCapture() {
  if (captureTimer) return;
  CONFIG.ENABLED = true;
  captureTimer = setInterval(doCapture, CONFIG.CAPTURE_INTERVAL_MS);
  console.log('[雀魂AI] 开始截图捕获');
}

function stopCapture() {
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
  CONFIG.ENABLED = false;
  mahjongTabId = null;
  console.log('[雀魂AI] 已停止截图捕获');
}

async function doCapture() {
  try {
    // 查找雀魂标签页（不限制为 active，后台标签也能截）
    const tab = await findMahjongTab();
    if (!tab) return;

    // 截图（captureVisibleTab 需要目标窗口可见/最近聚焦）
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
      format: 'jpeg',
      quality: 85
    });

    // 发送给 Vision Service
    const result = await callVisionService(dataUrl, tab.id);
    if (!result) return;

    // 检测手牌是否变化
    const handHash = hashHand(result.hand_tiles);
    if (handHash === lastHandHash) return;
    lastHandHash = handHash;

    // 推理成功，通知 content script 展示建议
    try {
      await chrome.tabs.sendMessage(tab.id, {
        type: 'SHOW_ADVICE',
        payload: result
      });
    } catch {
      // content script 可能未注入，忽略
    }

  } catch (err) {
    console.error('[雀魂AI] 截图或分析失败:', err.message);
  }
}

// ==================== Vision Service 调用 ====================

async function callVisionService(dataUrl, tabId) {
  const base64 = dataUrl.split(',')[1];

  const resp = await fetch(CONFIG.VISION_SERVICE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64, tab_id: tabId })
  });

  if (!resp.ok) {
    console.warn('[雀魂AI] Vision Service 返回错误:', resp.status);
    return null;
  }

  return await resp.json();
}

// ==================== 工具函数 ====================

function hashHand(tiles) {
  if (!tiles || !tiles.length) return '';
  return [...tiles].sort().join(',');
}

// 扩展安装/启动时恢复配置
chrome.runtime.onStartup.addListener(() => {
  chrome.storage.local.get(['enabled', 'captureInterval'], (data) => {
    if (data.captureInterval) CONFIG.CAPTURE_INTERVAL_MS = data.captureInterval;
    if (data.enabled) startCapture();
  });
});
