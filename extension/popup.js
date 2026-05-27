// popup.js - 控制面板逻辑

const VISION_URL = 'http://localhost:8000/health';
const CONFIG_URL = 'http://localhost:8000/config/llm';

// 预设服务商配置
const PRESETS = {
  deepseek: {
    base_url: 'https://api.deepseek.com/v1',
    model: 'deepseek-chat',
    label: 'DeepSeek'
  },
  glm: {
    base_url: 'https://open.bigmodel.cn/api/paas/v4',
    model: 'glm-4',
    label: '智谱 GLM-4'
  },
  qwen: {
    base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'qwen-turbo',
    label: '通义千问'
  },
  moonshot: {
    base_url: 'https://api.moonshot.cn/v1',
    model: 'moonshot-v1-8k',
    label: 'Moonshot'
  },
  zhipu: {
    base_url: 'https://api.lingyiwanwu.com/v1',
    model: 'yi-lightning',
    label: '零一万物'
  }
};

// ==================== 初始化 ====================

document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  checkServiceHealth();
  bindEvents();
});

// ==================== 设置读写 ====================

async function loadSettings() {
  const data = await chrome.storage.local.get([
    'enabled', 'model', 'captureInterval',
    'customBaseUrl', 'customApiKey', 'customModel',
    'claudeApiKey'
  ]);

  const toggle = document.getElementById('toggle-main');
  if (data.enabled) toggle.classList.add('active');

  const modelSelect = document.getElementById('model-select');
  if (data.model) modelSelect.value = data.model;

  const intervalSelect = document.getElementById('interval-select');
  if (data.captureInterval) intervalSelect.value = String(data.captureInterval);

  // 恢复自定义配置
  if (data.customBaseUrl) document.getElementById('custom-base-url').value = data.customBaseUrl;
  if (data.customApiKey) document.getElementById('custom-api-key').value = data.customApiKey;
  if (data.customModel) document.getElementById('custom-model').value = data.customModel;
  if (data.claudeApiKey) document.getElementById('claude-api-key').value = data.claudeApiKey;

  // 根据模型选择显示对应配置区
  toggleConfigPanel(modelSelect.value);
}

async function saveSettings() {
  const enabled = document.getElementById('toggle-main').classList.contains('active');
  const model = document.getElementById('model-select').value;
  const captureInterval = parseInt(document.getElementById('interval-select').value);

  await chrome.storage.local.set({ enabled, model, captureInterval });

  // 通知 background 更新配置
  await chrome.runtime.sendMessage({
    type: 'UPDATE_CONFIG',
    payload: { CAPTURE_INTERVAL_MS: captureInterval, MODEL: model }
  });
}

// ==================== 配置面板切换 ====================

function toggleConfigPanel(value) {
  const customFields = document.getElementById('custom-fields');
  const claudeFields = document.getElementById('claude-fields');

  customFields.classList.remove('show');
  claudeFields.classList.remove('show');

  if (value === 'custom') {
    customFields.classList.add('show');
  } else if (value === 'claude') {
    claudeFields.classList.add('show');
  }
}

// ==================== 自定义模型保存 ====================

async function saveCustomConfig() {
  const baseUrl = document.getElementById('custom-base-url').value.trim();
  const apiKey = document.getElementById('custom-api-key').value.trim();
  const model = document.getElementById('custom-model').value.trim();

  if (!baseUrl) { showToast('请填写 API 地址', true); return; }
  if (!apiKey) { showToast('请填写 API Key', true); return; }
  if (!model) { showToast('请填写模型名称', true); return; }

  // 保存到 storage
  await chrome.storage.local.set({
    model: 'openai',
    customBaseUrl: baseUrl,
    customApiKey: apiKey,
    customModel: model
  });

  // 通知后端服务更新配置
  try {
    const resp = await fetch(CONFIG_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        backend: 'openai',
        base_url: baseUrl,
        api_key: apiKey,
        model: model
      })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    showToast('自定义模型配置已保存');
  } catch (err) {
    showToast('保存成功（后端未连接，重启后生效）');
  }

  // 更新下拉框和 LLM 状态
  document.getElementById('model-select').value = 'openai';
  toggleConfigPanel('openai');
  await checkServiceHealth();
}

async function saveClaudeConfig() {
  const apiKey = document.getElementById('claude-api-key').value.trim();
  if (!apiKey) { showToast('请填写 Anthropic API Key', true); return; }

  await chrome.storage.local.set({ model: 'claude', claudeApiKey: apiKey });

  try {
    const resp = await fetch(CONFIG_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend: 'claude', api_key: apiKey })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    showToast('Claude 配置已保存');
  } catch {
    showToast('保存成功（后端未连接，重启后生效）');
  }

  await checkServiceHealth();
}

// ==================== 预设填充 ====================

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;
  document.getElementById('custom-base-url').value = preset.base_url;
  document.getElementById('custom-model').value = preset.model;
  showToast(`已填充 ${preset.label} 配置`);
}

// ==================== Toast 提示 ====================

let toastTimer = null;
function showToast(msg, isError = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show' + (isError ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = 'toast'; }, 2000);
}

// ==================== 事件绑定 ====================

function bindEvents() {
  // 主开关
  const toggle = document.getElementById('toggle-main');
  toggle.addEventListener('click', async () => {
    toggle.classList.toggle('active');
    const enabled = toggle.classList.contains('active');
    await chrome.runtime.sendMessage({ type: enabled ? 'START_CAPTURE' : 'STOP_CAPTURE' });
    await saveSettings();
  });

  // 模型选择变更 → 切换配置面板
  document.getElementById('model-select').addEventListener('change', async (e) => {
    const val = e.target.value;
    toggleConfigPanel(val);
    // 选择预设模型时同步保存
    if (val !== 'custom') {
      await chrome.storage.local.set({ model: val });
      await saveSettings();
      await checkServiceHealth();
    }
  });

  // 截图间隔变更
  document.getElementById('interval-select').addEventListener('change', saveSettings);

  // 保存自定义配置
  document.getElementById('save-custom-btn').addEventListener('click', saveCustomConfig);

  // 保存 Claude 配置
  document.getElementById('save-claude-btn').addEventListener('click', saveClaudeConfig);

  // 预设标签
  document.querySelectorAll('.preset-tag').forEach(tag => {
    tag.addEventListener('click', () => applyPreset(tag.dataset.preset));
  });
}

// ==================== 服务健康检查 ====================

async function checkServiceHealth() {
  const visionDot = document.getElementById('vision-dot');
  try {
    const resp = await fetch(VISION_URL, { signal: AbortSignal.timeout(2000) });
    visionDot.classList.toggle('online', resp.ok);
    visionDot.classList.toggle('offline', !resp.ok);
  } catch {
    visionDot.classList.add('offline');
  }

  const llmDot = document.getElementById('llm-dot');
  const llmLabel = document.getElementById('llm-label');
  const data = await chrome.storage.local.get('model');
  const model = data.model || 'ollama';

  const modelNames = {
    ollama: '本地 Ollama',
    openai: 'OpenAI / 兼容模型',
    claude: 'Claude 3.5',
    custom: '自定义模型'
  };
  llmLabel.textContent = modelNames[model] || model;

  if (model === 'ollama') {
    try {
      const resp = await fetch('http://localhost:11434/api/tags', { signal: AbortSignal.timeout(2000) });
      llmDot.classList.toggle('online', resp.ok);
      llmDot.classList.toggle('offline', !resp.ok);
    } catch {
      llmDot.classList.add('offline');
    }
  } else {
    // 云端/自定义模型默认标绿
    llmDot.classList.add('online');
  }
}
