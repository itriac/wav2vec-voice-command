# 離線語音指令系統 — wav2vec 2.0

完全離線的語音指令控制系統，以 Meta AI 的 wav2vec 2.0 模型作為語音辨識引擎，無需任何雲端 API 或網路連線。

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│  瀏覽器（index.html）                                            │
│                                                                 │
│  麥克風輸入                                                      │
│    │  MediaRecorder API                                         │
│    ▼                                                            │
│  VAD 靜音偵測  ──────────────────────────────────────────────── │
│  （RMS 門檻 + 持續幀數，過濾環境音與鍵盤聲）                      │
│    │  說話結束，自動停止錄音                                      │
│    ▼                                                            │
│  音訊轉換：WebM/Opus → 16kHz mono WAV                           │
│  （OfflineAudioContext 重新取樣）                                │
│    │  multipart/form-data POST                                  │
│    ▼                                                            │
│  HTTP /api/transcribe  ──────────────────────────────────────── │
│                                │                                │
└────────────────────────────────┼────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│  Python 伺服器（server.py）                                      │
│                                                                 │
│  PyAV 解碼 → 重新取樣至 16kHz float32                           │
│    │                                                            │
│    ▼                                                            │
│  Wav2Vec2Processor                                              │
│  正規化輸入（Z-score）→ input_values tensor                     │
│    │                                                            │
│    ▼                                                            │
│  Wav2Vec2ForCTC（facebook/wav2vec2-base-960h, 94M 參數）        │
│  Feature Encoder（7 層 1D CNN）                                 │
│    → 每 20ms 輸出 512-dim 局部特徵向量                          │
│  Transformer Encoder（12 層，全域 self-attention）               │
│    → 建立跨幀上下文表示                                         │
│  CTC Linear Head                                                │
│    → 每幀輸出字符機率 logits [1, T, vocab_size]                 │
│    │                                                            │
│    ▼                                                            │
│  CTC Greedy Decode                                              │
│  torch.argmax → batch_decode → 文字序列                         │
│    │                                                            │
│    ▼                                                            │
│  JSON {"text": "HELLO TESTER"}                                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│  狀態機（前端）                                                  │
│                                                                 │
│  IDLE → STANDBY → LISTEN_CMD → PROCESS → SPEAK_ACTION          │
│                                    ↓                            │
│                             AWAIT_CONFIRM → LISTEN_CONFIRM      │
│                                    ↓                ↓           │
│                               EXECUTE           RETRY           │
└─────────────────────────────────────────────────────────────────┘
```

---

## wav2vec 2.0 模型說明

**模型**：[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- 預訓練資料：LibriSpeech 960 小時英語語料
- 參數量：94M
- 輸入：16kHz mono float32 波形
- 輸出：英文字元序列（大寫，含空白）

### 推理流程

```
原始波形（16kHz PCM float32）
  │
  ▼
Feature Encoder — 7 層 1D CNN
  每層使用不同的 kernel size（10, 3, 3, 3, 3, 2, 2）與 stride
  壓縮比：320x（即每 320 個取樣點 = 20ms 輸出一幀）
  輸出：[T, 512] 局部特徵向量
  │
  ▼
Transformer Encoder — 12 層
  全域 Multi-Head Self-Attention（每層 8 heads）
  每幀可看到整段音訊的上下文（非因果遮罩）
  Relative Positional Encoding 取代絕對位置編碼
  輸出：[T, 768] 上下文特徵
  │
  ▼
CTC Linear Head
  Linear(768 → vocab_size=32)
  vocab：英文字母 A-Z + space + apostrophe + blank token
  每幀獨立輸出字符機率分佈
  │
  ▼
Greedy CTC Decode
  argmax 取每幀最高機率字符
  合併連續重複字符，移除 blank token
  輸出：文字字串（如 "HELLO TESTER"）
```

### 為何不用 MFCC？

傳統 ASR 需手工設計 MFCC 特徵，wav2vec 2.0 的 Feature Encoder 直接從原始波形端到端學習特徵，消除人工設計的偏差，在 LibriSpeech test-clean 上 WER 達 1.8%（fine-tuned）。

---

## 狀態機設計

系統採用九狀態有限狀態機（FSM），確保操作流程一致且可預測：

| 狀態 | 說明 | 觸發條件 |
|------|------|---------|
| `IDLE` | 系統待機 | 初始狀態 / 執行完成 / Esc |
| `STANDBY` | 等待喚醒詞 | 按下麥克風 |
| `LISTEN_CMD` | 等待指令 | 喚醒詞辨識成功 |
| `PROCESS` | wav2vec2 推理中 | VAD 偵測到語音結束 |
| `SPEAK_ACTION` | TTS 播報辨識結果 | 指令匹配成功 |
| `AWAIT_CONFIRM` | 等待確認 | TTS 播報完成 |
| `LISTEN_CONFIRM` | 等待 Yes/No | 按下麥克風 |
| `EXECUTE` | 執行指令 + 進度條 | 確認語音為 Yes |
| `RETRY` | 辨識失敗，重試 | 喚醒詞/指令/確認 均失敗 |

---

## VAD（語音活動偵測）

由於 wav2vec 2.0 為批次推理（非串流），系統在前端實作簡易 VAD 以自動決定何時停止錄音：

```
每 100ms 計算一次 RMS：
  rms = sqrt( mean( (sample - 128)² / 128² ) )

if rms > VAD_THRESHOLD:
  vadSpeechFrames++
  if vadSpeechFrames >= VAD_MIN_SPEECH_FRAMES:  # 連續 500ms 才算語音
    vadHasSpeech = true
else:
  vadSpeechFrames = 0
  if vadHasSpeech:
    等待 VAD_SILENCE_MS (900ms) 靜音後 → 自動停止錄音
```

| 參數 | 值 | 說明 |
|------|----|------|
| `VAD_THRESHOLD` | 0.06 | 過濾環境音與鍵盤聲（約 0.02）|
| `VAD_MIN_SPEECH_FRAMES` | 5 幀 | 需持續 500ms 才視為語音 |
| `VAD_SILENCE_MS` | 900ms | 說完後等待時間 |
| `VAD_MAX_MS` | 10000ms | 最長錄音時間 |

---

## 指令系統

### 可用指令（3 個）

| 指令 | 關鍵字範例 | 載具 | API |
|------|-----------|------|-----|
| Run Outdoor Script Number 1 | `outdoor script`, `outdoor` | UAV | `POST /api/v1/outdoor/script/1` |
| Run Indoor Script Number 1 | `indoor script number 1`, `indoor 1` | AMR | `POST /api/v1/indoor/script/1` |
| Run Indoor Script Number 2 | `indoor script number 2`, `indoor 2` | AMR | `POST /api/v1/indoor/script/2` |

### 喚醒詞容錯

wav2vec 2.0 對 "Hello Tester" 有多種可能的誤辨輸出，系統涵蓋以下變體：

```
hello tester / hello test / helo tester / yellow tester / hello texture
hello taster / hell tester / helo test / hello textur
ello tester / low tester / hello dester / hello des
tester / dester
```

### 確認語音

- **確認**：yes, correct, proceed, confirm, execute, go, sure, okay, ok, yep
- **取消**：no, cancel, abort, stop, negative

---

## 音效設計

| 時機 | 音效 | 說明 |
|------|------|------|
| 錄音開始 | 880Hz 短音（120ms）| 提示麥克風已開啟 |
| 喚醒詞偵測 | 600→900→1200Hz 三音（上升）| 系統已喚醒，準備聽指令 |
| AMR 執行 | 220/300Hz 方波交替（6 拍）| 機械脈衝聲模擬 AMR 啟動 |
| UAV 執行 | 140→520Hz 鋸齒波（漸升）| 旋翼加速聲模擬 UAV 起飛 |

---

## 快速開始

### 環境需求

- Python 3.8+
- [uv](https://docs.astral.sh/uv/)（套件管理）
- 支援 WebRTC 的現代瀏覽器（Chrome / Edge 建議）

### 啟動

```bash
# Windows
啟動伺服器.bat

# 或手動
uv sync
uv run server.py
```

首次啟動會從 Hugging Face 下載模型（約 360MB），之後快取於本機。

開啟瀏覽器：`http://127.0.0.1:3001`

### 使用流程

```
1. 等待「模型就緒」（綠燈）
2. 按下麥克風按鈕（聽到提示音）
3. 說出喚醒詞：Hello Tester
4. 系統自動偵測靜音，送出辨識
5. 聽到三音提示後，說出指令（如 outdoor script）
6. 確認動作後，說 Yes 執行 / No 取消
```

---

## 專案結構

```
wav2vec-voice-command/
├── server.py          # Python HTTP 伺服器 + wav2vec2 推理
├── index.html         # 前端（單頁應用，含狀態機 + VAD + 音效）
├── pyproject.toml     # 依賴宣告（uv）
├── 啟動伺服器.bat     # Windows 一鍵啟動
└── README.md          # 本文件
```

---

## 依賴套件

| 套件 | 用途 |
|------|------|
| `torch` | 模型推理（CPU / CUDA）|
| `transformers` | Wav2Vec2ForCTC、Wav2Vec2Processor |
| `av`（PyAV）| 音訊解碼（WAV/WebM/Opus/MP4）|
| `numpy` | 音訊陣列操作 |

---

## 與 voice-command-system 的差異

| | [voice-command-system](https://github.com/itriac/voice-command-system) | 本專案 |
|---|---|---|
| ASR 引擎 | Web Speech API（Google 雲端）| wav2vec 2.0（完全本機）|
| 網路需求 | 需要網路（STT）| **完全離線** |
| 辨識方式 | 串流即時辨識 | 批次推理（VAD 偵測結束後）|
| 雜訊處理 | 瀏覽器內建降噪 | 純 VAD 門檻過濾 |
| 語言 | 英文 | 英文（可換模型支援其他語言）|
| 架構 | 純前端（Node.js 靜態伺服器）| 前端 + Python 推理伺服器 |
