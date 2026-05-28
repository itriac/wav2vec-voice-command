#!/usr/bin/env python3
"""
wav2vec2 離線語音辨識伺服器

wav2vec 2.0 架構說明（Meta AI, 2020）：
  ┌─────────────────────────────────────────────────────────┐
  │  原始音訊 (16kHz PCM waveform)                          │
  │       ↓                                                 │
  │  Feature Encoder (7 層 1D CNN)                          │
  │  → 將波形壓縮成 20ms/幀的局部特徵向量                  │
  │       ↓                                                 │
  │  Quantization Module（訓練階段）                        │
  │  → Gumbel-softmax 將特徵離散化為「語音單元」           │
  │       ↓                                                 │
  │  Transformer Encoder（12~24 層）                        │
  │  → 建立長距離上下文表示（全域 attention）              │
  │       ↓                                                 │
  │  CTC Linear Head                                        │
  │  → 每幀輸出字符機率 (vocab_size)                       │
  │       ↓                                                 │
  │  CTC Greedy Decoding                                    │
  │  → 移除重複字符與 blank token → 文字序列               │
  └─────────────────────────────────────────────────────────┘

  本專案使用 fine-tuned 模型：
  jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
  （在 Common Voice 中文資料集上 fine-tune，支援普通話）
"""

import os
import io
import json
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np

# ── 全域狀態 ─────────────────────────────────────────────────────────────────
_processor = None
_model = None
_model_ready = False
_model_loading = False
_model_error = None

MODEL_ID = "facebook/wav2vec2-base-960h"  # ~360MB，英文，LibriSpeech 960h fine-tune
TARGET_SR = 16000  # wav2vec2 固定需要 16kHz


def load_model():
    """在背景執行緒中載入模型（首次執行時會下載 ~1.2GB）"""
    global _processor, _model, _model_ready, _model_loading, _model_error
    _model_loading = True
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, logging as hf_logging
        import torch

        # masked_spec_embed 僅預訓練階段使用，推理時不需要，隱藏此 MISSING 警告
        hf_logging.set_verbosity_error()

        print(f"[wav2vec] 載入處理器：{MODEL_ID}")
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

        print(f"[wav2vec] 載入模型：{MODEL_ID}")
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        _model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        print(f"[wav2vec] 模型就緒（執行於 {device}）")
        _model_ready = True
    except Exception as e:
        _model_error = str(e)
        print(f"[wav2vec] 載入失敗：{e}")
        traceback.print_exc()
    finally:
        _model_loading = False


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    接收原始 WAV bytes → 回傳辨識文字

    步驟：
      1. librosa.load() 讀取任意格式並重新取樣至 16kHz mono
      2. Wav2Vec2Processor 正規化並產生 input_values tensor
      3. 模型前向傳播取得 logits（shape: [1, time_frames, vocab_size]）
      4. torch.argmax → 每幀最可能的 token id
      5. processor.batch_decode → 合併 CTC 序列為文字
    """
    import torch
    import librosa

    audio_buffer = io.BytesIO(audio_bytes)
    # librosa 自動重新取樣並轉 mono
    speech, _ = librosa.load(audio_buffer, sr=TARGET_SR, mono=True)

    # 正規化：Processor 預期 float32 array，值域約 [-1, 1]
    inputs = _processor(
        speech,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )

    device = next(_model.parameters()).device
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = _model(input_values).logits  # [1, T, vocab]

    # CTC greedy decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = _processor.batch_decode(predicted_ids)[0]
    return transcription.strip()


# ── HTTP 處理器 ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} {fmt % args}")

    def send_json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_json(200, {
                "ready": _model_ready,
                "loading": _model_loading,
                "error": _model_error,
                "model": MODEL_ID,
            })
            return

        # 靜態檔案服務
        if path == "/" or path == "":
            path = "/index.html"

        file_path = os.path.join(os.path.dirname(__file__), path.lstrip("/"))
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            mime = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript",
                ".css": "text/css",
                ".json": "application/json",
            }.get(ext, "application/octet-stream")
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/transcribe":
            if not _model_ready:
                self.send_json(503, {"error": "模型尚未就緒"})
                return

            content_length = int(self.headers.get("Content-Length", 0))
            audio_bytes = self.rfile.read(content_length)

            try:
                text = transcribe_audio(audio_bytes)
                self.send_json(200, {"text": text})
            except Exception as e:
                traceback.print_exc()
                self.send_json(500, {"error": str(e)})
            return

        self.send_response(404)
        self.end_headers()


# ── 入口點 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser

    HOST, PORT = "127.0.0.1", 3001

    # 背景執行緒載入模型，不阻塞 HTTP 伺服器啟動
    threading.Thread(target=load_model, daemon=True).start()

    server = HTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"[server] 伺服器啟動：{url}")
    print("[server] 正在背景載入 wav2vec2 模型（首次需下載 ~1.2 GB）...")
    print("[server] Ctrl+C 停止")

    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 已停止")
