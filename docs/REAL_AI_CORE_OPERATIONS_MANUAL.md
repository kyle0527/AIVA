# AIVA 真實 AI 核心建制操作手冊 (可直接照做)

更新日期: 2025-11-10  
適用對象: 需要在 AIVA 中新建/替換「真實」AI 核心 (非雜湊偽 AI) 的工程師  
目標時長: 1~2 小時完成一個可推理、可管理權重、可相容舊 API 的新核心

---

## 目錄

1. 成功標準與介面合約 (Contract)
2. AI 能力與規格評估方法（立項前必讀）
3. 系統需求
4. 專案結構與放置位置
5. 流程總覽 Checklist
6. 步驟詳解 (逐步可落地)
7. 常見錯誤與排障
8. 性能優化建議
9. 安全與供應鏈
10. 參數—大小速查表 (float32)
11. 快速交付腳本 (選擇性)
12. 進階：與 RAG/多代理整合
13. 最小模板清單 (建立新核心時複製)

## 1. 成功標準與介面合約 (Contract)

建立的核心必須同時滿足以下條件：

- 真實神經網路 (PyTorch) + state_dict 權重，權重文件大小 ≈ 參數數 × 4 Bytes (float32)
- 具備推理與可訓練能力 (forward + optimizer/criterion)
- 提供穩定 API：
  - Model: `forward(x: Tensor[batch, in_dim]) -> Tensor[batch, out_dim]`
  - Engine: `generate_decision(task: str, context: str) -> dict`
  - Adapter: `forward(x: NDArray) -> NDArray` (softmax 機率)
- 權重管理：可版本化保存/載入、SHA256 校驗、備份與回復
- 兼容現有呼叫端：可作為 `ScalableBioNet` 的替代品接入 (不破壞上層)

成功檢驗：
- 權重檔案 > 10MB，能載入、能推理，輸出為機率分佈 (softmax)
- 驗證腳本可打印參數數、推理延遲、輸出分佈、權重檔案大小

---

## 2. AI 能力與規格評估方法（立項前必讀）

本章提供「如何分析與評估需要的 AI 能力與規格」的可操作流程，幫你在新核心立項前快速得到清晰結論與可落地的規格書。

### 2.1 使用情境盤點（Use-case Inventory）

- 定義輸入型態：文字/結構化向量/代碼嵌入/多模態（文+圖）
- 定義輸出型態：
    - 類別選擇（例如工具選擇/風險等級）
    - 連續值（分數/信心）
    - 排序/Top-K 候選
    - 複合輸出（主/輔輸出，多頭）
- 操作頻率與路徑：交互式（單次）、批次、流式、實時（<100ms）
- 數據約束：資料量、標註可得性、敏感/隱私、更新頻率

產出物：用例清單（UC-001/002/…）＋每個用例的 I/O、頻率、SLA、資料可得性。

### 2.2 能力矩陣（Capability Matrix）

為每個用例列出所需能力並打分（1=低、3=中、5=高）：

- 語義理解（文本/代碼）
- 長上下文處理（> 2k tokens 或大型特徵拼接）
- 工具選擇/規劃（多步推理）
- 不確定性處理/抗幻覺（可靠性）
- 可學習性（需持續微調或在線學習）
- 可解釋性（需提供理由/特徵重要性）

聚合後，挑選 Top 能力作為模型設計的第一優先級。

### 2.3 指標與 SLA 目標（Metrics & SLAs）

為每個用例設定可量化的成功條件：

- 準確度目標：分類（Accuracy/F1/ROC-AUC）、排序（NDCG/MRR）、回歸（MAE/RMSE）
- 延遲（P50/P95）：交互式 < 150ms、實時 < 100ms、批次可放寬
- 吞吐（QPS/批次大小）：例如 ≥ 50 QPS（CPU）或 ≥ 500 QPS（GPU/批次）
- 穩定度：錯誤率 < 0.1%、冷啟動 < 1s
- 資源邊界：CPU/記憶體/VRAM 上限、成本上限

將指標納入驗收門檻（Acceptance Gates），與產品/安全共同簽核。

### 2.4 參數與算力估算（Sizing Heuristics）

對於本專案這類 512 維輸入的前饋網路（MLP），可用下列方法估算參數量與延遲：

- 參數量（parameters）近似：`Σ (in_i × out_i + out_i)`
- 權重檔大小（float32）：`params × 4 Bytes`（範例：5M params ≈ 19.1 MB）
- FLOPs 估算：每層約 `2 × in × out`（乘加），總 FLOPs 為各層相加
- 單次推理延遲（CPU 粗估）：`FLOPs /（有效每秒運算量）`，再考慮記憶體訪存與 Python 開銷

常用等級建議（僅作啟始點）：

- 規則路由/簡單工具選擇：0.5M–5M 參數
- 多工具選擇/中等語義：3M–20M 參數
- 高複雜語義/長上下文：>20M 或改用向量檢索 + 專家路由

若為代碼理解/生成，建議以外部大模型（LLM）處理語義，核心網路專注於「策略/選擇/路由」，有效控制參數與延遲。

### 2.5 記憶體與部署預算（Memory/Deployability）

- 權重常駐記憶體 ≈ 權重大小（float32）
- 啟用半精度/量化：FP16（×0.5）/ INT8（×0.25）可明顯降記憶體與提升吞吐
- GPU 規劃：預留 20–30% VRAM 作為啟動與特徵開銷
- 容器化：固定依賴版本；將權重掛載為只讀 Volume；開啟健康檢查

### 2.6 數據品質與覆蓋（Data Readiness）

最常見瓶頸不是模型，而是資料：

- 標註一致性與噪音率（理想 < 5–10%）
- 類別分佈是否長尾？需重加權/重採樣
- 訓練/驗證/測試切分（時間切分以避免洩漏）
- 覆蓋代表性場景與極端案例（邊界/錯誤）

若資料不足，先做：資料合成、弱標註、規則生金標、蒐集管線。

### 2.7 評估方案設計（Evaluation Plan）

- 線下：標準資料集指標 + 針對用例的業務指標
- 線上：A/B 測試、Shadow 部署、離線回放
- 可靠性：抗幻覺檢查（閾值、置信區間）、一致性測試、對抗樣本
- 可解釋：重要特徵分析、錯案分析面板

### 2.8 升級決策樹（When to Scale What）

先後順序建議：

1) 改善資料與特徵（增量收益最高）
2) 調整損失/訓練策略（學習率、正則、early-stop）
3) 增加輸入特徵或改網路拓樸（多頭/分支）
4) 才考慮加大參數（成本/延遲上升）
5) 需要跨語義能力 → 以檢索/RAG/外部LLM 輔助

量化與裁剪：INT8 量化可在少量精度損失下換高吞吐；針對冷路徑做蒸餾/裁剪以降成本。

### 2.9 風險、合規與安全

- 權重供應鏈安全：偏好 `state_dict` + `weights_only=True`；校驗 SHA256
- 敏感資料：匿名化/最小化蒐集；權重不得內嵌敏感樣本
- 審計：開啟推理日誌（含版本、雜湊、參數檔名）與可回放樣本
- 模型治理：標記版本/資料來源；出現退化可一鍵回滾

### 2.10 立項輸出模板（可複製）

- 用例清單＋I/O 定義＋頻率與 SLA
- 能力矩陣（1–5 分）與 Top 能力
- 指標目標（Accuracy/F1、P95 延遲、QPS…）
- 模型規格（in/out、層級、參數量、預估檔案大小）
- 資源預算（CPU/GPU/記憶體/成本）與部署方式
- 數據計畫（蒐集/標註/切分/合規）
- 評估方案（離線/線上/回滾方案）
- 風險清單與緩解措施

完成以上輸出，即可進入「第 5 章 流程總覽」的建模與實作環節。

---

## 3. 系統需求

- OS: Windows 10/11 (PowerShell 7+)
- Python: 3.10/3.11 (建議 venv)
- 依賴: `torch`, `numpy` (其餘依據專案 `requirements.txt`)
- 硬體: CPU 皆可；若有 GPU (CUDA) 可自動使用

可選快速安裝 (PowerShell)：
```powershell
# 建議在專案根目錄執行
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# 若未包含 torch，請依官方指引安裝對應版本
```

---

## 4. 專案結構與放置位置

建議將新核心放在：`services/core/aiva_core/ai_engine/`

- `your_neural_core.py`    新模型與決策引擎 (對標 `real_neural_core.py`)
- `your_adapter.py`        向後相容適配器 (對標 `real_bio_net_adapter.py`)
- `weight_manager.py`      權重管理共用 (直接使用現有)
- 權重放置：`/weights/models/your_model_xxx.pth`

---

## 5. 流程總覽 Checklist

1) 參數預算與網路設計  → 決定 in/out 維度與隱藏層、估算參數與檔案大小  
2) 建立模型類 (Model)    → `forward()`、`save_weights()`、`load_weights()`  
3) 建立決策引擎 (Engine) → `generate_decision()` + 真實向量化 `encode_input()`  
4) 權重管理串接          → 使用 `AIWeightManager` 做版本化/校驗/備份  
5) 建立適配器 (Adapter)   → 與 `ScalableBioNet` 等價 API, 輸出 softmax 機率  
6) 整合核心              → 在 `bio_neuron_core.py` 用條件切換到新核心  
7) 測試驗證              → 參數/檔案大小/機率輸出/推理延遲  
8) 文檔與交付            → 記錄配置、保存範例權重、提交到 git

---

## 6. 步驟詳解 (逐步可落地)

### Step 1. 參數預算與網路設計

- 設計原則：
  - `params ≈ Σ (in_i × out_i + out_i)`, 權重檔大小 ≈ `params × 4 Bytes`
  - 5M 參數 ≈ 19.1 MB；3.7M 參數 ≈ 14.3 MB
- 範例設計 (輸入 512 維，輸出 128)：
  - 512→2048→1024→512→128，ReLU + Dropout(0.1~0.2)

### Step 2. 建立模型類 (YourAICore)

關鍵點：PyTorch `nn.Module`、`forward()`、權重 save/load（支援 5M 或 3.7M 兩種配置）。

範例骨架：
```python
# services/core/aiva_core/ai_engine/your_neural_core.py
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

class YourAICore(nn.Module):
    def __init__(self, input_size=512, hidden=[2048,1024,512], output_size=128):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, output_size)]
        self.net = nn.Sequential(*layers)
        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save_weights(self, path: str) -> None:
        state = { 'model_state_dict': self.state_dict(), 'total_params': self.total_params }
        torch.save(state, path)

    def load_weights(self, path: str) -> None:
        if Path(path).exists():
            state = torch.load(path, map_location='cpu')
            self.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
```

### Step 3. 建立決策引擎 (YourDecisionEngine)

- 功能：文本向量化 → 前向推理 → softmax 機率 → 產生結構化決策
- 重點：`encode_input()` 不可用 MD5/雜湊作為主決策，應用真實特徵或外部 embedding

骨架：
```python
import numpy as np, torch, torch.nn.functional as F

class YourDecisionEngine:
    def __init__(self, weights_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.core = YourAICore().to(self.device)
        if weights_path: self.core.load_weights(weights_path)

    def encode_input(self, text: str) -> torch.Tensor:
        v = np.zeros(512, dtype=np.float32)
        t = text.lower().strip()
        for i, ch in enumerate(t[:500]):
            v[i % 500] += ord(ch)/255.0
        v[510] = len(t)/1000.0
        v[511] = (sum(map(ord, t))/max(1,len(t)))/255.0
        return torch.from_numpy(v).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def generate_decision(self, task: str, context: str = "") -> dict:
        x = self.encode_input(f"{task} {context}")
        logits = self.core(x)
        probs = F.softmax(logits, dim=1)
        conf, idx = float(probs.max()), int(probs.argmax())
        return {
            'decision': task,
            'confidence': conf,
            'decision_index': idx,
            'is_real_ai': True
        }
```

### Step 4. 串接權重管理 (AIWeightManager)

直接使用現有 `weight_manager.py`：
```python
from .weight_manager import get_weight_manager

wm = get_weight_manager()
# 保存
path, meta = wm.save_model_weights(model=self.core, model_name='your_core')
# 載入 (最新/特定版本)
meta = wm.load_model_weights(model=self.core, model_name='your_core', version='latest')
```

建議：
- 所有訓練/導入後都用 `wm.save_model_weights()` 產生帶 metadata 的標準檔
- 部署時用 `wm.load_model_weights()` + SHA256 校驗

### Step 5. 建立向後相容適配器 (RealScalableBioNet 等價)

- 目標：保留舊 API (如 `forward(np.ndarray) -> np.ndarray`)
- 輸出需為 softmax 機率分布，以保持上層決策流程不變

骨架：
```python
# services/core/aiva_core/ai_engine/your_adapter.py
import numpy as np, torch
from numpy.typing import NDArray
from .your_neural_core import YourAICore

class YourScalableBioNet:
    def __init__(self, input_size: int, num_tools: int, weights_path: str | None = None):
        self.num_tools = num_tools
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.core = YourAICore(input_size=input_size, output_size=num_tools).to(self.device)
        if weights_path: self.core.load_weights(weights_path)

    @torch.no_grad()
    def forward(self, x: NDArray) -> NDArray:
        xt = torch.from_numpy(x.astype(np.float32))
        if xt.dim()==1: xt = xt.unsqueeze(0)
        xt = xt.to(self.device)
        probs = torch.softmax(self.core(xt), dim=1).cpu().numpy()
        return probs
```

### Step 6. 整合 `bio_neuron_core.py`

- 在初始化路徑中嘗試使用新核心 (`YourScalableBioNet`)；
- 若 `import torch` 失敗或權重缺失，允許降級 (不建議在生產使用)

範例整合點：
```python
# 於 BioNeuronRAGAgent 或 ScalableBioNet 建構處替換
from .your_adapter import YourScalableBioNet
self.decision_core = YourScalableBioNet(input_size=512, num_tools=len(self.tools), weights_path='weights/models/your_core_latest.pth')
```

### Step 7. 測試與驗證

驗證腳本要點：
- 列印參數數 (`sum(p.numel())`)、推理時間、輸出機率分佈合計=1
- 檔案大小 > 10MB；若 5M 參數，約 19.1MB

最小測試：
```python
from services.core.aiva_core.ai_engine.your_neural_core import YourAICore
import torch, time
m = YourAICore(); x=torch.randn(1,512)
start=time.time(); y=m(x); dt=(time.time()-start)*1000
print('params:', sum(p.numel() for p in m.parameters()))
print('latency(ms):', round(dt,3))
print('out_shape:', tuple(y.shape))
```

### Step 8. 文檔與交付

- 在 `docs/` 建立該核心的設計檔與使用說明 (本手冊可作模板)
- 保存一份可用權重與對應 metadata (.json)
- 提交 git：含程式、測試、文檔與權重 (若倉庫允許大檔)

---

## 7. 常見錯誤與排障

- `RuntimeError: size mismatch`：層輸入/輸出維度與權重不匹配 → 檢查網路結構或重新訓練/導出
- `Missing key(s) in state_dict`：新舊模型層名不同 → 保持層命名一致或手動過濾 key
- 權重載入但參數數不同：可能導出的 `state_dict` 不是當前網路 → 對齊網路定義，重新保存
- Windows 路徑空白/括號：PowerShell 需以雙引號包住路徑
- CUDA OOM：減少 batch size、層寬或改為 CPU
- 未安裝 Torch：先 `pip install torch` (依環境)

---

## 8. 性能優化建議

- 推理：`model.eval()`, `torch.no_grad()`、合併小請求為 batch、必要時開啟 `torch.autocast` (半精度)
- 訓練：AdamW、合適的 lr、監控 loss 曲線；必要時梯度裁剪
- CPU：設定 `torch.set_num_threads(n)` 以避免過度搶佔

---

## 9. 安全與供應鏈

- 優先使用 `weights_only=True` (PyTorch 2.6+) 安全載入，避免執行任意 pickle 物件
- 以 SHA256 校驗檔案完整性；啟用自動備份與保留策略
- 權重與 metadata 放入版本目錄，避免覆蓋

---

## 10. 參數—大小速查表 (float32)

- 1M params ≈ 3.81 MB  
- 3.7M params ≈ 14.3 MB  
- 5.0M params ≈ 19.1 MB

計算：`size_MB ≈ params × 4 / 1024 / 1024`

---

## 11. 快速交付腳本 (選擇性)

PowerShell 指令 (僅做文件化示例)：
```powershell
.\.venv\Scripts\Activate.ps1
pytest -q  # 如有測試
git add services/core/aiva_core/ai_engine/your_* docs/REAL_AI_CORE_OPERATIONS_MANUAL.md
git commit -m "feat(ai-core): add new real AI core with adapter and manual"
git push origin main
```

---

## 12. 進階：與 RAG/多代理整合

- RAG：以 `_create_real_input_vector()` 改善特徵，或替換為外部 embedding 模型
- 多輸出任務：可在核心提供 `forward_with_aux()` 返回主/輔輸出
- 多代理：在上層協調器中以 `decision_index` 選擇工具或路徑

---

## 13. 最小模板清單 (建立新核心時複製)

- 檔案
  - `services/core/aiva_core/ai_engine/your_neural_core.py`
  - `services/core/aiva_core/ai_engine/your_adapter.py`
  - (可選) `services/core/aiva_core/ai_engine/tests/test_your_core.py`
- 權重
  - `weights/models/your_core_vXXXXXXXX.pth`
  - `weights/metadata/your_core_vXXXXXXXX.json`
- 文檔
  - `docs/REAL_AI_CORE_OPERATIONS_MANUAL.md` (本檔作為模板)

---

如需我幫你以此手冊快速腳手架一個新的核心 (`YourAICore`) 並加上最小測試，告訴我你的輸入/輸出維度與目標參數量，我可以直接幫你落地。

---

##  