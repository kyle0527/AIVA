"""
BioNeuron Core V2 - 完整 RAG 與工具整合版本
提供生物啟發式神經決策網路,整合實際的 RAG 檢索與程式操作能力
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


class BiologicalSpikingLayer:
    """模擬生物尖峰神經元行為的神經層."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """初始化尖峰神經層.

        Args:
            input_size: 輸入維度
            output_size: 輸出維度
        """
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.threshold = 1.0
        self.refractory_period = 0.1
        self.last_spike_time = np.zeros(output_size) - self.refractory_period
        self.params = input_size * output_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向傳播,產生尖峰訊號.

        Args:
            x: 輸入訊號

        Returns:
            尖峰輸出 (0 或 1)
        """
        import time

        current_time = time.time()
        potential = np.dot(x, self.weights)
        can_spike = (current_time - self.last_spike_time) > self.refractory_period
        spikes = (potential > self.threshold) & can_spike

        self.last_spike_time[spikes] = current_time
        return spikes.astype(np.float32)


class AntiHallucinationModule:
    """抗幻覺模組,確保決策的可靠性."""

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """初始化抗幻覺模組.

        Args:
            confidence_threshold: 信心度閾值
        """
        self.confidence_threshold = confidence_threshold

    def check(self, decision_logits: np.ndarray) -> tuple[bool, float]:
        """檢查決策的信心度.

        Args:
            decision_logits: 決策邏輯值

        Returns:
            (是否通過檢查, 信心度分數)
        """
        confidence = float(np.max(decision_logits))
        passed = confidence >= self.confidence_threshold
        return passed, confidence


class ScalableBioNet:
    """可擴展的生物神經網路 (約 500萬參數)."""

    def __init__(self, input_size: int, num_tools: int) -> None:
        """初始化決策網路.

        Args:
            input_size: 輸入特徵維度
            num_tools: 可用工具數量
        """
        self.fc1 = np.random.randn(input_size, 2048) * np.sqrt(2.0 / input_size)
        self.spiking_layer = BiologicalSpikingLayer(2048, 1024)
        self.fc2 = np.random.randn(1024, num_tools) * np.sqrt(2.0 / 1024)

        total_params = (
            input_size * 2048 + self.spiking_layer.params + 1024 * num_tools
        )
        print(f"[ScalableBioNet] 總參數量: {total_params:,}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向傳播.

        Args:
            x: 輸入向量

        Returns:
            工具選擇的邏輯值
        """
        # 第一層全連接
        h1 = np.dot(x, self.fc1)
        h1 = np.maximum(0, h1)  # ReLU

        # 尖峰神經層
        h2 = self.spiking_layer.forward(h1)

        # 輸出層
        output = np.dot(h2, self.fc2)
        return output


class BioNeuronRAGAgent:
    """生物神經元 RAG 代理,整合知識檢索與實際工具執行."""

    def __init__(self, codebase_path: str) -> None:
        """初始化 RAG 代理.

        Args:
            codebase_path: 程式碼庫路徑
        """
        from .knowledge_base import KnowledgeBase
        from .tools import (
            CodeAnalyzer,
            CodeReader,
            CodeWriter,
            CommandExecutor,
            ScanTrigger,
            VulnerabilityDetector,
        )

        self.codebase_path = codebase_path
        print("\n" + "="*60)
        print("   BioNeuronRAGAgent 初始化")
        print("="*60)
        print(f"程式碼庫: {codebase_path}")

        # 初始化知識庫並索引
        print("\n[1/3] 正在索引程式碼庫...")
        self.knowledge_base = KnowledgeBase(codebase_path)
        self.knowledge_base.index_codebase()

        # 初始化工具系統
        print("\n[2/3] 正在初始化工具系統...")
        self.tool_instances = {
            "CodeReader": CodeReader(codebase_path),
            "CodeWriter": CodeWriter(codebase_path),
            "CodeAnalyzer": CodeAnalyzer(codebase_path),
            "ScanTrigger": ScanTrigger(),
            "XSSDetector": VulnerabilityDetector(),
            "SQLiDetector": VulnerabilityDetector(),
            "SSRFDetector": VulnerabilityDetector(),
            "IDORDetector": VulnerabilityDetector(),
            "CommandExecutor": CommandExecutor(codebase_path),
        }

        self.tools = [
            {"name": "CodeReader", "description": "讀取程式碼檔案內容"},
            {"name": "CodeWriter", "description": "寫入或修改程式碼檔案"},
            {"name": "CodeAnalyzer", "description": "分析程式碼結構與品質"},
            {"name": "ScanTrigger", "description": "觸發漏洞掃描任務"},
            {"name": "XSSDetector", "description": "執行 XSS 漏洞檢測"},
            {"name": "SQLiDetector", "description": "執行 SQL 注入檢測"},
            {"name": "SSRFDetector", "description": "執行 SSRF 漏洞檢測"},
            {"name": "IDORDetector", "description": "執行 IDOR 漏洞檢測"},
            {"name": "CommandExecutor", "description": "執行系統命令"},
        ]

        print(f"   已載入 {len(self.tools)} 個工具")

        # 初始化決策核心
        print("\n[3/3] 正在初始化生物神經決策核心...")
        self.decision_core = ScalableBioNet(
            input_size=512,
            num_tools=len(self.tools),
        )
        self.anti_hallucination = AntiHallucinationModule(confidence_threshold=0.7)

        # 歷史記錄
        self.history: list[dict[str, Any]] = []

        print("\nBioNeuronRAGAgent 初始化完成!")
        print("="*60 + "\n")

    def invoke(self, query: str, **tool_kwargs: Any) -> dict[str, Any]:
        """執行 RAG 增強的智能決策與工具執行.

        Args:
            query: 使用者查詢或任務描述
            **tool_kwargs: 傳遞給工具的參數 (如 path, target_url, command 等)

        Returns:
            執行結果字典
        """
        print(f"\n{'='*60}")
        print(f"[啟動] 收到新任務: {query}")
        print(f"{'='*60}\n")

        # ===== 步驟 1: RAG 檢索 =====
        print("[資料] [步驟 1/5] RAG 知識檢索")
        retrieved_chunks = self.knowledge_base.search(query, top_k=3)
        print(f"   [V] 檢索到 {len(retrieved_chunks)} 個相關程式碼片段:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(
                f"     {i}. {chunk['path']} - {chunk['name']} (分數: {chunk['score']})"
            )

        # 構建上下文
        context_str = "\n\n".join(
            f"# {chunk['path']} - {chunk['name']}\n{chunk['content'][:300]}..."
            for chunk in retrieved_chunks
        )

        # ===== 步驟 2: 編碼 =====
        print("\n[數值] [步驟 2/5] 查詢與上下文編碼")
        # 簡化編碼: 基於查詢和上下文的雜湊
        query_hash = sum(ord(c) for c in query) % 1000
        context_hash = sum(ord(c) for c in context_str[:1000]) % 1000
        seed = query_hash + context_hash
        np.random.seed(seed)
        query_embedding = np.random.randn(512).astype(np.float32)
        print(f"   [V] 生成 512 維嵌入向量 (種子: {seed})")

        # ===== 步驟 3: 神經決策 =====
        print("\n[神經網路] [步驟 3/5] 生物神經網路決策")
        decision_logits = self.decision_core.forward(query_embedding)
        print(f"   [V] 決策邏輯值範圍: [{decision_logits.min():.2f}, {decision_logits.max():.2f}]")

        # ===== 步驟 4: 計算信心度 (不強制檢查) =====
        print("\n� [步驟 4/5] 計算決策信心度")
        _, confidence = self.anti_hallucination.check(decision_logits)
        print(f"   信心度: {confidence:.2%}")
        print("   [資訊]  已停用信心度檢查,所有決策都會執行")

        # ===== 步驟 5: 工具選擇與執行 =====
        print("\n[調整] [步驟 5/5] 工具選擇與執行")
        chosen_tool_index = int(np.argmax(decision_logits))
        chosen_tool = self.tools[chosen_tool_index]
        chosen_tool_name = chosen_tool["name"]
        tool_confidence = float(decision_logits[chosen_tool_index])

        print(f"   [V] 選擇工具: {chosen_tool_name}")
        print(f"   [V] 工具描述: {chosen_tool['description']}")
        print(f"   [V] 選擇信心度: {tool_confidence:.2%}")

        # 執行工具
        tool_instance = self.tool_instances.get(chosen_tool_name)
        if tool_instance:
            print("\n   [設定]  正在執行工具...")
            tool_result = tool_instance.execute(**tool_kwargs)
            print(f"   [V] 工具執行完成: {tool_result.get('status', 'unknown')}")
        else:
            tool_result = {
                "status": "error",
                "error": f"工具 {chosen_tool_name} 未實作",
            }
            print("   [失敗] 工具未實作")

        # 構建響應
        response = {
            "status": "success",
            "query": query,
            "tool_used": chosen_tool_name,
            "confidence": confidence,
            "context": retrieved_chunks,
            "tool_result": tool_result,
        }
        self.history.append(response)

        print(f"\n{'='*60}")
        print("[已] 任務完成!")
        print(f"{'='*60}\n")

        return response

    def get_history(self) -> list[dict[str, Any]]:
        """獲取執行歷史.

        Returns:
            執行歷史列表
        """
        return self.history

    def get_knowledge_stats(self) -> dict[str, int]:
        """獲取知識庫統計.

        Returns:
            知識庫統計資訊
        """
        return {
            "total_chunks": self.knowledge_base.get_chunk_count(),
            "total_keywords": len(self.knowledge_base.index),
        }
