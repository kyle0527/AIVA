#!/usr/bin/env python3
"""
CLI 配對訓練 - 教會 AI 正確配對 CLI 命令與工具

通過監督學習訓練神經網路正確配對 CLI 命令
"""
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from services.core.aiva_core.ai_engine.bio_neuron_core import BioNeuronRAGAgent, ScalableBioNet
from services.core.aiva_core.ai_engine.cli_tools import get_all_tools


# 訓練數據集 - CLI 命令 -> 正確工具的配對
TRAINING_DATA = [
    # Scan 相關
    ("掃描目標網站 example.com", "ScanTrigger"),
    ("啟動掃描 https://test.com", "ScanTrigger"),
    ("開始網站掃描", "ScanTrigger"),
    ("scan target url", "ScanTrigger"),
    
    # SQLi 檢測
    ("檢測 SQL 注入漏洞在 login 頁面的 username 參數", "SQLiDetector"),
    ("測試 SQL injection", "SQLiDetector"),
    ("SQL 注入測試", "SQLiDetector"),
    ("檢查 SQL 漏洞", "SQLiDetector"),
    
    # XSS 檢測
    ("檢測 XSS 漏洞在 search 頁面的 q 參數", "XSSDetector"),
    ("測試跨站腳本攻擊", "XSSDetector"),
    ("XSS 漏洞檢測", "XSSDetector"),
    ("檢查 XSS", "XSSDetector"),
    
    # 代碼分析
    ("分析 services/core 目錄的代碼結構", "CodeAnalyzer"),
    ("分析代碼質量", "CodeAnalyzer"),
    ("檢查代碼結構", "CodeAnalyzer"),
    ("analyze code", "CodeAnalyzer"),
    
    # 文件讀取
    ("讀取 pyproject.toml 配置文件", "CodeReader"),
    ("讀取 README.md", "CodeReader"),
    ("查看文件內容", "CodeReader"),
    ("read file", "CodeReader"),
    
    # 文件寫入
    ("寫入配置到 config.json", "CodeWriter"),
    ("創建新文件", "CodeWriter"),
    ("修改文件內容", "CodeWriter"),
    ("write file", "CodeWriter"),
    
    # 報告生成
    ("生成掃描報告", "ReportGenerator"),
    ("創建分析報告", "ReportGenerator"),
    ("輸出檢測結果", "ReportGenerator"),
    ("generate report", "ReportGenerator"),
]


def train_neural_network(agent: BioNeuronRAGAgent, epochs: int = 100, learning_rate: float = 0.01):
    """訓練神經網路配對 CLI 命令"""
    
    print("="*70)
    print("訓練神經網路 - CLI 命令配對")
    print("="*70)
    
    # 獲取工具映射
    tool_names = [tool["name"] for tool in agent.tools]
    tool_to_idx = {name: idx for idx, name in enumerate(tool_names)}
    
    print(f"\n[訓練配置]")
    print(f"  訓練樣本: {len(TRAINING_DATA)}")
    print(f"  工具數量: {len(tool_names)}")
    print(f"  訓練輪數: {epochs}")
    print(f"  學習率: {learning_rate}")
    
    # 訓練歷史
    history = {
        "epochs": [],
        "losses": [],
        "accuracies": []
    }
    
    print(f"\n[開始訓練]")
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        
        # 隨機打亂訓練數據
        np.random.shuffle(TRAINING_DATA)
        
        for task_desc, correct_tool in TRAINING_DATA:
            # 簡單的文本嵌入（實際應該使用更好的編碼）
            # 這裡使用任務描述的哈希值生成偽隨機向量
            task_hash = hash(task_desc)
            np.random.seed(abs(task_hash) % (2**32))
            task_vector = np.random.randn(agent.input_vector_size).astype(np.float32)
            
            # 前向傳播
            raw_scores = agent.decision_core.forward(task_vector)
            
            # 創建目標（one-hot 編碼）
            target = np.zeros(len(tool_names), dtype=np.float32)
            target[tool_to_idx[correct_tool]] = 1.0
            
            # 計算損失（交叉熵）
            probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores))
            loss = -np.sum(target * np.log(probabilities + 1e-10))
            total_loss += loss
            
            # 檢查準確度
            predicted_idx = np.argmax(raw_scores)
            if predicted_idx == tool_to_idx[correct_tool]:
                correct += 1
            
            # 反向傳播（簡化版）
            # 計算輸出層梯度
            grad_output = probabilities - target
            
            # 更新輸出層權重
            agent.decision_core.fc2.weight -= learning_rate * np.outer(grad_output, agent.decision_core.fc2_input)
            agent.decision_core.fc2.bias -= learning_rate * grad_output
        
        # 計算平均損失和準確度
        avg_loss = total_loss / len(TRAINING_DATA)
        accuracy = correct / len(TRAINING_DATA)
        
        history["epochs"].append(epoch + 1)
        history["losses"].append(float(avg_loss))
        history["accuracies"].append(float(accuracy))
        
        # 每 10 輪顯示一次進度
        if (epoch + 1) % 10 == 0:
            print(f"  輪數 {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
    
    print(f"\n✓ 訓練完成")
    print(f"  最終損失: {history['losses'][-1]:.4f}")
    print(f"  最終準確度: {history['accuracies'][-1]:.2%}")
    
    return history


def test_trained_agent(agent: BioNeuronRAGAgent):
    """測試訓練後的代理"""
    
    print("\n" + "="*70)
    print("測試訓練後的 AI")
    print("="*70)
    
    test_cases = [
        ("掃描目標網站 example.com", "ScanTrigger"),
        ("檢測 SQL 注入漏洞在 login 頁面的 username 參數", "SQLiDetector"),
        ("檢測 XSS 漏洞在 search 頁面的 q 參數", "XSSDetector"),
        ("分析 services/core 目錄的代碼結構", "CodeAnalyzer"),
        ("讀取 pyproject.toml 配置文件", "CodeReader"),
    ]
    
    print(f"\n測試 {len(test_cases)} 個案例:\n")
    
    correct = 0
    for i, (task, expected_tool) in enumerate(test_cases, 1):
        result = agent.invoke(task)
        actual_tool = result.get('tool_used')
        confidence = result.get('confidence', 0)
        
        is_correct = actual_tool == expected_tool
        if is_correct:
            correct += 1
        
        status_icon = "✓" if is_correct else "✗"
        print(f"[測試 {i}] {status_icon}")
        print(f"  任務: {task}")
        print(f"  預期: {expected_tool}")
        print(f"  實際: {actual_tool}")
        print(f"  信心度: {confidence:.1%}")
        print()
    
    accuracy = correct / len(test_cases)
    print(f"測試準確度: {correct}/{len(test_cases)} = {accuracy:.1%}")
    
    return accuracy


def main():
    """主訓練流程"""
    
    print("\n" + "="*70)
    print("BioNeuronCore AI - CLI 命令配對訓練")
    print("="*70)
    print(f"\n時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 初始化 AI
    print("[步驟 1] 初始化 AI")
    codebase = str(Path(__file__).parent)
    
    agent = BioNeuronRAGAgent(
        codebase_path=codebase,
        enable_planner=False,
        enable_tracer=False,
        enable_experience=False
    )
    
    # 載入 CLI 工具
    cli_tools_dict = get_all_tools()
    agent.tools = [
        {"name": tool_name, "instance": tool_obj}
        for tool_name, tool_obj in cli_tools_dict.items()
    ]
    agent.tool_map = {tool["name"]: tool for tool in agent.tools}
    
    # 重新創建決策核心
    agent.decision_core = ScalableBioNet(
        agent.input_vector_size, 
        len(agent.tools)
    )
    
    print(f"✓ AI 初始化完成（{len(agent.tools)} 個工具）\n")
    
    # 2. 訓練神經網路
    print("[步驟 2] 訓練神經網路")
    history = train_neural_network(agent, epochs=100, learning_rate=0.01)
    
    # 3. 測試訓練結果
    print("[步驟 3] 測試訓練結果")
    test_accuracy = test_trained_agent(agent)
    
    # 4. 保存訓練結果
    print("\n[步驟 4] 保存訓練結果")
    
    output_dir = Path("data/ai_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存訓練歷史
    history_file = output_dir / "training_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"✓ 訓練歷史已保存: {history_file}")
    
    # 保存模型權重
    weights_file = output_dir / "model_weights.npz"
    np.savez(
        weights_file,
        fc1_weight=agent.decision_core.fc1.weight,
        fc1_bias=agent.decision_core.fc1.bias,
        fc2_weight=agent.decision_core.fc2.weight,
        fc2_bias=agent.decision_core.fc2.bias,
    )
    print(f"✓ 模型權重已保存: {weights_file}")
    
    # 保存訓練報告
    report = {
        "timestamp": datetime.now().isoformat(),
        "training_samples": len(TRAINING_DATA),
        "epochs": len(history["epochs"]),
        "final_loss": history["losses"][-1],
        "final_training_accuracy": history["accuracies"][-1],
        "test_accuracy": test_accuracy,
        "tools": [tool["name"] for tool in agent.tools],
        "neural_network": {
            "input_size": agent.input_vector_size,
            "hidden_size": 2048,
            "output_size": len(agent.tools),
            "total_params": agent.decision_core.total_params
        }
    }
    
    report_file = output_dir / "matching_training_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ 訓練報告已保存: {report_file}")
    
    # 最終總結
    print("\n" + "="*70)
    print("🎉 訓練完成！")
    print("="*70)
    print(f"\n📊 訓練結果:")
    print(f"  訓練樣本: {len(TRAINING_DATA)}")
    print(f"  訓練輪數: {len(history['epochs'])}")
    print(f"  最終損失: {history['losses'][-1]:.4f}")
    print(f"  訓練準確度: {history['accuracies'][-1]:.1%}")
    print(f"  測試準確度: {test_accuracy:.1%}")
    
    print(f"\n💾 已保存:")
    print(f"  {history_file}")
    print(f"  {weights_file}")
    print(f"  {report_file}")
    
    print(f"\n💡 下一步:")
    print(f"  1. 使用訓練好的模型執行 CLI 命令")
    print(f"  2. 收集更多訓練數據提高準確度")
    print(f"  3. 實際部署到生產環境")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
