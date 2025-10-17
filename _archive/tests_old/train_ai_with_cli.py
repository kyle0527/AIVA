#!/usr/bin/env python3
"""
BioNeuronCore AI - 完整訓練和 CLI 配對系統

功能：
1. 啟用經驗學習
2. 執行 CLI 命令並學習
3. 持續優化神經網路
4. 記憶和持久化
"""
import sys
from pathlib import Path
import asyncio
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

class AITrainingSystem:
    """AI 訓練系統 - 配對 CLI 命令執行"""
    
    def __init__(self, codebase_path: str):
        """初始化訓練系統"""
        self.codebase_path = codebase_path
        self.agent = None
        self.training_data = {
            "start_time": datetime.now().isoformat(),
            "sessions": [],
            "total_executions": 0,
            "total_experiences": 0,
        }
        
    def initialize_ai(self):
        """初始化 AI 代理（啟用所有功能）"""
        print("="*70)
        print("初始化 BioNeuronCore AI（完整功能）")
        print("="*70)
        
        from services.core.aiva_core.ai_engine.bio_neuron_core import BioNeuronRAGAgent
        from services.core.aiva_core.ai_engine.cli_tools import get_all_tools
        
        # 創建數據目錄
        data_dir = Path("data/ai_training")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = data_dir / "aiva_experience.db"
        
        print(f"\n[配置]")
        print(f"  代碼庫: {self.codebase_path}")
        print(f"  數據庫: {db_path}")
        
        try:
            # 啟用完整功能
            self.agent = BioNeuronRAGAgent(
                codebase_path=self.codebase_path,
                enable_planner=True,      # 啟用計畫執行
                enable_tracer=True,       # 啟用執行追蹤
                enable_experience=True,   # 啟用經驗學習
                database_url=f"sqlite:///{db_path}"
            )
            
            # 加載完整 CLI 工具集並替換默認工具
            cli_tools_dict = get_all_tools()
            self.agent.tools = [
                {"name": tool_name, "instance": tool_obj}
                for tool_name, tool_obj in cli_tools_dict.items()
            ]
            self.agent.tool_map = {tool["name"]: tool for tool in self.agent.tools}
            
            # 重新創建決策核心以匹配新工具數量
            from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
            self.agent.decision_core = ScalableBioNet(
                self.agent.input_vector_size, 
                len(self.agent.tools)
            )
            
            print("\n✓ AI 完整初始化成功")
            print(f"  - 計畫執行器: {'✓' if self.agent.orchestrator else '✗'}")
            print(f"  - 執行追蹤: {'✓' if self.agent.execution_monitor else '✗'}")
            print(f"  - 經驗學習: {'✓' if self.agent.experience_repo else '✗'}")
            print(f"  - CLI 工具集: {len(self.agent.tools)} 個工具")
            for i, tool in enumerate(self.agent.tools, 1):
                print(f"    {i}. {tool['name']}")
            
        except Exception as e:
            print(f"\n⚠ 部分功能初始化失敗: {e}")
            print("  降級為基礎模式...")
            import traceback
            traceback.print_exc()
            
            # 降級：創建基礎 AI
            self.agent = BioNeuronRAGAgent(
                codebase_path=self.codebase_path,
                enable_planner=False,
                enable_tracer=False,
                enable_experience=False
            )
            
            # 仍然嘗試加載 CLI 工具
            try:
                cli_tools_dict = get_all_tools()
                self.agent.tools = [
                    {"name": tool_name, "instance": tool_obj}
                    for tool_name, tool_obj in cli_tools_dict.items()
                ]
                self.agent.tool_map = {tool["name"]: tool for tool in self.agent.tools}
                
                from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
                self.agent.decision_core = ScalableBioNet(
                    self.agent.input_vector_size, 
                    len(self.agent.tools)
                )
                print("✓ 基礎模式 AI 創建成功（包含 CLI 工具集）")
                print(f"  - 工具集: {len(self.agent.tools)} 個工具")
            except Exception as tool_err:
                print(f"⚠ 工具加載失敗: {tool_err}")
        
        return self.agent
    
    def simulate_cli_commands(self):
        """模擬 CLI 命令執行（配對）"""
        print("\n" + "="*70)
        print("CLI 命令配對執行")
        print("="*70)
        
        # 定義 CLI 命令和對應的 AI 任務
        cli_tasks = [
            {
                "cli": "aiva scan start https://example.com",
                "task": "掃描目標網站 example.com",
                "tool_expected": "ScanTrigger"
            },
            {
                "cli": "aiva detect sqli https://example.com/login --param username",
                "task": "檢測 SQL 注入漏洞在 login 頁面的 username 參數",
                "tool_expected": "SQLiDetector"
            },
            {
                "cli": "aiva analyze code services/core/",
                "task": "分析 services/core 目錄的代碼結構",
                "tool_expected": "CodeAnalyzer"
            },
            {
                "cli": "aiva read config pyproject.toml",
                "task": "讀取 pyproject.toml 配置文件",
                "tool_expected": "CodeReader"
            },
            {
                "cli": "aiva detect xss https://example.com/search --param q",
                "task": "檢測 XSS 漏洞在 search 頁面的 q 參數",
                "tool_expected": "XSSDetector"
            },
        ]
        
        session = {
            "session_id": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "executions": []
        }
        
        print(f"\n訓練會話: {session['session_id']}")
        print(f"任務數量: {len(cli_tasks)}\n")
        
        for i, task_info in enumerate(cli_tasks, 1):
            print(f"\n[任務 {i}/{len(cli_tasks)}]")
            print(f"CLI 命令: {task_info['cli']}")
            print(f"AI 任務: {task_info['task']}")
            
            # AI 執行任務
            result = self.agent.invoke(task_info['task'])
            
            # 記錄執行
            execution = {
                "cli_command": task_info['cli'],
                "ai_task": task_info['task'],
                "tool_used": result.get('tool_used'),
                "tool_expected": task_info['tool_expected'],
                "status": result.get('status'),
                "confidence": result.get('confidence', 0),
                "matched": result.get('tool_used') == task_info['tool_expected'],
                "timestamp": datetime.now().isoformat()
            }
            
            session['executions'].append(execution)
            self.training_data['total_executions'] += 1
            
            # 顯示結果
            match_icon = "✓" if execution['matched'] else "✗"
            print(f"  結果: {result.get('status')}")
            print(f"  工具: {result.get('tool_used')} {match_icon}")
            print(f"  信心度: {result.get('confidence', 0):.1%}")
            print(f"  配對: {'成功' if execution['matched'] else '失敗'}")
        
        self.training_data['sessions'].append(session)
        return session
    
    def train_from_experience(self):
        """從經驗訓練模型"""
        print("\n" + "="*70)
        print("從經驗訓練 AI 模型")
        print("="*70)
        
        if not hasattr(self.agent, 'train_from_experiences'):
            print("\n⚠ 經驗學習未啟用，跳過訓練")
            return None
        
        try:
            print("\n正在從經驗庫訓練...")
            result = self.agent.train_from_experiences(
                min_score=0.5,
                max_samples=1000
            )
            
            print(f"✓ 訓練完成")
            print(f"  狀態: {result.get('status')}")
            
            if result.get('status') == 'success':
                self.training_data['total_experiences'] = result.get('samples_used', 0)
            
            return result
            
        except Exception as e:
            print(f"⚠ 訓練過程發生錯誤: {e}")
            return None
    
    def save_training_report(self):
        """保存訓練報告"""
        print("\n" + "="*70)
        print("保存訓練報告")
        print("="*70)
        
        report_file = Path("data/ai_training/training_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 計算統計
        total_tasks = self.training_data['total_executions']
        matched_count = sum(
            1 for session in self.training_data['sessions']
            for exec in session['executions']
            if exec.get('matched', False)
        )
        
        success_rate = (matched_count / total_tasks * 100) if total_tasks > 0 else 0
        
        report = {
            **self.training_data,
            "end_time": datetime.now().isoformat(),
            "statistics": {
                "total_tasks": total_tasks,
                "matched_tasks": matched_count,
                "success_rate": f"{success_rate:.1f}%",
                "total_sessions": len(self.training_data['sessions']),
            },
            "neural_network": {
                "total_params": self.agent.decision_core.total_params,
                "input_size": self.agent.input_vector_size,
                "tools_count": len(self.agent.tools),
            },
            "memory": {
                "history_count": len(self.agent.history),
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 報告已保存: {report_file}")
        print(f"  文件大小: {report_file.stat().st_size} bytes")
        
        return report
    
    def display_summary(self, report):
        """顯示訓練總結"""
        print("\n" + "="*70)
        print("訓練總結")
        print("="*70)
        
        stats = report['statistics']
        
        print(f"\n[執行統計]")
        print(f"  總任務數: {stats['total_tasks']}")
        print(f"  配對成功: {stats['matched_tasks']}")
        print(f"  成功率: {stats['success_rate']}")
        print(f"  訓練會話: {stats['total_sessions']}")
        
        print(f"\n[神經網路]")
        nn_info = report['neural_network']
        print(f"  總參數: {nn_info['total_params']:,}")
        print(f"  輸入維度: {nn_info['input_size']}")
        print(f"  工具數量: {nn_info['tools_count']}")
        
        print(f"\n[記憶系統]")
        print(f"  歷史記錄: {report['memory']['history_count']}")
        print(f"  經驗數據: {self.training_data['total_experiences']}")
        
        print(f"\n[時間]")
        print(f"  開始: {report['start_time']}")
        print(f"  結束: {report['end_time']}")
        
        # 顯示每個會話的詳情
        print(f"\n[會話詳情]")
        for i, session in enumerate(self.training_data['sessions'], 1):
            matched = sum(1 for e in session['executions'] if e.get('matched', False))
            total = len(session['executions'])
            print(f"  會話 {i}: {matched}/{total} 配對成功 ({matched/total*100:.0f}%)")


async def main():
    """主訓練流程"""
    print("\n" + "="*70)
    print("BioNeuronCore AI - 完整訓練系統")
    print("配對 CLI 命令執行和持續學習")
    print("="*70)
    
    # 初始化訓練系統
    codebase = str(Path(__file__).parent)
    trainer = AITrainingSystem(codebase)
    
    # 步驟 1: 初始化 AI
    trainer.initialize_ai()
    
    # 步驟 2: CLI 命令配對執行
    session = trainer.simulate_cli_commands()
    
    # 步驟 3: 從經驗訓練
    trainer.train_from_experience()
    
    # 步驟 4: 保存報告
    report = trainer.save_training_report()
    
    # 步驟 5: 顯示總結
    trainer.display_summary(report)
    
    # 最終提示
    print("\n" + "="*70)
    print("🎉 訓練完成！")
    print("="*70)
    print("\n📊 關鍵成果:")
    print(f"  ✓ AI 可執行 CLI 命令")
    print(f"  ✓ 配對成功率: {report['statistics']['success_rate']}")
    print(f"  ✓ 記憶了 {report['memory']['history_count']} 次執行")
    print(f"  ✓ 神經網路: {report['neural_network']['total_params']:,} 參數")
    
    print("\n💡 下一步:")
    print("  1. 執行更多 CLI 命令增加訓練數據")
    print("  2. 調整神經網路參數提高配對準確度")
    print("  3. 啟用自動訓練循環")
    print("  4. 部署到生產環境")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
