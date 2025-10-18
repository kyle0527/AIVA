#!/usr/bin/env python3
"""
AIVA 抗幻覺驗證模組
用途: 基於知識庫驗證 AI 生成的攻擊計畫，移除不合理步驟
基於: BioNeuron_模型_AI核心大腦.md 分析建議
"""

import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加 AIVA 模組路徑
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

class AntiHallucinationModule:
    """抗幻覺驗證模組 - 防止 AI 產生不合理的攻擊步驟"""
    
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base
        self.validation_history = []
        self.confidence_threshold = 0.7
        self.logger = self._setup_logger()
        
        # 已知攻擊技術分類 (基於 MITRE ATT&CK)
        self.known_techniques = {
            "reconnaissance": ["port_scan", "service_enum", "web_crawl"],
            "initial_access": ["phishing", "exploit_public", "brute_force"],
            "execution": ["command_injection", "script_execution", "malware"],
            "persistence": ["account_creation", "scheduled_task", "service_install"],
            "privilege_escalation": ["exploit_elevation", "token_manipulation"],
            "defense_evasion": ["obfuscation", "disable_security", "masquerade"],
            "credential_access": ["credential_dump", "keylogging", "password_crack"],
            "discovery": ["system_info", "network_discovery", "process_enum"],
            "collection": ["data_collection", "screen_capture", "keylog_capture"],
            "exfiltration": ["data_transfer", "encrypted_channel", "physical_media"]
        }
        
    def _setup_logger(self):
        """設置日誌記錄器"""
        logger = logging.getLogger("AntiHallucination")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def validate_attack_plan(self, attack_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        驗證整個攻擊計畫，移除明顯不合理的步驟
        
        Args:
            attack_plan: 包含攻擊步驟的計畫字典
            
        Returns:
            經過驗證和清理的攻擊計畫
        """
        self.logger.info(f"🔍 開始驗證攻擊計畫: {attack_plan.get('name', 'Unknown')}")
        
        if not attack_plan.get('steps'):
            self.logger.warning("⚠️  攻擊計畫缺少步驟，可能是幻覺")
            return attack_plan
        
        original_steps = len(attack_plan['steps'])
        validated_steps = []
        removed_steps = []
        
        for i, step in enumerate(attack_plan['steps']):
            validation_result = self._validate_single_step(step, i + 1)
            
            if validation_result['is_valid']:
                validated_steps.append(step)
            else:
                removed_steps.append({
                    'step': step,
                    'reason': validation_result['reason']
                })
                self.logger.warning(
                    f"🚫 移除可疑步驟 #{i+1}: {step.get('description', 'Unknown')} "
                    f"原因: {validation_result['reason']}"
                )
        
        # 更新計畫
        attack_plan['steps'] = validated_steps
        
        # 記錄驗證結果
        validation_summary = {
            'original_steps': original_steps,
            'validated_steps': len(validated_steps),
            'removed_steps': len(removed_steps),
            'removal_rate': len(removed_steps) / original_steps if original_steps > 0 else 0,
            'removed_details': removed_steps
        }
        
        self.validation_history.append(validation_summary)
        
        self.logger.info(
            f"✅ 計畫驗證完成: {original_steps} → {len(validated_steps)} 步驟 "
            f"(移除 {len(removed_steps)} 個可疑步驟)"
        )
        
        return attack_plan
    
    def _validate_single_step(self, step: Dict[str, Any], step_number: int) -> Dict[str, Any]:
        """
        驗證單個攻擊步驟的合理性
        
        Args:
            step: 攻擊步驟字典
            step_number: 步驟編號
            
        Returns:
            包含驗證結果的字典
        """
        
        # 1. 基本結構檢查
        if not isinstance(step, dict):
            return {'is_valid': False, 'reason': '步驟格式錯誤，非字典類型'}
        
        required_fields = ['action', 'description']
        for field in required_fields:
            if field not in step:
                return {'is_valid': False, 'reason': f'缺少必要欄位: {field}'}
        
        # 2. 技術分類驗證
        action = step.get('action', '').lower()
        if not self._is_known_technique(action):
            return {'is_valid': False, 'reason': f'未知攻擊技術: {action}'}
        
        # 3. 知識庫驗證 (如果有的話)
        if self.knowledge_base:
            knowledge_validation = self._validate_with_knowledge_base(step)
            if not knowledge_validation['is_valid']:
                return knowledge_validation
        
        # 4. 邏輯一致性檢查
        logic_validation = self._validate_step_logic(step, step_number)
        if not logic_validation['is_valid']:
            return logic_validation
        
        return {'is_valid': True, 'reason': '步驟驗證通過'}
    
    def _is_known_technique(self, action: str) -> bool:
        """檢查攻擊技術是否為已知技術"""
        action_lower = action.lower()
        
        for category, techniques in self.known_techniques.items():
            if action_lower in techniques:
                return True
        
        # 檢查常見變體
        common_variations = [
            'scan', 'enum', 'exploit', 'inject', 'dump', 'crack',
            'discover', 'collect', 'transfer', 'execute', 'escalate'
        ]
        
        return any(variation in action_lower for variation in common_variations)
    
    def _validate_with_knowledge_base(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """使用知識庫驗證步驟"""
        try:
            # 搜尋相關知識條目
            query = step.get('description', '') + ' ' + step.get('action', '')
            
            # 假設知識庫有 search 方法
            if hasattr(self.knowledge_base, 'search'):
                results = self.knowledge_base.search(query)
                
                if not results or len(results) == 0:
                    return {
                        'is_valid': False, 
                        'reason': '知識庫中無相關技術資料，可能是幻覺'
                    }
                
                # 檢查相關性分數
                if hasattr(results[0], 'score') and results[0].score < self.confidence_threshold:
                    return {
                        'is_valid': False,
                        'reason': f'知識庫匹配度過低: {results[0].score:.2f}'
                    }
            
            return {'is_valid': True, 'reason': '知識庫驗證通過'}
            
        except Exception as e:
            self.logger.error(f"知識庫驗證異常: {e}")
            return {'is_valid': True, 'reason': '知識庫驗證異常，預設通過'}
    
    def _validate_step_logic(self, step: Dict[str, Any], step_number: int) -> Dict[str, Any]:
        """驗證步驟邏輯合理性"""
        
        action = step.get('action', '').lower()
        description = step.get('description', '').lower()
        
        # 邏輯矛盾檢查
        contradictions = [
            # 初始步驟不應該是高級技術
            (step_number <= 2 and any(advanced in action for advanced in 
             ['privilege_escalation', 'persistence', 'exfiltration']), 
             '初始步驟使用高級攻擊技術，邏輯不合理'),
            
            # 描述與動作不符
            ('scan' in action and 'inject' in description, 
             '掃描動作與注入描述不符'),
            
            # 不可能的組合
            ('brute_force' in action and 'stealth' in description,
             '暴力破解與隱蔽操作矛盾'),
        ]
        
        for condition, reason in contradictions:
            if condition:
                return {'is_valid': False, 'reason': reason}
        
        return {'is_valid': True, 'reason': '邏輯驗證通過'}
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """獲取驗證統計資料"""
        if not self.validation_history:
            return {"總驗證次數": 0}
        
        total_original = sum(v['original_steps'] for v in self.validation_history)
        total_validated = sum(v['validated_steps'] for v in self.validation_history)
        total_removed = sum(v['removed_steps'] for v in self.validation_history)
        
        return {
            "總驗證次數": len(self.validation_history),
            "原始步驟總數": total_original,
            "驗證通過步驟": total_validated,
            "移除可疑步驟": total_removed,
            "整體移除率": f"{(total_removed / max(1, total_original)) * 100:.1f}%",
            "平均計畫大小": f"{total_original / len(self.validation_history):.1f} 步驟"
        }
    
    def export_validation_report(self, output_path: str = None) -> str:
        """匯出驗證報告"""
        if not output_path:
            output_path = f"anti_hallucination_report_{int(time.time())}.json"
        
        report = {
            "模組資訊": {
                "名稱": "AIVA 抗幻覺驗證模組",
                "版本": "1.0",
                "信心閾值": self.confidence_threshold
            },
            "驗證統計": self.get_validation_stats(),
            "驗證歷史": self.validation_history,
            "已知技術分類": self.known_techniques
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📊 驗證報告已輸出至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"報告輸出失敗: {e}")
            return ""

# 使用範例
def demo_anti_hallucination():
    """示範抗幻覺模組的使用"""
    
    print("🧠 AIVA 抗幻覺驗證模組示範")
    print("=" * 50)
    
    # 創建驗證模組
    validator = AntiHallucinationModule()
    
    # 測試攻擊計畫 (包含一些可疑步驟)
    test_plan = {
        "name": "Web 應用滲透測試",
        "target": "http://example.com",
        "steps": [
            {
                "action": "port_scan",
                "description": "掃描目標開放端口",
                "tool": "nmap"
            },
            {
                "action": "quantum_hack",  # 明顯的幻覺
                "description": "使用量子算法破解加密",
                "tool": "quantum_tool"
            },
            {
                "action": "web_crawl",
                "description": "爬取網站結構",
                "tool": "spider"
            },
            {
                "action": "privilege_escalation",  # 邏輯問題：太早使用高級技術
                "description": "提升系統權限",
                "tool": "exploit"
            },
            {
                "action": "sql_injection",
                "description": "測試 SQL 注入漏洞",
                "tool": "sqlmap"
            }
        ]
    }
    
    print(f"📋 原始計畫包含 {len(test_plan['steps'])} 個步驟")
    
    # 執行驗證
    validated_plan = validator.validate_attack_plan(test_plan)
    
    print(f"✅ 驗證後剩餘 {len(validated_plan['steps'])} 個步驟")
    
    # 顯示統計
    stats = validator.get_validation_stats()
    print("\n📊 驗證統計:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 匯出報告
    report_path = validator.export_validation_report()
    if report_path:
        print(f"\n📄 詳細報告: {report_path}")

if __name__ == "__main__":
    demo_anti_hallucination()