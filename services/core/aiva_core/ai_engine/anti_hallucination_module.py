#!/usr/bin/env python3
"""AIVA 抗幻覺驗證模組
用途: 基於知識庫驗證 AI 生成的攻擊計畫，移除不合理步驟
基於: BioNeuron_模型_AI核心大腦.md 分析建議

改進內容:
- 增加知識庫回退機制 (Knowledge Base Fallback)
- 強化錯誤處理和恢復能力
- 添加備用驗證策略
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional
import warnings

# 添加 AIVA 模組路徑
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))


class AntiHallucinationModule:
    """抗幻覺驗證模組 - 防止 AI 產生不合理的攻擊步驟
    
    改進功能:
    - 知識庫回退機制：當主要知識庫不可用時使用備用策略
    - 多層驗證：結合規則驗證、知識庫驗證和統計驗證
    - 錯誤恢復：自動處理知識庫連線失敗等問題
    """

    def __init__(self, knowledge_base=None, fallback_enabled=True):
        self.knowledge_base = knowledge_base
        self.fallback_enabled = fallback_enabled
        self.knowledge_base_status = "unknown"  # unknown, available, unavailable, fallback
        self.validation_history = []
        self.confidence_threshold = 0.7
        self.logger = self._setup_logger()
        
        # 知識庫健康檢查
        self._check_knowledge_base_health()

        # 已知攻擊技術分類 (基於 MITRE ATT&CK) - 作為fallback知識
        self.known_techniques = {
            "reconnaissance": ["port_scan", "service_enum", "web_crawl", "dns_enum", "subdomain_scan"],
            "initial_access": ["phishing", "exploit_public", "brute_force", "spear_phishing", "watering_hole"],
            "execution": ["command_injection", "script_execution", "malware", "powershell", "cmd_exec"],
            "persistence": ["account_creation", "scheduled_task", "service_install", "registry_mod", "startup_folder"],
            "privilege_escalation": ["exploit_elevation", "token_manipulation", "dll_hijacking", "uac_bypass"],
            "defense_evasion": ["obfuscation", "disable_security", "masquerade", "process_injection", "rootkit"],
            "credential_access": ["credential_dump", "keylogging", "password_crack", "hash_dump", "ticket_attack"],
            "discovery": ["system_info", "network_discovery", "process_enum", "file_discovery", "service_discovery"],
            "collection": ["data_collection", "screen_capture", "keylog_capture", "clipboard_data", "audio_capture"],
            "exfiltration": ["data_transfer", "encrypted_channel", "physical_media", "web_service", "dns_exfil"],
        }
        
        # 技術相依性映射（用於邏輯檢查）
        self.technique_dependencies = {
            "privilege_escalation": ["reconnaissance", "initial_access"],
            "persistence": ["initial_access", "execution"],
            "exfiltration": ["discovery", "collection"],
            "credential_access": ["initial_access"],
        }
        
        self.logger.info(f"抗幻覺模組初始化完成，知識庫狀態: {self.knowledge_base_status}")

    def _check_knowledge_base_health(self):
        """檢查知識庫健康狀態"""
        try:
            if self.knowledge_base is None:
                self.knowledge_base_status = "unavailable"
                self.logger.warning("知識庫未提供，將使用fallback機制")
                return
                
            # 嘗試簡單查詢測試
            if hasattr(self.knowledge_base, 'search'):
                test_results = self.knowledge_base.search("test")
                if test_results is not None:
                    self.knowledge_base_status = "available"
                    self.logger.info("知識庫健康檢查通過")
                else:
                    self.knowledge_base_status = "fallback"
                    self.logger.warning("知識庫查詢回應異常，啟用fallback模式")
            else:
                self.knowledge_base_status = "unavailable"
                self.logger.warning("知識庫缺少search方法，將使用內建知識")
                
        except Exception as e:
            self.knowledge_base_status = "unavailable"
            self.logger.error(f"知識庫健康檢查失敗: {e}，將使用fallback機制")

    def _fallback_knowledge_validation(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """備用知識驗證機制（當主知識庫不可用時）"""
        action = step.get("action", "").lower()
        description = step.get("description", "").lower()
        
        # 1. 檢查是否為已知技術
        if not self._is_known_technique(action):
            return {
                "is_valid": False,
                "reason": f"fallback驗證：未知攻擊技術 '{action}'"
            }
        
        # 2. 關鍵字黑名單檢查（明顯的幻覺）
        hallucination_keywords = [
            "quantum", "ai_sentient", "mind_control", "teleport", "magic",
            "supernatural", "alien", "time_travel", "psychic", "telepathy"
        ]
        
        content = (action + " " + description).lower()
        for keyword in hallucination_keywords:
            if keyword in content:
                return {
                    "is_valid": False,
                    "reason": f"fallback驗證：檢測到幻覺關鍵字 '{keyword}'"
                }
        
        # 3. 技術分類一致性檢查
        technique_category = self._get_technique_category(action)
        if technique_category and not self._validate_technique_consistency(description, technique_category):
            return {
                "is_valid": False,
                "reason": "fallback驗證：技術描述與分類不一致"
            }
        
        return {
            "is_valid": True,
            "reason": "fallback驗證通過"
        }

    def _get_technique_category(self, action: str) -> Optional[str]:
        """獲取技術所屬分類"""
        action_lower = action.lower()
        for category, techniques in self.known_techniques.items():
            if any(technique in action_lower for technique in techniques):
                return category
        return None

    def _validate_technique_consistency(self, description: str, category: str) -> bool:
        """驗證技術與描述的一致性"""
        # 定義每個類別的關鍵描述詞
        category_keywords = {
            "reconnaissance": ["掃描", "列舉", "發現", "探測", "偵察"],
            "initial_access": ["入侵", "進入", "獲得存取", "突破", "登入"],
            "execution": ["執行", "運行", "啟動", "命令", "腳本"],
            "persistence": ["持久", "維持", "服務", "任務", "註冊表"],
            "privilege_escalation": ["提權", "權限", "管理員", "系統", "escalate"],
            "defense_evasion": ["隱藏", "逃避", "繞過", "偽裝", "混淆"],
            "credential_access": ["憑證", "密碼", "金鑰", "認證", "credential"],
            "discovery": ["發現", "枚舉", "偵測", "資訊收集", "系統資訊"],
            "collection": ["收集", "擷取", "記錄", "監控", "capture"],
            "exfiltration": ["外洩", "傳輸", "匯出", "資料外送", "exfil"]
        }
        
        expected_keywords = category_keywords.get(category, [])
        description_lower = description.lower()
        
        return any(keyword in description_lower for keyword in expected_keywords)

    def _setup_logger(self):
        """設置日誌記錄器"""
        logger = logging.getLogger("AntiHallucination")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def validate_attack_plan(self, attack_plan: Dict[str, Any]) -> Dict[str, Any]:
        """驗證整個攻擊計畫，移除明顯不合理的步驟

        Args:
            attack_plan: 包含攻擊步驟的計畫字典

        Returns:
            經過驗證和清理的攻擊計畫
        """
        self.logger.info(f"🔍 開始驗證攻擊計畫: {attack_plan.get('name', 'Unknown')}")

        if not attack_plan.get("steps"):
            self.logger.warning("⚠️  攻擊計畫缺少步驟，可能是幻覺")
            return attack_plan

        original_steps = len(attack_plan["steps"])
        validated_steps = []
        removed_steps = []

        for i, step in enumerate(attack_plan["steps"]):
            validation_result = self._validate_single_step(step, i + 1)

            if validation_result["is_valid"]:
                validated_steps.append(step)
            else:
                removed_steps.append(
                    {"step": step, "reason": validation_result["reason"]}
                )
                self.logger.warning(
                    f"🚫 移除可疑步驟 #{i+1}: {step.get('description', 'Unknown')} "
                    f"原因: {validation_result['reason']}"
                )

        # 更新計畫
        attack_plan["steps"] = validated_steps

        # 記錄驗證結果
        validation_summary = {
            "original_steps": original_steps,
            "validated_steps": len(validated_steps),
            "removed_steps": len(removed_steps),
            "removal_rate": (
                len(removed_steps) / original_steps if original_steps > 0 else 0
            ),
            "removed_details": removed_steps,
        }

        self.validation_history.append(validation_summary)

        self.logger.info(
            f"✅ 計畫驗證完成: {original_steps} → {len(validated_steps)} 步驟 "
            f"(移除 {len(removed_steps)} 個可疑步驟)"
        )

        return attack_plan

    def _validate_single_step(
        self, step: Dict[str, Any], step_number: int
    ) -> Dict[str, Any]:
        """驗證單個攻擊步驟的合理性（改進版with fallback）

        Args:
            step: 攻擊步驟字典
            step_number: 步驟編號

        Returns:
            包含驗證結果的字典
        """
        # 1. 基本結構檢查
        if not isinstance(step, dict):
            return {"is_valid": False, "reason": "步驟格式錯誤，非字典類型"}

        required_fields = ["action", "description"]
        for field in required_fields:
            if field not in step:
                return {"is_valid": False, "reason": f"缺少必要欄位: {field}"}

        # 2. 技術分類驗證
        action = step.get("action", "").lower()
        if not self._is_known_technique(action):
            return {"is_valid": False, "reason": f"未知攻擊技術: {action}"}

        # 3. 知識庫驗證 (with fallback機制)
        knowledge_validation = self._validate_with_knowledge_base_fallback(step)
        if not knowledge_validation["is_valid"]:
            return knowledge_validation

        # 4. 步驟順序邏輯檢查
        sequence_validation = self._validate_step_sequence(step, step_number)
        if not sequence_validation["is_valid"]:
            return sequence_validation

        # 5. 邏輯一致性檢查
        logic_validation = self._validate_step_logic(step, step_number)
        if not logic_validation["is_valid"]:
            return logic_validation

        return {"is_valid": True, "reason": "步驟驗證通過"}

    def _validate_with_knowledge_base_fallback(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """使用知識庫驗證步驟（包含fallback機制）"""
        # 首先嘗試主要知識庫
        if self.knowledge_base_status == "available":
            try:
                primary_result = self._validate_with_knowledge_base(step)
                if primary_result["is_valid"]:
                    return primary_result
                else:
                    # 主知識庫驗證失敗，記錄但繼續使用fallback
                    self.logger.debug(f"主知識庫驗證失敗: {primary_result['reason']}")
            except Exception as e:
                self.logger.warning(f"主知識庫查詢異常: {e}，切換至fallback模式")
                self.knowledge_base_status = "fallback"

        # 使用fallback機制
        if self.fallback_enabled:
            fallback_result = self._fallback_knowledge_validation(step)
            self.logger.debug(f"使用fallback驗證: {fallback_result['reason']}")
            return fallback_result
        else:
            # 如果禁用fallback且主知識庫不可用，則預設通過
            self.logger.warning("知識庫不可用且fallback已禁用，預設通過驗證")
            return {"is_valid": True, "reason": "知識庫不可用，預設通過"}

    def _validate_step_sequence(self, step: Dict[str, Any], step_number: int) -> Dict[str, Any]:
        """驗證攻擊步驟順序的合理性"""
        action = step.get("action", "").lower()
        technique_category = self._get_technique_category(action)
        
        if not technique_category:
            return {"is_valid": True, "reason": "無法識別技術分類，跳過順序檢查"}

        # 高級技術不應在早期步驟出現
        advanced_techniques = ["privilege_escalation", "persistence", "exfiltration"]
        if technique_category in advanced_techniques and step_number <= 2:
            return {
                "is_valid": False, 
                "reason": f"高級技術 '{technique_category}' 在第 {step_number} 步出現太早"
            }

        return {"is_valid": True, "reason": "步驟順序檢查通過"}

    def _is_known_technique(self, action: str) -> bool:
        """檢查攻擊技術是否為已知技術"""
        action_lower = action.lower()

        for category, techniques in self.known_techniques.items():
            if action_lower in techniques:
                return True

        # 檢查常見變體
        common_variations = [
            "scan",
            "enum",
            "exploit",
            "inject",
            "dump",
            "crack",
            "discover",
            "collect",
            "transfer",
            "execute",
            "escalate",
        ]

        return any(variation in action_lower for variation in common_variations)

    def _validate_with_knowledge_base(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """使用主要知識庫驗證步驟"""
        try:
            # 搜尋相關知識條目
            query = step.get("description", "") + " " + step.get("action", "")

            if self.knowledge_base and hasattr(self.knowledge_base, "search"):
                results = self.knowledge_base.search(query, top_k=3)

                if not results or len(results) == 0:
                    return {
                        "is_valid": False,
                        "reason": "知識庫中無相關技術資料，可能是幻覺",
                    }

                # 檢查相關性分數（支援新版knowledge_base.py的回應格式）
                best_result = results[0]
                score = 0.0  # 預設分數
                
                if isinstance(best_result, dict):
                    # 優先檢查字典格式的分數
                    if "relevance_score" in best_result:
                        score = best_result["relevance_score"]
                    elif "score" in best_result:
                        score = best_result["score"]
                    else:
                        # 如果字典中沒有分數，但有結果，就認為是有效的
                        score = self.confidence_threshold + 0.1
                elif hasattr(best_result, "score"):
                    # 物件格式的分數
                    score = best_result.score
                elif hasattr(best_result, "relevance_score"):
                    # 物件格式的相關性分數
                    score = best_result.relevance_score
                else:
                    # 如果沒有分數，但有結果，就認為是有效的
                    score = self.confidence_threshold + 0.1

                if score < self.confidence_threshold:
                    return {
                        "is_valid": False,
                        "reason": f"知識庫匹配度過低: {score:.2f} < {self.confidence_threshold}",
                    }

                return {"is_valid": True, "reason": f"知識庫驗證通過 (分數: {score:.2f})"}
            else:
                return {"is_valid": False, "reason": "知識庫不可用或缺少search方法"}

        except Exception as e:
            self.logger.error(f"知識庫驗證異常: {e}")
            # 拋出異常讓上層fallback機制處理
            raise

    def _validate_step_logic(
        self, step: Dict[str, Any], step_number: int
    ) -> Dict[str, Any]:
        """驗證步驟邏輯合理性"""
        action = step.get("action", "").lower()
        description = step.get("description", "").lower()

        # 邏輯矛盾檢查
        contradictions = [
            # 初始步驟不應該是高級技術
            (
                step_number <= 2
                and any(
                    advanced in action
                    for advanced in [
                        "privilege_escalation",
                        "persistence",
                        "exfiltration",
                    ]
                ),
                "初始步驟使用高級攻擊技術，邏輯不合理",
            ),
            # 描述與動作不符
            ("scan" in action and "inject" in description, "掃描動作與注入描述不符"),
            # 不可能的組合
            (
                "brute_force" in action and "stealth" in description,
                "暴力破解與隱蔽操作矛盾",
            ),
        ]

        for condition, reason in contradictions:
            if condition:
                return {"is_valid": False, "reason": reason}

        return {"is_valid": True, "reason": "邏輯驗證通過"}

    def get_validation_stats(self) -> Dict[str, Any]:
        """獲取驗證統計資料"""
        if not self.validation_history:
            return {"總驗證次數": 0}

        total_original = sum(v["original_steps"] for v in self.validation_history)
        total_validated = sum(v["validated_steps"] for v in self.validation_history)
        total_removed = sum(v["removed_steps"] for v in self.validation_history)

        return {
            "總驗證次數": len(self.validation_history),
            "原始步驟總數": total_original,
            "驗證通過步驟": total_validated,
            "移除可疑步驟": total_removed,
            "整體移除率": f"{(total_removed / max(1, total_original)) * 100:.1f}%",
            "平均計畫大小": f"{total_original / len(self.validation_history):.1f} 步驟",
        }

    def export_validation_report(self, output_path: str | None = None) -> str:
        """匯出驗證報告（包含fallback使用統計）"""
        if not output_path:
            output_path = f"anti_hallucination_report_{int(time.time())}.json"

        report = {
            "模組資訊": {
                "名稱": "AIVA 抗幻覺驗證模組（改進版）",
                "版本": "2.0",
                "信心閾值": self.confidence_threshold,
                "知識庫狀態": self.knowledge_base_status,
                "fallback模式": "啟用" if self.fallback_enabled else "禁用"
            },
            "驗證統計": self.get_validation_stats(),
            "驗證歷史": self.validation_history,
            "技術分類庫": self.known_techniques,
            "技術相依性": self.technique_dependencies
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            self.logger.info(f"📊 驗證報告已輸出至: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"報告輸出失敗: {e}")
            return ""

    def reset_knowledge_base(self, new_knowledge_base=None):
        """重設知識庫並重新檢查健康狀態"""
        self.knowledge_base = new_knowledge_base
        self._check_knowledge_base_health()
        self.logger.info(f"知識庫已重設，新狀態: {self.knowledge_base_status}")


# 使用範例（包含fallback演示）
def demo_anti_hallucination():
    """示範抗幻覺模組的使用（包含fallback機制）"""
    print("🧠 AIVA 抗幻覺驗證模組示範（改進版）")
    print("=" * 60)

    # 創建驗證模組（無知識庫，測試fallback）
    print("🔸 測試 1: 無知識庫模式（fallback驗證）")
    validator_no_kb = AntiHallucinationModule(knowledge_base=None)

    # 測試攻擊計畫 (包含一些可疑步驟)
    test_plan = {
        "name": "Web 應用滲透測試",
        "target": "http://example.com",
        "steps": [
            {"action": "port_scan", "description": "掃描目標開放端口", "tool": "nmap"},
            {
                "action": "quantum_hack",  # 明顯的幻覺
                "description": "使用量子算法破解加密",
                "tool": "quantum_tool",
            },
            {"action": "web_crawl", "description": "爬取網站結構", "tool": "spider"},
            {
                "action": "privilege_escalation",  # 邏輯問題：太早使用高級技術
                "description": "提升系統權限",
                "tool": "exploit",
            },
            {
                "action": "sql_injection",
                "description": "測試 SQL 注入漏洞",
                "tool": "sqlmap",
            },
        ],
    }

    print(f"📋 原始計畫包含 {len(test_plan['steps'])} 個步驟")

    # 執行fallback驗證
    validated_plan = validator_no_kb.validate_attack_plan(test_plan)
    print(f"✅ Fallback驗證後剩餘 {len(validated_plan['steps'])} 個步驟")

    # 顯示統計
    stats = validator_no_kb.get_validation_stats()
    print("\n📊 Fallback驗證統計:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 測試簡化模式（無知識庫）
    print("\n🔸 測試 2: 簡化模式驗證")
    try:
        # 使用內建驗證規則
        validator_simple = AntiHallucinationModule()
        
        # 重新測試相同計畫
        validated_plan_simple = validator_simple.validate_attack_plan(test_plan.copy())
        print(f"✅ 簡化驗證後剩餘 {len(validated_plan_simple['steps'])} 個步驟")
        
    except Exception as e:
        print(f"⚠️  簡化驗證失敗: {e}")
        print("   繼續使用fallback模式")

    # 匯出報告
    report_path = validator_no_kb.export_validation_report()
    if report_path:
        print(f"\n📄 詳細報告: {report_path}")


if __name__ == "__main__":
    demo_anti_hallucination()
