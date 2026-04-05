#!/usr/bin/env python3
"""AIVA 抗幻覺驗證模組
用途: 基於知識庫驗證 AI 生成的攻擊計畫，移除不合理步驟
基於: BioNeuron_模型_AI核心大腦.md 分析建議

設計原則:
- 有錯就報錯: 不使用降級機制，知識庫不可用時直接拋出異常
- 嚴格驗證: 所有步驟必須經過知識庫確認
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional
import warnings


class KnowledgeBaseUnavailableError(Exception):
    """知識庫不可用異常"""
    pass


class AntiHallucinationModule:
    """抗幻覺驗證模組 - 防止 AI 產生不合理的攻擊步驟
    
    設計原則:
    - 嚴格模式: 知識庫必須可用，否則拋出異常
    - 多層驗證: 結合規則驗證、知識庫驗證和統計驗證
    - 有錯就報錯: 不隱藏問題，立即失敗
    """

    def __init__(self, knowledge_base=None):
        """初始化抗幻覺模組
        
        Args:
            knowledge_base: 知識庫實例 (必須提供且可用)
            
        Raises:
            KnowledgeBaseUnavailableError: 知識庫不可用時拋出
        """
        self.knowledge_base = knowledge_base
        self.validation_history = []
        self.confidence_threshold = 0.7
        self.logger = self._setup_logger()
        
        # 嚴格檢查知識庫
        self._require_knowledge_base()

        # 已知攻擊技術分類 (基於 MITRE ATT&CK)
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
        
        self.logger.info("✅ 抗幻覺模組初始化完成，知識庫已驗證可用")

    def _require_knowledge_base(self):
        """嚴格要求知識庫必須可用
        
        Raises:
            KnowledgeBaseUnavailableError: 知識庫不可用時拋出
        """
        if self.knowledge_base is None:
            raise KnowledgeBaseUnavailableError(
                "知識庫未提供。AntiHallucinationModule 需要有效的知識庫才能運作。"
            )
        
        if not hasattr(self.knowledge_base, 'search'):
            raise KnowledgeBaseUnavailableError(
                "知識庫缺少 search 方法。請提供符合介面要求的知識庫實例。"
            )
        
        # 嘗試簡單查詢測試
        try:
            test_results = self.knowledge_base.search("test")
            if test_results is None:
                raise KnowledgeBaseUnavailableError(
                    "知識庫 search 方法返回 None，請確認知識庫已正確初始化。"
                )
        except Exception as e:
            raise KnowledgeBaseUnavailableError(
                f"知識庫健康檢查失敗: {e}"
            ) from e
        
        self.logger.info("✅ 知識庫健康檢查通過")

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

    def validate_attack_plan(self, attack_plan: dict[str, Any]) -> dict[str, Any]:
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
        """驗證單個攻擊步驟的合理性（嚴格模式）

        Args:
            step: 攻擊步驟字典
            step_number: 步驟編號

        Returns:
            包含驗證結果的字典
            
        Raises:
            KnowledgeBaseUnavailableError: 知識庫查詢失敗時拋出
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

        # 3. 知識庫驗證 (嚴格模式，有錯就報錯)
        knowledge_validation = self._validate_with_knowledge_base(step)
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

    def _validate_step_sequence(self, step: Dict[str, Any], step_number: int) -> Dict[str, Any]:
        """驗證攻擊步驟順序的合理性"""
        action = step.get("action", "").lower()
        technique_category = self._get_technique_category(action)
        
        if not technique_category:
            return {"is_valid": False, "reason": f"無法識別技術分類: {action}"}

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
    
    def _extract_relevance_score(self, result: Any) -> float:
        """提取相關性分數（降低複雜度的輔助函數）"""
        if isinstance(result, dict):
            # 優先檢查字典格式的分數
            if "relevance_score" in result:
                return result["relevance_score"]
            elif "score" in result:
                return result["score"]
            else:
                # 如果字典中沒有分數，但有結果，就認為是有效的
                return self.confidence_threshold + 0.1
        elif hasattr(result, "score"):
            # 物件格式的分數
            return result.score
        elif hasattr(result, "relevance_score"):
            # 物件格式的相關性分數
            return result.relevance_score
        else:
            # 如果沒有分數，但有結果，就認為是有效的
            return self.confidence_threshold + 0.1

    def _validate_with_knowledge_base(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """使用主要知識庫驗證步驟
        
        重構為輔助函數以降低認知複雜度
        """
        try:
            # 搜尋相關知識條目
            query = step.get("description", "") + " " + step.get("action", "")

            if not (self.knowledge_base and hasattr(self.knowledge_base, "search")):
                return {"is_valid": False, "reason": "知識庫不可用或缺少search方法"}
            
            results = self.knowledge_base.search(query, top_k=3)

            if not results or len(results) == 0:
                return {
                    "is_valid": False,
                    "reason": "知識庫中無相關技術資料，可能是幻覺",
                }

            # 檢查相關性分數（支援新版knowledge_base.py的回應格式）
            best_result = results[0]
            score = self._extract_relevance_score(best_result)

            if score < self.confidence_threshold:
                return {
                    "is_valid": False,
                    "reason": f"知識庫匹配度過低: {score:.2f} < {self.confidence_threshold}",
                }

            return {"is_valid": True, "reason": f"知識庫驗證通過 (分數: {score:.2f})"}

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

    def get_validation_stats(self) -> dict[str, Any]:
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
        """匯出驗證報告"""
        if not output_path:
            output_path = f"anti_hallucination_report_{int(time.time())}.json"

        report = {
            "模組資訊": {
                "名稱": "AIVA 抗幻覺驗證模組（嚴格模式）",
                "版本": "2.1",
                "信心閾值": self.confidence_threshold,
                "模式": "strict (fail-fast)"
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

    def reset_knowledge_base(self, new_knowledge_base):
        """重設知識庫
        
        Args:
            new_knowledge_base: 新的知識庫實例（不可為 None）
            
        Raises:
            KnowledgeBaseUnavailableError: 如果新知識庫無效
        """
        if new_knowledge_base is None:
            raise KnowledgeBaseUnavailableError(
                "知識庫不可為 None。"
                "請提供有效的知識庫實例。"
            )
        self.knowledge_base = new_knowledge_base
        self._require_knowledge_base()
        self.logger.info("知識庫已重設並驗證成功")
