"""
Simple Task Matcher - 簡單任務工具配對器

基於關鍵字匹配，不需要訓練神經網路
快速、準確、易於維護
"""
from typing import Optional, Dict, List
import re


class SimpleTaskMatcher:
    """
    簡單的任務-工具配對器
    
    使用關鍵字匹配來快速準確地配對 CLI 命令到工具
    """
    
    def __init__(self, tools: List[Dict]):
        """
        初始化匹配器
        
        Args:
            tools: 可用工具列表
        """
        self.tools = tools
        self.tool_names = [tool["name"] for tool in tools]
        
        # 關鍵字映射表 - 可以輕鬆擴展
        self.keyword_patterns = {
            "ScanTrigger": [
                r"掃描",
                r"scan",
                r"啟動.*掃描",
                r"開始.*掃描",
                r"目標.*掃描",
                r"網站.*掃描",
            ],
            "SQLiDetector": [
                r"SQL\s*注入",
                r"sqli",
                r"sql\s*injection",
                r"檢測.*SQL",
                r"測試.*SQL",
                r"sql.*漏洞",
            ],
            "XSSDetector": [
                r"XSS",
                r"xss",
                r"跨站.*腳本",
                r"cross[\s-]?site",
                r"檢測.*XSS",
                r"xss.*漏洞",
            ],
            "CodeAnalyzer": [
                r"分析.*代碼",
                r"分析.*程式",
                r"analyze.*code",
                r"code.*analysis",
                r"檢查.*代碼",
                r"代碼.*結構",
                r"代碼.*質量",
            ],
            "CodeReader": [
                r"讀取.*文件",
                r"讀取.*檔案",
                r"read.*file",
                r"查看.*文件",
                r"打開.*文件",
                r"file.*content",
                r"讀取.*配置",
            ],
            "CodeWriter": [
                r"寫入.*文件",
                r"寫入.*檔案",
                r"write.*file",
                r"創建.*文件",
                r"修改.*文件",
                r"保存.*文件",
                r"file.*write",
            ],
            "ReportGenerator": [
                r"生成.*報告",
                r"產生.*報告",
                r"generate.*report",
                r"創建.*報告",
                r"輸出.*報告",
                r"report.*generate",
            ],
        }
        
        # 編譯正則表達式以提高性能
        self.compiled_patterns = {
            tool: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for tool, patterns in self.keyword_patterns.items()
        }
    
    def match(self, task_description: str) -> tuple[str, float]:
        """
        匹配任務描述到工具
        
        Args:
            task_description: 任務描述文本
            
        Returns:
            (工具名稱, 匹配信心度)
        """
        task_lower = task_description.lower()
        
        # 記錄每個工具的匹配分數
        scores = {tool: 0 for tool in self.tool_names}
        
        # 對每個工具檢查關鍵字匹配
        for tool_name, patterns in self.compiled_patterns.items():
            if tool_name not in self.tool_names:
                continue
                
            for pattern in patterns:
                if pattern.search(task_description):
                    scores[tool_name] += 1
        
        # 找到最高分數的工具
        max_score = max(scores.values())
        
        if max_score == 0:
            # 沒有匹配，返回默認工具
            return "CodeReader", 0.3
        
        # 找到得分最高的工具
        matched_tool = max(scores.items(), key=lambda x: x[1])[0]
        
        # 計算信心度 (匹配數量越多，信心度越高)
        confidence = min(0.5 + (max_score * 0.2), 1.0)
        
        return matched_tool, confidence
    
    def match_with_context(
        self, 
        task_description: str, 
        context: Optional[str] = None
    ) -> tuple[str, float, Dict]:
        """
        匹配任務描述到工具（包含上下文信息）
        
        Args:
            task_description: 任務描述文本
            context: 額外的上下文信息
            
        Returns:
            (工具名稱, 匹配信心度, 詳細信息)
        """
        # 組合任務描述和上下文
        full_text = task_description
        if context:
            full_text = f"{task_description} {context}"
        
        matched_tool, confidence = self.match(full_text)
        
        # 生成詳細信息
        details = {
            "matched_tool": matched_tool,
            "confidence": confidence,
            "task": task_description,
            "context": context,
            "method": "keyword_matching"
        }
        
        return matched_tool, confidence, details
    
    def add_keyword(self, tool_name: str, keyword: str):
        """
        添加新的關鍵字映射
        
        Args:
            tool_name: 工具名稱
            keyword: 關鍵字（支持正則表達式）
        """
        if tool_name not in self.keyword_patterns:
            self.keyword_patterns[tool_name] = []
            self.compiled_patterns[tool_name] = []
        
        self.keyword_patterns[tool_name].append(keyword)
        self.compiled_patterns[tool_name].append(
            re.compile(keyword, re.IGNORECASE)
        )
    
    def get_statistics(self) -> Dict:
        """
        獲取匹配器統計信息
        
        Returns:
            統計信息字典
        """
        return {
            "total_tools": len(self.tool_names),
            "total_patterns": sum(
                len(patterns) for patterns in self.keyword_patterns.values()
            ),
            "tools": self.tool_names,
            "patterns_per_tool": {
                tool: len(patterns)
                for tool, patterns in self.keyword_patterns.items()
            }
        }


# 方便的工廠函數
def create_task_matcher(tools: List[Dict]) -> SimpleTaskMatcher:
    """
    創建任務匹配器實例
    
    Args:
        tools: 工具列表
        
    Returns:
        SimpleTaskMatcher 實例
    """
    return SimpleTaskMatcher(tools)


if __name__ == "__main__":
    # 測試
    from services.core.aiva_core.ai_engine.cli_tools import get_all_tools
    
    print("="*70)
    print("Simple Task Matcher 測試")
    print("="*70)
    
    # 創建匹配器
    cli_tools_dict = get_all_tools()
    tools = [
        {"name": tool_name, "instance": tool_obj}
        for tool_name, tool_obj in cli_tools_dict.items()
    ]
    
    matcher = SimpleTaskMatcher(tools)
    
    # 顯示統計
    stats = matcher.get_statistics()
    print(f"\n[統計信息]")
    print(f"  工具數量: {stats['total_tools']}")
    print(f"  關鍵字模式總數: {stats['total_patterns']}")
    print(f"  每個工具的模式數:")
    for tool, count in stats['patterns_per_tool'].items():
        print(f"    {tool}: {count}")
    
    # 測試案例
    test_cases = [
        "掃描目標網站 example.com",
        "檢測 SQL 注入漏洞在 login 頁面的 username 參數",
        "檢測 XSS 漏洞在 search 頁面的 q 參數",
        "分析 services/core 目錄的代碼結構",
        "讀取 pyproject.toml 配置文件",
        "寫入配置到 config.json",
        "生成掃描報告",
        "未知的任務描述",
    ]
    
    print(f"\n[測試案例]")
    print()
    
    correct = 0
    expected_tools = [
        "ScanTrigger",
        "SQLiDetector", 
        "XSSDetector",
        "CodeAnalyzer",
        "CodeReader",
        "CodeWriter",
        "ReportGenerator",
        None,  # 未知任務
    ]
    
    for i, (task, expected) in enumerate(zip(test_cases, expected_tools), 1):
        matched_tool, confidence = matcher.match(task)
        
        is_correct = (matched_tool == expected) if expected else True
        if is_correct and expected:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"[測試 {i}] {status}")
        print(f"  任務: {task}")
        if expected:
            print(f"  預期: {expected}")
        print(f"  匹配: {matched_tool}")
        print(f"  信心度: {confidence:.1%}")
        print()
    
    if len([e for e in expected_tools if e]) > 0:
        accuracy = correct / len([e for e in expected_tools if e])
        print(f"配對準確度: {correct}/{len([e for e in expected_tools if e])} = {accuracy:.1%}")
    
    print("="*70)
