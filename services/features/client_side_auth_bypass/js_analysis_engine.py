# services/features/client_side_auth_bypass/js_analysis_engine.py
"""
JavaScript 靜態分析引擎

負責分析 JavaScript 代碼中的授權繞過風險
"""

import logging
import re
from typing import Dict, Any, List, Optional


logger = logging.getLogger(__name__)

class JavaScriptAnalysisEngine:
    """
    JavaScript 靜態分析引擎，用於檢測客戶端授權繞過漏洞
    """

    def __init__(self, http_client=None):
        self.http_client = http_client
        logger.info("JavaScriptAnalysisEngine initialized.")
        
        # 定義檢測模式
        self.auth_patterns = {
            'localStorage_auth': {
                'pattern': r'localStorage\.(getItem|setItem)\s*\(\s*["\'](?:token|auth|user|role|permission)["\']',
                'description': 'LocalStorage中存儲敏感授權信息',
                'severity': 'medium'
            },
            'sessionStorage_auth': {
                'pattern': r'sessionStorage\.(getItem|setItem)\s*\(\s*["\'](?:token|auth|user|role|permission)["\']',
                'description': 'SessionStorage中存儲敏感授權信息',
                'severity': 'medium'
            },
            'hardcoded_admin': {
                'pattern': r'(?:role|permission|auth).*["\'](?:admin|administrator|root|superuser)["\']',
                'description': '硬編碼的管理員角色檢查',
                'severity': 'high'
            },
            'client_side_validation': {
                'pattern': r'if\s*\(\s*(?:user\.role|role)\s*[!=]=\s*["\'](?:admin|user)["\']',
                'description': '僅客戶端的權限驗證',
                'severity': 'high'
            },
            'hidden_elements': {
                'pattern': r'\.hide\(\)|\.show\(\)|display\s*:\s*["\']none["\'].*(?:admin|permission)',
                'description': '基於權限隱藏/顯示元素',
                'severity': 'low'
            },
            'jwt_client_decode': {
                'pattern': r'atob\s*\(|JSON\.parse.*token|jwt.*decode',
                'description': '客戶端JWT解析可能洩露信息',
                'severity': 'medium'
            }
        }

    async def analyze(self, url: str, scripts: List[str]) -> List[Dict[str, Any]]:
        """
        分析 JavaScript 腳本列表，尋找授權繞過漏洞

        Args:
            url: 目標URL
            scripts: JavaScript腳本內容列表

        Returns:
            發現的潛在問題列表
        """
        issues = []
        logger.info(f"Analyzing {len(scripts)} JavaScript files for client-side auth bypass")

        for i, script_content in enumerate(scripts):
            if not script_content or not script_content.strip():
                continue
                
            script_issues = await self._analyze_single_script(script_content, f"script_{i}")
            issues.extend(script_issues)

        # 進行跨腳本分析
        cross_script_issues = await self._analyze_cross_script_patterns(scripts, url)
        issues.extend(cross_script_issues)

        logger.info(f"Found {len(issues)} potential client-side auth issues")
        return issues

    async def _analyze_single_script(self, script_content: str, script_identifier: str) -> List[Dict[str, Any]]:
        """分析單個腳本"""
        issues = []
        
        for pattern_name, pattern_info in self.auth_patterns.items():
            try:
                matches = re.finditer(pattern_info['pattern'], script_content, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    # 獲取匹配行號
                    line_num = script_content[:match.start()].count('\n') + 1
                    
                    # 獲取代碼片段上下文
                    lines = script_content.split('\n')
                    start_line = max(0, line_num - 3)
                    end_line = min(len(lines), line_num + 2)
                    context = '\n'.join(lines[start_line:end_line])
                    
                    issue = {
                        'type': pattern_name,
                        'description': pattern_info['description'],
                        'severity': pattern_info['severity'],
                        'script_identifier': script_identifier,
                        'line_number': line_num,
                        'matched_text': match.group(),
                        'context': context,
                        'snippet': match.group()
                    }
                    issues.append(issue)
                    
            except re.error as e:
                logger.error(f"Regex error in pattern {pattern_name}: {e}")

        return issues

    async def _analyze_cross_script_patterns(self, scripts: List[str], url: str) -> List[Dict[str, Any]]:
        """分析跨腳本模式"""
        issues = []
        all_content = '\n'.join(scripts)
        
        # 檢查是否存在統一的授權檢查函數
        auth_functions = re.findall(r'function\s+(?:check|verify|validate)(?:Auth|Permission|Role)\s*\([^}]*\}', 
                                   all_content, re.IGNORECASE | re.DOTALL)
        
        if auth_functions:
            for func in auth_functions:
                # 檢查函數是否只是客戶端驗證
                if 'return true' in func or 'return false' in func:
                    issue = {
                        'type': 'client_auth_function',
                        'description': '發現可能只在客戶端執行的授權檢查函數',
                        'severity': 'high',
                        'script_identifier': 'cross_script',
                        'line_number': 0,
                        'matched_text': func[:200] + '...' if len(func) > 200 else func,
                        'context': func,
                        'snippet': func[:100] + '...' if len(func) > 100 else func
                    }
                    issues.append(issue)

        # 檢查是否有API調用但缺少授權頭
        api_calls = re.findall(r'(?:fetch|axios|ajax)\s*\([^)]*\)', all_content, re.IGNORECASE)
        
        for call in api_calls:
            if 'authorization' not in call.lower() and 'token' not in call.lower():
                issue = {
                    'type': 'unprotected_api_call',
                    'description': 'API調用可能缺少授權頭',
                    'severity': 'medium',
                    'script_identifier': 'cross_script',
                    'line_number': 0,
                    'matched_text': call,
                    'context': call,
                    'snippet': call
                }
                issues.append(issue)

        return issues

    async def analyze_dom_manipulation(self, html_content: str) -> List[Dict[str, Any]]:
        """
        分析DOM操作相關的授權繞過問題
        
        Args:
            html_content: HTML頁面內容
            
        Returns:
            發現的DOM相關授權問題
        """
        issues = []
        
        # 檢查是否有基於CSS類或屬性的權限控制
        permission_classes = re.findall(r'class=["\'][^"\']*(?:admin|hidden|protected)[^"\']*["\']', 
                                       html_content, re.IGNORECASE)
        
        for perm_class in permission_classes:
            issue = {
                'type': 'css_based_auth',
                'description': '基於CSS類的權限控制可能被繞過',
                'severity': 'low',
                'script_identifier': 'html_dom',
                'line_number': 0,
                'matched_text': perm_class,
                'context': perm_class,
                'snippet': perm_class
            }
            issues.append(issue)

        return issues

    def get_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """
        根據發現的問題提供修復建議
        
        Args:
            issues: 發現的問題列表
            
        Returns:
            修復建議列表
        """
        recommendations = []
        issue_types = {issue['type'] for issue in issues}
        
        if 'localStorage_auth' in issue_types or 'sessionStorage_auth' in issue_types:
            recommendations.append(
                "避免在客戶端存儲敏感授權信息，考慮使用HttpOnly Cookies或服務端Session"
            )
            
        if 'hardcoded_admin' in issue_types:
            recommendations.append(
                "移除硬編碼的角色檢查，所有授權驗證應在服務端進行"
            )
            
        if 'client_side_validation' in issue_types:
            recommendations.append(
                "客戶端權限檢查只能用於UI優化，關鍵授權驗證必須在服務端實現"
            )
            
        if 'unprotected_api_call' in issue_types:
            recommendations.append(
                "確保所有API調用都包含適當的授權頭，並在服務端驗證權限"
            )
            
        return recommendations