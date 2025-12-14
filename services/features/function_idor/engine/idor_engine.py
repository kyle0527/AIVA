from __future__ import annotations
import re
import httpx
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class IdCandidate:
    value: str
    span: tuple[int,int]

@dataclass
class IDORIssue:
    kind: str
    url: str
    description: str
    severity: str = "HIGH"
    cwe: Optional[str] = None
    evidence: Optional[str] = None

class IDOREngine:
    def __init__(self, *, timeout: float, allow_active: bool, safe_mode: bool):
        self.allow_active = allow_active and not safe_mode
        self.safe_mode = safe_mode
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self.client.aclose()

    @staticmethod
    def extract_ids_from_url(url: str) -> List[IdCandidate]:
        ids: List[IdCandidate] = []
        for m in re.finditer(r'(?<![\w])(\d{1,12})(?![\w])', url):
            ids.append(IdCandidate(m.group(1), m.span(1)))
        return ids

    @staticmethod
    def generate_variants(raw: str, count: int) -> List[str]:
        try:
            base = int(raw)
            pool = []
            for d in range(1, count+1):
                pool += [str(base+d), str(max(0, base-d))]
            return list(dict.fromkeys(pool))
        except ValueError:
            return [raw]

    @staticmethod
    def replace_id_in_url(url: str, old: str, new: str) -> str:
        return url.replace(old, new, 1)

    async def test_horizontal(self, url: str, user_a_hdr: dict, user_b_hdr: dict) -> Optional[IDORIssue]:
        if not self.allow_active:
            return IDORIssue(kind="IDOR_HORIZONTAL_POTENTIAL", url=url, description="Potential horizontal IDOR (safe_mode)", cwe="CWE-639", severity="MEDIUM")
        try:
            ra = await self.client.get(url, headers=user_a_hdr)
            rb = await self.client.get(url, headers=user_b_hdr)
            
            if rb.status_code == 200 and ra.status_code in (200, 403, 401) and rb.text and len(rb.text) > 0:
                # ✅ 強化驗證 1: 檢查是否為公開資源
                if self._is_public_resource(rb.text, url):
                    return None  # 公開資源，非 IDOR
                
                # ✅ 強化驗證 2: 檢查是否有團隊/組織共享關係
                if self._has_shared_access(ra.text, rb.text):
                    return None  # 合法共享，非 IDOR
                
                # ✅ 強化驗證 3: 檢查敏感度
                sensitivity = self._calculate_sensitivity(rb.text)
                
                if rb.text == ra.text or len(rb.text) > 0:
                    # 根據敏感度調整嚴重性
                    severity = "CRITICAL" if sensitivity >= 3 else "HIGH"
                    evidence = f"Sensitivity: {sensitivity}/5, Response size: {len(rb.text)} bytes"
                    
                    return IDORIssue(
                        kind="IDOR_HORIZONTAL", 
                        url=url, 
                        description=f"User B accessed resource of User A (Sensitivity: {sensitivity}/5)", 
                        cwe="CWE-639", 
                        severity=severity,
                        evidence=evidence
                    )
        except Exception as e:
            return IDORIssue(kind="IDOR_HORIZONTAL_POTENTIAL", url=url, description=f"Active test failed: {e}", cwe="CWE-639", severity="MEDIUM")
        return None

    async def test_vertical(self, url: str, low_auth_hdr: dict) -> Optional[IDORIssue]:
        if not self.allow_active:
            return IDORIssue(kind="IDOR_VERTICAL_POTENTIAL", url=url, description="Potential vertical privilege escalation (safe_mode)", cwe="CWE-269", severity="MEDIUM")
        try:
            r = await self.client.get(url, headers=low_auth_hdr)
            if r.status_code == 200:
                # ✅ 強化驗證: 檢查敏感度
                sensitivity = self._calculate_sensitivity(r.text)
                evidence = f"Sensitivity: {sensitivity}/5, Access granted to privileged endpoint"
                
                return IDORIssue(
                    kind="IDOR_VERTICAL", 
                    url=url, 
                    description=f"Low-privilege user accessed privileged endpoint (Sensitivity: {sensitivity}/5)", 
                    cwe="CWE-269", 
                    severity="CRITICAL",
                    evidence=evidence
                )
        except Exception as e:
            return IDORIssue(kind="IDOR_VERTICAL_POTENTIAL", url=url, description=f"Active test failed: {e}", cwe="CWE-269", severity="MEDIUM")
        return None
    
    def _is_public_resource(self, response_text: str, url: str) -> bool:
        """
        檢查是否為公開資源
        
        Args:
            response_text: 響應內容
            url: URL
            
        Returns:
            bool: True 表示公開資源
        """
        # 檢查公開資源指標
        public_indicators = [
            '"public":true',
            '"visibility":"public"',
            '"access":"public"',
            '"is_public":true',
            'class="public-profile"',
            'data-public="true"'
        ]
        
        response_lower = response_text.lower()
        for indicator in public_indicators:
            if indicator.lower() in response_lower:
                return True
        
        # 檢查 URL 模式
        public_url_patterns = [
            '/public/',
            '/shared/',
            '/browse/',
            '/explore/'
        ]
        
        url_lower = url.lower()
        for pattern in public_url_patterns:
            if pattern in url_lower:
                return True
        
        return False
    
    def _has_shared_access(self, user_a_text: str, user_b_text: str) -> bool:
        """
        檢查是否有團隊/組織共享關係
        
        Args:
            user_a_text: 用戶 A 的響應
            user_b_text: 用戶 B 的響應
            
        Returns:
            bool: True 表示可能有共享關係
        """
        # 嘗試從響應中提取組織/團隊資訊
        try:
            # 嘗試解析為 JSON
            data_a = json.loads(user_a_text) if user_a_text.strip().startswith('{') else {}
            data_b = json.loads(user_b_text) if user_b_text.strip().startswith('{') else {}
            
            # 檢查組織 ID
            org_a = data_a.get('organization_id') or data_a.get('team_id') or data_a.get('workspace_id')
            org_b = data_b.get('organization_id') or data_b.get('team_id') or data_b.get('workspace_id')
            
            if org_a and org_b and org_a == org_b:
                return True  # 同一組織，可能是合法共享
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # 檢查共享指標
        shared_indicators = [
            'shared_with',
            'collaborators',
            'team_members',
            'organization_members'
        ]
        
        text_lower = user_b_text.lower()
        for indicator in shared_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def _calculate_sensitivity(self, response_text: str) -> int:
        """
        計算數據敏感度 (1-5)
        
        Args:
            response_text: 響應內容
            
        Returns:
            int: 敏感度評分 1-5
        """
        sensitive_fields = {
            # 高敏感 (5分)
            'password': 5,
            'password_hash': 5,
            'api_key': 5,
            'secret': 5,
            'token': 5,
            'credit_card': 5,
            'card_number': 5,
            'cvv': 5,
            'ssn': 5,
            'social_security': 5,
            
            # 中高敏感 (4分)
            'bank_account': 4,
            'account_number': 4,
            'routing_number': 4,
            'passport': 4,
            'driver_license': 4,
            
            # 中等敏感 (3分)
            'email': 3,
            'phone': 3,
            'address': 3,
            'date_of_birth': 3,
            'dob': 3,
            
            # 低敏感 (2分)
            'name': 2,
            'username': 2,
            'profile': 2,
            
            # 極低敏感 (1分)
            'title': 1,
            'description': 1,
        }
        
        text_lower = response_text.lower()
        max_sensitivity = 0
        
        for field, score in sensitive_fields.items():
            if field in text_lower:
                max_sensitivity = max(max_sensitivity, score)
        
        return max_sensitivity if max_sensitivity > 0 else 1
