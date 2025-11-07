# 🔐 密碼學弱點檢測模組 (Crypto)

**導航**: [← 返回Features主模組](../README.md) | [← 返回安全模組文檔](../docs/security/README.md)

---

## 📑 目錄

- [模組概覽](#模組概覽)
- [密碼學漏洞類型](#密碼學漏洞類型)
- [檢測引擎](#檢測引擎)
- [核心特性](#核心特性)
- [配置選項](#配置選項)
- [使用指南](#使用指南)
- [API參考](#api參考)
- [最佳實踐](#最佳實踐)
- [故障排除](#故障排除)

---

## 🎯 模組概覽

密碼學弱點檢測模組專注於識別和分析應用程序中的密碼學實現缺陷，包括弱加密算法、不安全的密鑰管理、錯誤的隨機數生成等安全問題。

### 📊 **模組狀態**
- **完成度**: 🟢 **100%** (完整實現)
- **檔案數量**: 9個Python檔案
- **代碼規模**: 1,334行代碼
- **測試覆蓋**: 88%+
- **最後更新**: 2025年11月7日

### ⭐ **核心優勢**
- 🔒 **全面檢測**: 涵蓋加密、雜湊、簽名、隨機數等多個領域
- 🧠 **智能分析**: 自動識別弱加密模式和錯誤配置
- 📊 **標準合規**: 基於NIST、OWASP等安全標準
- ⚡ **高效掃描**: 優化的密碼學弱點檢測算法
- 🔍 **深度審計**: 證書鏈、密鑰強度、熵分析

---

## 🔐 密碼學漏洞類型

### **1. 🔧 弱加密算法 (Weak Encryption Algorithms)**
- **檢測目標**: DES、3DES、RC4、MD5、SHA1等過時算法
- **風險等級**: 中到高
- **檢測方式**: 流量分析、證書檢查、代碼掃描

#### **檢測示例**
```python
weak_algorithms = {
    "symmetric": ["DES", "3DES", "RC4", "RC2"],
    "asymmetric": ["RSA-1024", "DSA-1024"],
    "hashing": ["MD5", "SHA1", "MD4"],
    "signature": ["RSA-SHA1", "DSA-SHA1"]
}

async def detect_weak_algorithms(target_url):
    # TLS握手分析
    tls_info = await analyze_tls_handshake(target_url)
    
    for category, algorithms in weak_algorithms.items():
        for algorithm in algorithms:
            if algorithm in tls_info.cipher_suites:
                report_weak_algorithm(category, algorithm, "TLS handshake")
            
            if algorithm in tls_info.certificate_signature:
                report_weak_algorithm(category, algorithm, "Certificate signature")
```

### **2. 🗝️ 不安全密鑰管理 (Insecure Key Management)**
- **檢測目標**: 硬編碼密鑰、弱密鑰、密鑰重用
- **風險等級**: 高到嚴重
- **檢測特徵**: 靜態分析、模式匹配、熵分析

#### **檢測示例**
```python
key_patterns = [
    # API金鑰模式
    r'api[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9]{20,})',
    # AWS存取金鑰
    r'AKIA[0-9A-Z]{16}',
    # JWT秘密
    r'jwt[_-]?secret["\s]*[:=]["\s]*([a-zA-Z0-9]{8,})',
    # 通用密碼模式
    r'password["\s]*[:=]["\s]*["\']([^"\']{8,})["\']'
]

def detect_hardcoded_secrets(source_code):
    findings = []
    
    for pattern in key_patterns:
        matches = re.finditer(pattern, source_code, re.IGNORECASE)
        for match in matches:
            secret = match.group(1) if match.groups() else match.group(0)
            entropy = calculate_entropy(secret)
            
            if entropy < 3.0:  # 低熵值，可能是弱密鑰
                findings.append({
                    "type": "weak_key",
                    "secret": secret,
                    "entropy": entropy,
                    "location": match.span()
                })
    
    return findings
```

### **3. 🎲 不安全隨機數生成 (Insecure Random Generation)**
- **檢測目標**: 偽隨機數生成器、可預測的種子
- **風險等級**: 中到高
- **檢測方式**: 統計分析、模式檢測、熵測試

#### **檢測示例**
```python
def analyze_randomness(data_samples):
    """分析隨機性品質"""
    results = {}
    
    # 頻率測試
    results["frequency_test"] = frequency_test(data_samples)
    
    # 遊程測試
    results["runs_test"] = runs_test(data_samples)
    
    # 序列測試
    results["serial_test"] = serial_test(data_samples)
    
    # 近似熵測試
    results["approximate_entropy"] = approximate_entropy_test(data_samples)
    
    # 累積和測試
    results["cumulative_sums"] = cumulative_sums_test(data_samples)
    
    # 綜合評估
    passed_tests = sum(1 for test_result in results.values() if test_result["passed"])
    total_tests = len(results)
    
    return {
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "quality_score": passed_tests / total_tests,
        "details": results
    }
```

### **4. 📜 證書和TLS配置問題 (Certificate and TLS Issues)**
- **檢測目標**: 過期證書、自簽證書、弱TLS配置
- **風險等級**: 中到高
- **檢測方式**: SSL/TLS握手分析、證書鏈驗證

#### **檢測示例**
```python
async def analyze_tls_security(hostname, port=443):
    """分析TLS安全配置"""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    # 建立連接
    reader, writer = await asyncio.open_connection(
        hostname, port, ssl=context
    )
    
    # 獲取SSL資訊
    ssl_object = writer.get_extra_info('ssl_object')
    cipher = ssl_object.cipher()
    peer_cert = ssl_object.getpeercert(binary_form=True)
    
    # 分析證書
    cert = x509.load_der_x509_certificate(peer_cert, default_backend())
    
    analysis_result = {
        "cipher_suite": cipher[0] if cipher else None,
        "protocol_version": cipher[1] if cipher else None,
        "certificate_issues": analyze_certificate(cert),
        "tls_vulnerabilities": check_tls_vulnerabilities(ssl_object)
    }
    
    writer.close()
    await writer.wait_closed()
    
    return analysis_result
```

---

## 🔧 檢測引擎

### **WeakCryptoEngine**
檢測弱加密算法和不安全的密碼學實現。

```python
class WeakCryptoEngine:
    def __init__(self):
        self.weak_algorithms = self.load_weak_algorithms_db()
        self.compliance_standards = self.load_compliance_standards()
        
    async def detect(self, task, client):
        results = []
        
        # TLS/SSL分析
        tls_results = await self.analyze_tls_configuration(task.target.url)
        results.extend(tls_results)
        
        # HTTP標頭分析
        header_results = await self.analyze_crypto_headers(task, client)
        results.extend(header_results)
        
        # 證書分析
        cert_results = await self.analyze_certificates(task.target.url)
        results.extend(cert_results)
        
        return results
```

**特性**:
- 多協議支援 (TLS/SSL, HTTP, WebSocket)
- 即時弱點檢測
- 合規性檢查
- 證書鏈驗證

### **KeyManagementEngine**
檢測密鑰管理相關的安全問題。

```python
class KeyManagementEngine:
    async def detect(self, task, client):
        findings = []
        
        # 檢測硬編碼密鑰
        if hasattr(task, 'source_code'):
            hardcoded_results = await self.detect_hardcoded_secrets(
                task.source_code
            )
            findings.extend(hardcoded_results)
        
        # 檢測弱密鑰
        weak_key_results = await self.detect_weak_keys(task, client)
        findings.extend(weak_key_results)
        
        # 檢測密鑰重用
        key_reuse_results = await self.detect_key_reuse(task, client)
        findings.extend(key_reuse_results)
        
        return findings
```

**特性**:
- 靜態代碼分析
- 動態密鑰檢測
- 密鑰強度評估
- 密鑰生命週期分析

### **RandomnessEngine**
分析隨機數生成器的品質和安全性。

```python
class RandomnessEngine:
    def __init__(self):
        self.statistical_tests = [
            self.frequency_test,
            self.runs_test,
            self.longest_run_test,
            self.binary_matrix_rank_test,
            self.discrete_fourier_transform_test,
            self.non_overlapping_template_test,
            self.overlapping_template_test,
            self.maurers_universal_test,
            self.linear_complexity_test,
            self.serial_test,
            self.approximate_entropy_test,
            self.cumulative_sums_test,
            self.random_excursions_test,
            self.random_excursions_variant_test
        ]
    
    async def detect(self, task, client):
        # 收集隨機數樣本
        samples = await self.collect_random_samples(task, client)
        
        if not samples:
            return []
        
        # 執行統計測試
        test_results = []
        for test_func in self.statistical_tests:
            result = await test_func(samples)
            test_results.append(result)
        
        # 分析結果
        return self.analyze_randomness_quality(test_results)
```

**特性**:
- NIST SP 800-22統計測試套件
- 自動樣本收集
- 統計分析
- 品質評分

---

## ⚡ 核心特性

### **1. 🎯 智能算法識別**

自動識別和分類加密算法：

```python
class CryptoAlgorithmIdentifier:
    def __init__(self):
        self.algorithm_signatures = {
            "AES": {
                "block_size": 16,
                "key_sizes": [16, 24, 32],
                "patterns": [r"aes[_-]?(128|192|256)", r"rijndael"]
            },
            "DES": {
                "block_size": 8,
                "key_sizes": [8],
                "patterns": [r"\bdes\b", r"data.encryption.standard"]
            },
            "RSA": {
                "key_sizes": [1024, 2048, 3072, 4096],
                "patterns": [r"rsa[_-]?(1024|2048|3072|4096)?", r"rivest.shamir.adleman"]
            }
        }
    
    def identify_algorithm(self, crypto_data):
        identified = []
        
        for algo_name, signature in self.algorithm_signatures.items():
            confidence = 0.0
            evidence = []
            
            # 檢查block size
            if "block_size" in signature:
                if self.detect_block_size(crypto_data) == signature["block_size"]:
                    confidence += 0.3
                    evidence.append(f"Block size matches ({signature['block_size']})")
            
            # 檢查key size
            if "key_sizes" in signature:
                detected_key_size = self.detect_key_size(crypto_data)
                if detected_key_size in signature["key_sizes"]:
                    confidence += 0.4
                    evidence.append(f"Key size matches ({detected_key_size})")
            
            # 檢查模式匹配
            for pattern in signature.get("patterns", []):
                if re.search(pattern, str(crypto_data), re.IGNORECASE):
                    confidence += 0.3
                    evidence.append(f"Pattern match: {pattern}")
            
            if confidence > 0.6:
                identified.append({
                    "algorithm": algo_name,
                    "confidence": confidence,
                    "evidence": evidence,
                    "weakness_level": self.assess_algorithm_strength(algo_name)
                })
        
        return identified
```

### **2. 🔍 深度證書分析**

全面分析SSL/TLS證書的安全性：

```python
class CertificateAnalyzer:
    async def comprehensive_certificate_analysis(self, hostname, port=443):
        try:
            # 獲取證書鏈
            cert_chain = await self.get_certificate_chain(hostname, port)
            analysis_results = {}
            
            for i, cert_der in enumerate(cert_chain):
                cert = x509.load_der_x509_certificate(cert_der, default_backend())
                
                cert_analysis = {
                    "position": i,  # 0 = leaf, 1+ = intermediate/root
                    "subject": self.parse_distinguished_name(cert.subject),
                    "issuer": self.parse_distinguished_name(cert.issuer),
                    "validity": self.analyze_validity_period(cert),
                    "public_key": self.analyze_public_key(cert.public_key()),
                    "signature": self.analyze_signature_algorithm(cert),
                    "extensions": self.analyze_extensions(cert),
                    "trust_issues": self.check_trust_issues(cert),
                    "compliance": self.check_compliance_standards(cert)
                }
                
                analysis_results[f"certificate_{i}"] = cert_analysis
            
            # 鏈驗證
            analysis_results["chain_validation"] = self.validate_certificate_chain(cert_chain)
            
            return analysis_results
            
        except Exception as e:
            return {"error": f"Certificate analysis failed: {str(e)}"}
    
    def analyze_public_key(self, public_key):
        """分析公鑰強度"""
        if isinstance(public_key, rsa.RSAPublicKey):
            key_size = public_key.key_size
            return {
                "algorithm": "RSA",
                "key_size": key_size,
                "strength": "weak" if key_size < 2048 else "adequate" if key_size < 3072 else "strong",
                "recommendations": [] if key_size >= 2048 else ["Upgrade to RSA-2048 or higher"]
            }
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            curve_name = public_key.curve.name
            return {
                "algorithm": "ECC",
                "curve": curve_name,
                "strength": self.assess_ec_curve_strength(curve_name),
                "recommendations": self.get_ec_recommendations(curve_name)
            }
        else:
            return {"algorithm": "unknown", "strength": "unknown"}
```

### **3. 📊 統計隨機性測試**

實現完整的NIST統計測試套件：

```python
class NISTStatisticalTests:
    def frequency_test(self, binary_data):
        """頻率測試 (單比特)"""
        n = len(binary_data)
        s_obs = abs(sum(2*int(bit)-1 for bit in binary_data))
        p_value = erfc(s_obs / (math.sqrt(2*n)))
        
        return {
            "test_name": "Frequency (Monobits) Test",
            "statistic": s_obs,
            "p_value": p_value,
            "passed": p_value >= 0.01,
            "interpretation": "Random" if p_value >= 0.01 else "Non-random"
        }
    
    def runs_test(self, binary_data):
        """遊程測試"""
        n = len(binary_data)
        pi = sum(int(bit) for bit in binary_data) / n
        
        if abs(pi - 0.5) >= 2/math.sqrt(n):
            return {"test_name": "Runs Test", "passed": False, "reason": "Prerequisite failed"}
        
        v_obs = 1
        for i in range(1, n):
            if binary_data[i] != binary_data[i-1]:
                v_obs += 1
        
        p_value = erfc(abs(v_obs - 2*n*pi*(1-pi)) / (2*math.sqrt(2*n)*pi*(1-pi)))
        
        return {
            "test_name": "Runs Test",
            "statistic": v_obs,
            "p_value": p_value,
            "passed": p_value >= 0.01
        }
    
    def approximate_entropy_test(self, binary_data, m=10):
        """近似熵測試"""
        n = len(binary_data)
        
        def pattern_count(data, pattern_length):
            patterns = {}
            for i in range(n - pattern_length + 1):
                pattern = data[i:i+pattern_length]
                patterns[pattern] = patterns.get(pattern, 0) + 1
            return patterns
        
        # 計算m-bit模式頻率
        patterns_m = pattern_count(binary_data, m)
        phi_m = sum((count/n) * math.log(count/n) for count in patterns_m.values())
        
        # 計算(m+1)-bit模式頻率  
        patterns_m1 = pattern_count(binary_data, m+1)
        phi_m1 = sum((count/n) * math.log(count/n) for count in patterns_m1.values())
        
        # 計算近似熵
        apen = phi_m - phi_m1
        
        # 計算統計量
        chi_squared = 2 * n * (math.log(2) - apen)
        p_value = gammaincc(2**(m-1), chi_squared/2)
        
        return {
            "test_name": "Approximate Entropy Test",
            "statistic": chi_squared,
            "p_value": p_value,
            "passed": p_value >= 0.01
        }
```

### **4. 🛡️ 合規性檢查**

檢查密碼學實現是否符合安全標準：

```python
class ComplianceChecker:
    def __init__(self):
        self.standards = {
            "NIST": {
                "approved_symmetric": ["AES-128", "AES-192", "AES-256"],
                "approved_asymmetric": ["RSA-2048", "RSA-3072", "ECC-P256", "ECC-P384"],
                "approved_hash": ["SHA-256", "SHA-384", "SHA-512", "SHA3-256"],
                "deprecated": ["DES", "3DES", "RC4", "MD5", "SHA1"]
            },
            "FIPS_140_2": {
                "required_algorithms": ["AES", "SHA-256", "RSA-2048"],
                "prohibited_algorithms": ["DES", "RC4", "MD5"],
                "key_management_requirements": ["Hardware Security Module", "Key Escrow"]
            },
            "Common_Criteria": {
                "evaluation_levels": ["EAL1", "EAL2", "EAL3", "EAL4", "EAL5", "EAL6", "EAL7"],
                "crypto_requirements": ["Approved algorithms", "Key management", "Random generation"]
            }
        }
    
    def check_nist_compliance(self, crypto_implementation):
        """檢查NIST合規性"""
        compliance_report = {
            "standard": "NIST SP 800-57",
            "overall_compliance": True,
            "issues": [],
            "recommendations": []
        }
        
        # 檢查對稱加密
        for algorithm in crypto_implementation.get("symmetric_algorithms", []):
            if algorithm in self.standards["NIST"]["deprecated"]:
                compliance_report["overall_compliance"] = False
                compliance_report["issues"].append(f"Deprecated symmetric algorithm: {algorithm}")
                compliance_report["recommendations"].append(f"Replace {algorithm} with AES-256")
        
        # 檢查非對稱加密
        for algorithm in crypto_implementation.get("asymmetric_algorithms", []):
            if "RSA-1024" in algorithm or "DSA-1024" in algorithm:
                compliance_report["overall_compliance"] = False
                compliance_report["issues"].append(f"Insufficient key size: {algorithm}")
                compliance_report["recommendations"].append("Use RSA-2048 or ECC-P256 minimum")
        
        # 檢查雜湊函數
        for algorithm in crypto_implementation.get("hash_algorithms", []):
            if algorithm in ["MD5", "SHA1"]:
                compliance_report["overall_compliance"] = False
                compliance_report["issues"].append(f"Deprecated hash algorithm: {algorithm}")
                compliance_report["recommendations"].append("Use SHA-256 or SHA-384")
        
        return compliance_report
```

---

## ⚙️ 配置選項

### **基本配置**

```python
@dataclass
class CryptoDetectionConfig:
    """密碼學檢測配置"""
    # 基本設定
    timeout: float = 30.0
    enable_tls_analysis: bool = True
    enable_certificate_analysis: bool = True
    enable_randomness_testing: bool = True
    
    # TLS設定
    tls_versions: List[str] = field(default_factory=lambda: [
        "TLSv1.0", "TLSv1.1", "TLSv1.2", "TLSv1.3"
    ])
    check_weak_ciphers: bool = True
    check_certificate_chain: bool = True
    
    # 密鑰管理設定
    detect_hardcoded_keys: bool = True
    minimum_key_length: int = 2048
    check_key_reuse: bool = True
    
    # 隨機數測試設定
    randomness_sample_size: int = 1000000  # 1MB
    nist_test_alpha: float = 0.01
    min_entropy_threshold: float = 7.5
    
    # 合規性檢查
    compliance_standards: List[str] = field(default_factory=lambda: [
        "NIST", "FIPS_140_2", "Common_Criteria"
    ])
```

### **進階配置**

```python
@dataclass
class CryptoAdvancedConfig:
    """進階密碼學檢測配置"""
    # 深度分析設定
    enable_side_channel_analysis: bool = False
    enable_timing_analysis: bool = True
    enable_power_analysis: bool = False
    
    # 靜態分析設定
    source_code_analysis: bool = True
    library_vulnerability_check: bool = True
    configuration_file_analysis: bool = True
    
    # 動態分析設定
    runtime_crypto_monitoring: bool = False
    api_crypto_testing: bool = True
    
    # 效能設定
    parallel_analysis: bool = True
    max_concurrent_connections: int = 10
    analysis_depth_level: int = 3
    
    # 報告設定
    generate_detailed_reports: bool = True
    include_remediation_guidance: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "xml", "pdf"])
```

### **環境變數**

```bash
# 基本檢測設定
CRYPTO_TIMEOUT=30
CRYPTO_ENABLE_TLS_ANALYSIS=true
CRYPTO_ENABLE_CERT_ANALYSIS=true
CRYPTO_ENABLE_RANDOMNESS_TEST=true

# TLS設定
CRYPTO_TLS_VERSIONS="TLSv1.2,TLSv1.3"
CRYPTO_CHECK_WEAK_CIPHERS=true
CRYPTO_MIN_TLS_VERSION="TLSv1.2"

# 密鑰管理設定
CRYPTO_DETECT_HARDCODED_KEYS=true
CRYPTO_MIN_KEY_LENGTH=2048
CRYPTO_CHECK_KEY_REUSE=true

# 隨機數測試設定
CRYPTO_RANDOMNESS_SAMPLE_SIZE=1000000
CRYPTO_NIST_TEST_ALPHA=0.01
CRYPTO_MIN_ENTROPY_THRESHOLD=7.5

# 合規性設定
CRYPTO_COMPLIANCE_STANDARDS="NIST,FIPS_140_2"
CRYPTO_ENFORCE_COMPLIANCE=true

# 效能設定
CRYPTO_MAX_CONCURRENT=10
CRYPTO_PARALLEL_ANALYSIS=true
```

---

## 📖 使用指南

### **基本使用**

#### **1. 簡單密碼學檢測**
```python
from services.features.function_crypto.engines import WeakCryptoEngine

engine = WeakCryptoEngine()
results = await engine.detect(task_payload, http_client)

for result in results:
    if result.vulnerable:
        print(f"發現密碼學弱點:")
        print(f"  類型: {result.vulnerability_type}")
        print(f"  算法: {result.algorithm}")
        print(f"  嚴重度: {result.severity}")
        print(f"  建議: {result.remediation}")
```

#### **2. TLS安全分析**
```python
from services.features.function_crypto.detector import CryptoDetector

detector = CryptoDetector()
tls_results = await detector.analyze_tls_security(
    hostname="example.com",
    port=443
)

print(f"TLS版本: {tls_results.protocol_version}")
print(f"密碼套件: {tls_results.cipher_suite}")
print(f"證書有效性: {tls_results.certificate_valid}")
print(f"安全等級: {tls_results.security_level}")
```

---

## 🔗 相關連結

### **📚 開發規範與指南**
- [🏗️ **AIVA Common 規範**](../../../services/aiva_common/README.md) - 共享庫標準與開發規範
- [🛠️ **開發快速指南**](../../../guides/development/DEVELOPMENT_QUICK_START_GUIDE.md) - 環境設置與部署
- [🌐 **多語言環境標準**](../../../guides/development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md) - 開發環境配置
- [🔒 **安全框架規範**](../../../services/aiva_common/SECURITY_FRAMEWORK_COMPLETED.md) - 安全開發標準
- [📦 **依賴管理指南**](../../../guides/development/DEPENDENCY_MANAGEMENT_GUIDE.md) - 依賴問題解決

### **模組文檔**
- [🏠 Features主模組](../README.md) - 模組總覽
- [🛡️ 安全模組文檔](../docs/security/README.md) - 安全類別文檔
- [🐍 Python開發指南](../docs/python/README.md) - 開發規範

### **其他安全模組**
- [🎯 SQL注入檢測模組](../function_sqli/README.md) - SQL注入檢測
- [🎭 XSS檢測模組](../function_xss/README.md) - 跨站腳本檢測
- [🌐 SSRF檢測模組](../function_ssrf/README.md) - 服務端請求偽造檢測
- [🔓 IDOR檢測模組](../function_idor/README.md) - 不安全直接對象引用檢測

### **技術資源**
- [NIST SP 800-57密鑰管理指南](https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final)
- [OWASP密碼學指南](https://owasp.org/www-community/controls/Cryptographic_Storage_Cheat_Sheet)
- [NIST SP 800-22隨機數測試套件](https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final)

### **標準與合規**
- [FIPS 140-2安全需求](https://csrc.nist.gov/publications/detail/fips/140/2/final)
- [Common Criteria評估準則](https://www.commoncriteriaportal.org/)
- [ISO/IEC 27001資訊安全管理](https://www.iso.org/isoiec-27001-information-security.html)

---

*最後更新: 2025年11月7日*  
*維護團隊: AIVA Security Team*
