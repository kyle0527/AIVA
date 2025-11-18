#!/usr/bin/env python3
"""
AIVA Features 威脅模型組圖生成器
按照 OWASP 和安全威脅類型重新組織架構
"""

import json
from pathlib import Path
from collections import defaultdict

def create_owasp_top10_mapping():
    """按 OWASP Top 10 組織架構"""
    
    with open("_out/architecture_diagrams/features_diagram_classification.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    classifications = data.get('classifications', {})
    
    owasp_mapping = {
        'A01_Broken_Access_Control': {
            'description': 'Broken Access Control',
            'components': []
        },
        'A02_Cryptographic_Failures': {
            'description': 'Cryptographic Failures', 
            'components': []
        },
        'A03_Injection': {
            'description': 'Injection',
            'components': []
        },
        'A04_Insecure_Design': {
            'description': 'Insecure Design',
            'components': []
        },
        'A05_Security_Misconfiguration': {
            'description': 'Security Misconfiguration',
            'components': []
        },
        'A06_Vulnerable_Components': {
            'description': 'Vulnerable and Outdated Components',
            'components': []
        },
        'A07_Authentication_Failures': {
            'description': 'Identification and Authentication Failures',
            'components': []
        },
        'A08_Software_Data_Integrity': {
            'description': 'Software and Data Integrity Failures',
            'components': []
        },
        'A09_Security_Logging': {
            'description': 'Security Logging and Monitoring Failures',
            'components': []
        },
        'A10_SSRF': {
            'description': 'Server-Side Request Forgery',
            'components': []
        }
    }
    
    # 分類組件到 OWASP 類別
    for name, info in classifications.items():
        lower_name = name.lower()
        file_path = info.get('file_path', '').lower()
        
        # A01 - Broken Access Control
        if any(x in lower_name for x in ['idor', 'authz', 'authorization', 'access', 'privilege']):
            owasp_mapping['A01_Broken_Access_Control']['components'].append((name, info))
        
        # A02 - Cryptographic Failures
        elif any(x in lower_name for x in ['crypto', 'hash', 'encrypt', 'decrypt', 'certificate', 'tls', 'ssl']):
            owasp_mapping['A02_Cryptographic_Failures']['components'].append((name, info))
        
        # A03 - Injection
        elif any(x in lower_name for x in ['injection', 'sqli', 'sql', 'xss', 'script', 'command']):
            owasp_mapping['A03_Injection']['components'].append((name, info))
        
        # A04 - Insecure Design
        elif any(x in lower_name for x in ['design', 'architecture', 'pattern', 'threat', 'model']):
            owasp_mapping['A04_Insecure_Design']['components'].append((name, info))
        
        # A05 - Security Misconfiguration
        elif any(x in lower_name for x in ['config', 'setting', 'parameter', 'default']):
            owasp_mapping['A05_Security_Misconfiguration']['components'].append((name, info))
        
        # A06 - Vulnerable Components
        elif any(x in lower_name for x in ['component', 'dependency', 'library', 'package', 'version']):
            owasp_mapping['A06_Vulnerable_Components']['components'].append((name, info))
        
        # A07 - Authentication Failures
        elif any(x in lower_name for x in ['auth', 'login', 'session', 'token', 'jwt', 'oauth', 'credential']):
            owasp_mapping['A07_Authentication_Failures']['components'].append((name, info))
        
        # A08 - Software and Data Integrity
        elif any(x in lower_name for x in ['integrity', 'validation', 'verify', 'check', 'hash']):
            owasp_mapping['A08_Software_Data_Integrity']['components'].append((name, info))
        
        # A09 - Security Logging
        elif any(x in lower_name for x in ['log', 'monitor', 'telemetry', 'statistic', 'track', 'audit']):
            owasp_mapping['A09_Security_Logging']['components'].append((name, info))
        
        # A10 - SSRF
        elif any(x in lower_name for x in ['ssrf', 'request', 'fetch', 'url', 'redirect']):
            owasp_mapping['A10_SSRF']['components'].append((name, info))
    
    return owasp_mapping

def create_attack_chain_mapping():
    """按攻擊鏈組織架構"""
    
    with open("_out/architecture_diagrams/features_diagram_classification.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    classifications = data.get('classifications', {})
    
    attack_chain = {
        'reconnaissance': {
            'description': 'Reconnaissance & Information Gathering',
            'components': []
        },
        'initial_access': {
            'description': 'Initial Access',
            'components': []
        },
        'execution': {
            'description': 'Execution',
            'components': []
        },
        'persistence': {
            'description': 'Persistence',
            'components': []
        },
        'privilege_escalation': {
            'description': 'Privilege Escalation',
            'components': []
        },
        'defense_evasion': {
            'description': 'Defense Evasion',
            'components': []
        },
        'credential_access': {
            'description': 'Credential Access',
            'components': []
        },
        'discovery': {
            'description': 'Discovery',
            'components': []
        },
        'lateral_movement': {
            'description': 'Lateral Movement',
            'components': []
        },
        'collection': {
            'description': 'Collection',
            'components': []
        },
        'exfiltration': {
            'description': 'Exfiltration',
            'components': []
        },
        'impact': {
            'description': 'Impact',
            'components': []
        }
    }
    
    # 分類組件到攻擊鏈階段
    for name, info in classifications.items():
        lower_name = name.lower()
        file_path = info.get('file_path', '').lower()
        
        # Reconnaissance
        if any(x in lower_name for x in ['scan', 'discover', 'enum', 'probe', 'fingerprint']):
            attack_chain['reconnaissance']['components'].append((name, info))
        
        # Initial Access
        elif any(x in lower_name for x in ['exploit', 'bypass', 'injection', 'upload', 'rce']):
            attack_chain['initial_access']['components'].append((name, info))
        
        # Execution
        elif any(x in lower_name for x in ['execute', 'run', 'command', 'script', 'payload']):
            attack_chain['execution']['components'].append((name, info))
        
        # Persistence
        elif any(x in lower_name for x in ['persist', 'backdoor', 'implant', 'rootkit']):
            attack_chain['persistence']['components'].append((name, info))
        
        # Privilege Escalation
        elif any(x in lower_name for x in ['escalation', 'privilege', 'admin', 'root', 'sudo']):
            attack_chain['privilege_escalation']['components'].append((name, info))
        
        # Defense Evasion
        elif any(x in lower_name for x in ['evasion', 'obfuscation', 'steganography', 'hiding']):
            attack_chain['defense_evasion']['components'].append((name, info))
        
        # Credential Access
        elif any(x in lower_name for x in ['credential', 'password', 'hash', 'token', 'key']):
            attack_chain['credential_access']['components'].append((name, info))
        
        # Discovery
        elif any(x in lower_name for x in ['discovery', 'reconnaissance', 'network', 'service']):
            attack_chain['discovery']['components'].append((name, info))
        
        # Lateral Movement
        elif any(x in lower_name for x in ['lateral', 'movement', 'pivot', 'tunnel']):
            attack_chain['lateral_movement']['components'].append((name, info))
        
        # Collection
        elif any(x in lower_name for x in ['collect', 'gather', 'harvest', 'scrape']):
            attack_chain['collection']['components'].append((name, info))
        
        # Exfiltration
        elif any(x in lower_name for x in ['exfil', 'export', 'download', 'transfer']):
            attack_chain['exfiltration']['components'].append((name, info))
        
        # Impact
        elif any(x in lower_name for x in ['impact', 'damage', 'destroy', 'corrupt', 'ransom']):
            attack_chain['impact']['components'].append((name, info))
    
    return attack_chain

def create_security_testing_methodology_mapping():
    """按安全測試方法學組織"""
    
    with open("_out/architecture_diagrams/features_diagram_classification.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    classifications = data.get('classifications', {})
    
    methodology = {
        'static_analysis': {
            'description': 'Static Application Security Testing (SAST)',
            'components': []
        },
        'dynamic_analysis': {
            'description': 'Dynamic Application Security Testing (DAST)',
            'components': []
        },
        'interactive_analysis': {
            'description': 'Interactive Application Security Testing (IAST)',
            'components': []
        },
        'dependency_analysis': {
            'description': 'Software Composition Analysis (SCA)',
            'components': []
        },
        'configuration_analysis': {
            'description': 'Cloud Security Posture Management (CSPM)',
            'components': []
        },
        'manual_testing': {
            'description': 'Manual Penetration Testing',
            'components': []
        },
        'automated_testing': {
            'description': 'Automated Security Testing',
            'components': []
        },
        'threat_modeling': {
            'description': 'Threat Modeling & Architecture Review',
            'components': []
        }
    }
    
    # 分類組件到測試方法學
    for name, info in classifications.items():
        lower_name = name.lower()
        file_path = info.get('file_path', '').lower()
        
        # SAST
        if 'sast' in file_path or any(x in lower_name for x in ['static', 'ast', 'code_analysis']):
            methodology['static_analysis']['components'].append((name, info))
        
        # DAST  
        elif any(x in lower_name for x in ['dynamic', 'runtime', 'live', 'active']):
            methodology['dynamic_analysis']['components'].append((name, info))
        
        # IAST
        elif any(x in lower_name for x in ['interactive', 'instrument', 'hybrid']):
            methodology['interactive_analysis']['components'].append((name, info))
        
        # SCA
        elif any(x in lower_name for x in ['component', 'dependency', 'library', 'package']):
            methodology['dependency_analysis']['components'].append((name, info))
        
        # CSPM
        elif any(x in lower_name for x in ['cloud', 'config', 'posture', 'compliance']):
            methodology['configuration_analysis']['components'].append((name, info))
        
        # Manual Testing
        elif any(x in lower_name for x in ['manual', 'exploit', 'proof', 'poc']):
            methodology['manual_testing']['components'].append((name, info))
        
        # Automated Testing
        elif any(x in lower_name for x in ['auto', 'batch', 'mass', 'bulk']):
            methodology['automated_testing']['components'].append((name, info))
        
        # Threat Modeling
        elif any(x in lower_name for x in ['threat', 'model', 'architecture', 'design']):
            methodology['threat_modeling']['components'].append((name, info))
    
    return methodology

def generate_threat_model_mermaid(mapping_type, data, title):
    """生成威脅模型 Mermaid 圖表"""
    
    mermaid_content = f"""---
title: {title}
---
flowchart TD
    subgraph "AIVA Features - {title}"
        direction TB
"""
    
    node_id = 1
    connections = []
    
    # 根據映射類型生成不同的圖表結構
    if mapping_type == "owasp":
        # 創建 OWASP Top 10 的層次結構
        for category_id, category_data in data.items():
            if category_data['components']:  # 只顯示有組件的類別
                category_node = f"OWASP{node_id}"
                node_id += 1
                
                description = category_data['description']
                component_count = len(category_data['components'])
                
                # 按嚴重程度著色
                severity_class = "critical" if category_id in ['A01_Broken_Access_Control', 'A03_Injection'] else "high"
                
                mermaid_content += f'        {category_node}["{category_id}<br/>{description}<br/>{component_count} 組件"]:::{severity_class}\n'
                
                # 分析組件語言分佈
                languages = defaultdict(int)
                for comp_name, comp_info in category_data['components']:
                    lang = comp_info.get('language', 'unknown')
                    languages[lang] += 1
                
                # 顯示主要語言
                if languages:
                    lang_node = f"L{node_id}"
                    node_id += 1
                    main_lang = max(languages, key=languages.get)
                    mermaid_content += f'        {lang_node}["{main_lang}: {languages[main_lang]}/{component_count}"]:::language\n'
                    connections.append(f'        {category_node} --> {lang_node}')
    
    elif mapping_type == "attack_chain":
        # 創建攻擊鏈的順序流程
        chain_order = [
            'reconnaissance', 'initial_access', 'execution', 'persistence',
            'privilege_escalation', 'defense_evasion', 'credential_access',
            'discovery', 'lateral_movement', 'collection', 'exfiltration', 'impact'
        ]
        
        prev_node = None
        for phase in chain_order:
            phase_data = data.get(phase, {})
            if phase_data.get('components'):  # 只顯示有組件的階段
                phase_node = f"ATK{node_id}"
                node_id += 1
                
                description = phase_data['description']
                component_count = len(phase_data['components'])
                
                mermaid_content += f'        {phase_node}["{phase.replace("_", " ").title()}<br/>{description}<br/>{component_count} 組件"]:::attack_phase\n'
                
                # 連接到前一個階段
                if prev_node:
                    connections.append(f'        {prev_node} --> {phase_node}')
                
                prev_node = phase_node
    
    elif mapping_type == "methodology":
        # 創建測試方法學的分類結構
        for method_id, method_data in data.items():
            if method_data['components']:  # 只顯示有組件的方法
                method_node = f"METH{node_id}"
                node_id += 1
                
                description = method_data['description']
                component_count = len(method_data['components'])
                
                mermaid_content += f'        {method_node}["{method_id.replace("_", " ").title()}<br/>{description}<br/>{component_count} 組件"]:::methodology\n'
                
                # 分析複雜度分佈
                complexities = defaultdict(int)
                for comp_name, comp_info in method_data['components']:
                    complexity = comp_info.get('complexity', 'unknown')
                    complexities[complexity] += 1
                
                # 顯示複雜度分佈
                if complexities:
                    complexity_node = f"COMP{node_id}"
                    node_id += 1
                    main_complexity = max(complexities, key=complexities.get)
                    mermaid_content += f'        {complexity_node}["{main_complexity}: {complexities[main_complexity]}/{component_count}"]:::complexity\n'
                    connections.append(f'        {method_node} --> {complexity_node}')
    
    # 添加所有連接
    for connection in connections:
        mermaid_content += f'{connection}\n'
    
    # 添加樣式定義
    mermaid_content += """
    end

    classDef critical fill:#ffebee,stroke:#c62828,stroke-width:3px
    classDef high fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef language fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef attack_phase fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef methodology fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef complexity fill:#f9fbe7,stroke:#689f38,stroke-width:2px
"""
    
    return mermaid_content

def generate_all_threat_model_diagrams():
    """生成所有威脅模型組圖"""
    
    print("🛡️ 生成 OWASP Top 10 組圖...")
    owasp_mapping = create_owasp_top10_mapping()
    
    print("⚔️ 生成攻擊鏈組圖...")
    attack_chain = create_attack_chain_mapping()
    
    print("🔬 生成安全測試方法學組圖...")
    methodology = create_security_testing_methodology_mapping()
    
    # 生成 Mermaid 圖表
    diagrams = [
        ("owasp", owasp_mapping, "OWASP Top 10 威脅分類", "FEATURES_OWASP_TOP10"),
        ("attack_chain", attack_chain, "MITRE ATT&CK 攻擊鏈", "FEATURES_ATTACK_CHAIN"),
        ("methodology", methodology, "安全測試方法學", "FEATURES_SECURITY_METHODOLOGY")
    ]
    
    output_dir = Path("_out/architecture_diagrams/threat_model")
    output_dir.mkdir(exist_ok=True)
    
    generated_files = []
    
    for diagram_type, data, title, filename in diagrams:
        print(f"📊 生成 {title} 圖表...")
        
        mermaid_content = generate_threat_model_mermaid(diagram_type, data, title)
        
        output_file = output_dir / f"{filename}.mmd"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)
        
        generated_files.append(str(output_file))
        print(f"✅ 已生成: {output_file}")
    
    # 生成威脅模型統計報告
    generate_threat_model_report(owasp_mapping, attack_chain, methodology)
    
    return generated_files

def generate_threat_model_report(owasp_mapping, attack_chain, methodology):
    """生成威脅模型分析報告"""
    
    report_content = """# AIVA Features 威脅模型組圖分析報告

## 🛡️ **OWASP Top 10 威脅分析**

### 威脅覆蓋統計

"""
    
    # OWASP 統計
    total_owasp_components = 0
    for category_id, category_data in owasp_mapping.items():
        component_count = len(category_data['components'])
        total_owasp_components += component_count
        
        if component_count > 0:
            report_content += f"- **{category_id}** ({category_data['description']}): {component_count} 個組件\n"
            
            # 語言分佈
            languages = defaultdict(int)
            for comp_name, comp_info in category_data['components']:
                lang = comp_info.get('language', 'unknown')
                languages[lang] += 1
            
            lang_info = ', '.join(f"{lang}: {count}" for lang, count in languages.items())
            report_content += f"  - 語言分佈: {lang_info}\n"
    
    report_content += f"\n**總覆蓋組件**: {total_owasp_components}\n\n"
    
    # 攻擊鏈統計
    report_content += "## ⚔️ **MITRE ATT&CK 攻擊鏈分析**\n\n### 攻擊階段覆蓋\n\n"
    
    total_attack_components = 0
    for phase_id, phase_data in attack_chain.items():
        component_count = len(phase_data['components'])
        total_attack_components += component_count
        
        if component_count > 0:
            report_content += f"- **{phase_id.replace('_', ' ').title()}**: {component_count} 個組件\n"
    
    report_content += f"\n**總覆蓋組件**: {total_attack_components}\n\n"
    
    # 測試方法學統計
    report_content += "## 🔬 **安全測試方法學分析**\n\n### 測試方法覆蓋\n\n"
    
    total_method_components = 0
    for method_id, method_data in methodology.items():
        component_count = len(method_data['components'])
        total_method_components += component_count
        
        if component_count > 0:
            report_content += f"- **{method_data['description']}**: {component_count} 個組件\n"
    
    report_content += f"\n**總覆蓋組件**: {total_method_components}\n\n"
    
    # 新組圖建議
    report_content += """## 💡 **威脅導向組圖建議**

### 🎯 **按風險等級組織**
1. **Critical Risk**: 影響核心業務的高危險威脅
2. **High Risk**: 可能造成重大損失的威脅
3. **Medium Risk**: 需要監控的中等威脅
4. **Low Risk**: 影響較小的威脅

### 🔄 **按檢測能力組織**
1. **Real-time Detection**: 即時檢測能力
2. **Batch Analysis**: 批次分析能力
3. **Deep Inspection**: 深度檢查能力
4. **Compliance Check**: 合規檢查能力

### 📊 **按攻擊面組織**
1. **Web Application**: Web 應用程式攻擊面
2. **API Security**: API 安全攻擊面
3. **Infrastructure**: 基礎設施攻擊面
4. **Supply Chain**: 供應鏈攻擊面

---

**📊 威脅模型統計**:
- **OWASP 覆蓋**: {owasp_total} 個組件
- **攻擊鏈覆蓋**: {attack_total} 個組件  
- **測試方法覆蓋**: {method_total} 個組件
- **新組圖方案**: 3 種威脅導向組織方式

*這些威脅模型組圖從安全防禦的角度重新組織了 AIVA Features 模組，有助於安全團隊理解和管理威脅防護能力。*
""".format(
        owasp_total=total_owasp_components,
        attack_total=total_attack_components,
        method_total=total_method_components
    )
    
    # 保存報告
    report_file = Path("_out/architecture_diagrams/threat_model/THREAT_MODEL_ANALYSIS_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📋 威脅模型報告已生成: {report_file}")

if __name__ == "__main__":
    print("🚀 開始生成威脅模型組圖...")
    
    generated_files = generate_all_threat_model_diagrams()
    
    print(f"\n✨ 完成！共生成 {len(generated_files)} 個威脅模型組圖:")
    for file in generated_files:
        print(f"  🛡️ {file}")
    
    print("\n💡 威脅模型組圖包括:")
    print("  🛡️ OWASP Top 10 威脅分類 - 按已知安全風險分組")
    print("  ⚔️ MITRE ATT&CK 攻擊鏈 - 按攻擊階段分組") 
    print("  🔬 安全測試方法學 - 按測試類型分組")
    print("  📊 威脅模型分析報告 - 威脅覆蓋統計和建議")