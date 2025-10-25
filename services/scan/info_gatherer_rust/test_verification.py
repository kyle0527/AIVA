#!/usr/bin/env python3
"""
API 驗證功能測試腳本
測試從 Rust 服務返回的帶有驗證資訊的結果
"""

import json
from typing import Dict, Any, List

def format_verification_result(finding: Dict[str, Any]) -> str:
    """格式化驗證結果為可讀字符串"""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"🔍 Finding: {finding.get('rule_name', 'Unknown')}")
    lines.append(f"{'='*60}")
    
    # 基本資訊
    lines.append(f"Type: {finding['info_type']}")
    lines.append(f"Location: {finding['location']}")
    lines.append(f"Severity: {finding.get('severity', 'N/A')}")
    lines.append(f"Confidence: {finding['confidence']:.2f}")
    
    # 熵值
    if finding.get('entropy'):
        lines.append(f"Entropy: {finding['entropy']:.2f}")
    
    # 驗證狀態
    verified = finding.get('verified')
    if verified is not None:
        status_icon = "✓" if verified else "✗"
        status_text = "VALID" if verified else "INVALID"
        lines.append(f"\n{status_icon} Verification Status: {status_text}")
        
        # 驗證訊息
        if finding.get('verification_message'):
            lines.append(f"Message: {finding['verification_message']}")
        
        # 驗證元數據
        if finding.get('verification_metadata'):
            lines.append("\nMetadata:")
            for key, value in finding['verification_metadata'].items():
                lines.append(f"  - {key}: {value}")
    else:
        lines.append("\n⊘ Verification Status: NOT VERIFIED")
    
    # 密鑰值 (遮罩處理)
    value = finding['value']
    if len(value) > 20:
        masked_value = f"{value[:8]}...{value[-8:]}"
    else:
        masked_value = value[:4] + "****" + value[-4:]
    lines.append(f"\nValue (masked): {masked_value}")
    
    return "\n".join(lines)


def analyze_findings(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析掃描結果統計"""
    stats = {
        'total': len(findings),
        'verified_valid': 0,
        'verified_invalid': 0,
        'not_verified': 0,
        'by_severity': {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'NONE': 0
        },
        'by_type': {}
    }
    
    for finding in findings:
        # 驗證狀態統計
        verified = finding.get('verified')
        if verified is True:
            stats['verified_valid'] += 1
        elif verified is False:
            stats['verified_invalid'] += 1
        else:
            stats['not_verified'] += 1
        
        # 嚴重性統計
        severity = finding.get('severity', 'NONE')
        if severity in stats['by_severity']:
            stats['by_severity'][severity] += 1
        else:
            stats['by_severity']['NONE'] += 1
        
        # 類型統計
        info_type = finding['info_type']
        stats['by_type'][info_type] = stats['by_type'].get(info_type, 0) + 1
    
    return stats


def print_summary(stats: Dict[str, Any]):
    """打印統計摘要"""
    print("\n" + "="*60)
    print("📊 SCAN RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nTotal Findings: {stats['total']}")
    
    print("\n🔐 Verification Status:")
    print(f"  ✓ Valid:         {stats['verified_valid']}")
    print(f"  ✗ Invalid:       {stats['verified_invalid']}")
    print(f"  ⊘ Not Verified:  {stats['not_verified']}")
    
    print("\n⚠️  Severity Distribution:")
    for severity, count in sorted(stats['by_severity'].items()):
        if count > 0:
            print(f"  {severity:12s} {count:3d}")
    
    print("\n📑 Finding Types:")
    for finding_type, count in sorted(stats['by_type'].items()):
        print(f"  {finding_type:20s} {count:3d}")
    
    print("\n" + "="*60)


def main():
    """主函數 - 模擬測試"""
    # 模擬掃描結果 (從 RabbitMQ 接收的 JSON)
    sample_findings = [
        {
            "task_id": "scan-001",
            "info_type": "secret",
            "value": "ghp_1234567890abcdefghijklmnopqrstuvwxyz",
            "confidence": 0.9,
            "location": "config/secrets.env:12",
            "severity": "CRITICAL",
            "entropy": 5.2,
            "rule_name": "GitHub Personal Access Token",
            "verified": True,
            "verification_message": "Valid GitHub token for user: octocat",
            "verification_metadata": {
                "username": "octocat"
            }
        },
        {
            "task_id": "scan-001",
            "info_type": "secret",
            "value": "xoxb-1234-5678-abcd-efghijklmnop",
            "confidence": 0.9,
            "location": "src/slack_client.py:45",
            "severity": "CRITICAL",
            "rule_name": "Slack Bot Token",
            "verified": False,
            "verification_message": "Invalid Slack token: token_revoked",
            "verification_metadata": None
        },
        {
            "task_id": "scan-001",
            "info_type": "secret",
            "value": "sk_test_1234567890abcdefghijklmnopqrstuvwxyz",
            "confidence": 0.9,
            "location": "payment/stripe_config.py:8",
            "severity": "HIGH",
            "rule_name": "Stripe API Key",
            "verified": True,
            "verification_message": "Valid Stripe API key",
            "verification_metadata": None
        },
        {
            "task_id": "scan-001",
            "info_type": "secret",
            "value": "AKIAIOSFODNN7EXAMPLE",
            "confidence": 0.9,
            "location": "aws/credentials:3",
            "severity": "CRITICAL",
            "rule_name": "AWS Access Key ID",
            "verified": None,
            "verification_message": "AWS verification requires both Access Key ID and Secret Access Key",
            "verification_metadata": None
        },
        {
            "task_id": "scan-001",
            "info_type": "secret",
            "value": "mongodb+srv://user:pass@cluster.mongodb.net/db",
            "confidence": 0.9,
            "location": "config/database.yml:5",
            "severity": "HIGH",
            "rule_name": "MongoDB Connection String",
            "verified": None,
            "verification_message": None,
            "verification_metadata": None
        }
    ]
    
    print("\n🚀 AIVA API Verification Test")
    print(f"Testing {len(sample_findings)} findings...")
    
    # 顯示每個發現
    for finding in sample_findings:
        print(format_verification_result(finding))
    
    # 統計分析
    stats = analyze_findings(sample_findings)
    print_summary(stats)
    
    # 高優先級警報
    print("\n🚨 HIGH PRIORITY ALERTS")
    print("="*60)
    for finding in sample_findings:
        if finding.get('verified') is True and finding.get('severity') in ['CRITICAL', 'HIGH']:
            print(f"\n⚠️  ACTIVE CREDENTIAL LEAK DETECTED!")
            print(f"   Rule: {finding['rule_name']}")
            print(f"   Location: {finding['location']}")
            if finding.get('verification_metadata'):
                print(f"   Details: {finding['verification_metadata']}")
    
    print("\n✅ Test completed!\n")


if __name__ == "__main__":
    main()
