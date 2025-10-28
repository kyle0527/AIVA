#!/usr/bin/env python3
"""
AIVA Schema Generator - 單一事實來源工具

從 core_schema_sot.yaml 生成所有語言的一致 schema 定義
確保 Python、Go、Rust 之間的完全一致性
"""

import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse
import sys


class SchemaGenerator:
    """Schema 生成器 - 從 YAML SOT 生成跨語言 schema"""
    
    def __init__(self, sot_file: Path):
        """初始化生成器
        
        Args:
            sot_file: core_schema_sot.yaml 文件路径
        """
        self.sot_file = sot_file
        self.config = self._load_sot()
        self.timestamp = datetime.now().isoformat()
        
    def _load_sot(self) -> Dict[str, Any]:
        """載入 SOT YAML 配置"""
        try:
            with open(self.sot_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"無法載入 SOT 文件 {self.sot_file}: {e}")
    
    def generate_all(self):
        """生成所有語言的 schema"""
        print(f"🚀 開始生成 AIVA Schema - 基於 SOT: {self.sot_file}")
        print(f"📅 時間戳: {self.timestamp}")
        print(f"📋 Schema 版本: {self.config.get('version', '1.0.0')}")
        
        # 生成 Python schema
        self.generate_python()
        
        # 生成 Rust schema  
        self.generate_rust()
        
        # 生成 Go schema
        self.generate_go()
        
        print("✅ 所有 schema 生成完成！")
    
    def generate_python(self):
        """生成 Python Pydantic schema"""
        print("\n🐍 生成 Python schema...")
        
        python_config = self.config['generation_config']['python']
        target_dir = Path(python_config['target_dir'])
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成基礎類型
        self._generate_python_base_types(target_dir)
        
        # 生成發現定義
        self._generate_python_findings(target_dir)
        
        # 生成 __init__.py
        self._generate_python_init(target_dir)
        
        print(f"✅ Python schema 已生成至: {target_dir}")
    
    def _generate_python_base_types(self, target_dir: Path):
        """生成 Python 基礎類型"""
        content = '''"""
AIVA 基礎類型 Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義

⚠️  此檔案由core_schema_sot.yaml自動生成，請勿手動修改
📅 最後更新: {timestamp}
🔄 Schema 版本: {version}
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


'''.format(timestamp=self.timestamp, version=self.config.get('version', '1.0.0'))
        
        # 生成基礎類型定義
        base_types = self.config.get('base_types', {})
        
        for type_name, type_def in base_types.items():
            content += self._generate_python_class(type_name, type_def)
            content += "\n\n"
        
        with open(target_dir / 'base_types.py', 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_python_findings(self, target_dir: Path):
        """生成 Python 發現定義"""
        content = '''"""
AIVA Findings Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義

⚠️  此檔案由core_schema_sot.yaml自動生成，請勿手動修改
📅 最後更新: {timestamp}
🔄 Schema 版本: {version}
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .base_types import *


'''.format(timestamp=self.timestamp, version=self.config.get('version', '1.0.0'))
        
        # 生成發現相關定義
        findings = self.config.get('findings', {})
        
        for type_name, type_def in findings.items():
            content += self._generate_python_class(type_name, type_def)
            content += "\n\n"
        
        with open(target_dir / 'findings.py', 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_python_class(self, class_name: str, class_def: Dict[str, Any]) -> str:
        """生成 Python class 定義"""
        fields = class_def.get('fields', {})
        description = class_def.get('description', '')
        extends = class_def.get('extends')
        
        # 開始 class 定義
        if extends:
            class_header = f'class {class_name}({extends}):'
        else:
            class_header = f'class {class_name}(BaseModel):'
        
        content = f'{class_header}\n'
        content += f'    """{description}"""\n\n'
        
        # 生成字段
        for field_name, field_def in fields.items():
            field_type = field_def.get('type', 'str')
            required = field_def.get('required', True)
            default = field_def.get('default')
            description = field_def.get('description', '')
            validation = field_def.get('validation', {})
            
            # 處理字段類型
            if not required and not field_type.startswith('Optional['):
                field_type = f'Optional[{field_type}]'
            
            # 生成字段定義
            field_line = f'    {field_name}: {field_type}'
            
            # 添加 Field 驗證
            field_constraints = []
            if 'enum' in validation:
                field_constraints.append(f"choices={validation['enum']}")
            if 'pattern' in validation:
                field_constraints.append(f"pattern=r'{validation['pattern']}'")
            if 'max_length' in validation:
                field_constraints.append(f"max_length={validation['max_length']}")
            if 'minimum' in validation:
                field_constraints.append(f"ge={validation['minimum']}")
            if 'maximum' in validation:
                field_constraints.append(f"le={validation['maximum']}")
            
            if default is not None:
                if isinstance(default, str):
                    field_line += f' = "{default}"'
                elif isinstance(default, dict) and not default:
                    field_line += f' = Field(default_factory=dict)'
                elif isinstance(default, list) and not default:
                    field_line += f' = Field(default_factory=list)'
                else:
                    field_line += f' = {default}'
            elif not required:
                field_line += ' = None'
            elif field_constraints:
                field_line += f' = Field({", ".join(field_constraints)})'
            
            content += field_line + '\n'
            content += f'    """{description}"""\n\n'
        
        return content
    
    def generate_rust(self):
        """生成統一的 Rust schema"""
        print("\n🦀 生成 Rust schema...")
        
        rust_config = self.config['generation_config']['rust']
        target_dir = Path(rust_config['target_dir'])
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成統一的 Rust schema
        self._generate_rust_unified(target_dir)
        
        print(f"✅ Rust schema 已生成至: {target_dir}")
    
    def _generate_rust_unified(self, target_dir: Path):
        """生成統一的 Rust schema 文件"""
        content = '''// AIVA Rust Schema - 自動生成
// 版本: {version}
// 基於 core_schema_sot.yaml 作為單一事實來源
// 此文件與 Python aiva_common 保持完全一致性
//
// ⚠️  此檔案自動生成，請勿手動修改
// 📅 最後更新: {timestamp}

use serde::{{Deserialize, Serialize}};
use std::collections::HashMap;
use chrono::{{DateTime, Utc}};

// ==================== 枚舉定義 ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Severity {{
    #[serde(rename = "critical")]
    Critical,
    #[serde(rename = "high")]
    High,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "info")]
    Info,
}}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Confidence {{
    #[serde(rename = "confirmed")]
    Confirmed,
    #[serde(rename = "firm")]
    Firm,
    #[serde(rename = "tentative")]
    Tentative,
}}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FindingStatus {{
    #[serde(rename = "new")]
    New,
    #[serde(rename = "confirmed")]
    Confirmed,
    #[serde(rename = "false_positive")]
    FalsePositive,
    #[serde(rename = "fixed")]
    Fixed,
    #[serde(rename = "ignored")]
    Ignored,
}}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HttpMethod {{
    #[serde(rename = "GET")]
    Get,
    #[serde(rename = "POST")]
    Post,
    #[serde(rename = "PUT")]
    Put,
    #[serde(rename = "DELETE")]
    Delete,
    #[serde(rename = "PATCH")]
    Patch,
    #[serde(rename = "HEAD")]
    Head,
    #[serde(rename = "OPTIONS")]
    Options,
}}

'''.format(version=self.config.get('version', '1.0.0'), timestamp=self.timestamp)
        
        # 生成結構定義
        content += self._generate_rust_structs()
        
        with open(target_dir / 'mod.rs', 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_rust_structs(self) -> str:
        """生成 Rust 結構定義"""
        content = '''
// ==================== 核心結構定義 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    pub message_id: String,
    pub trace_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    pub source_module: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default = "default_version")]
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter: Option<String>,
    #[serde(default = "default_get_method")]
    pub method: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwe: Option<String>,
    pub severity: Severity,
    pub confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_time_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingImpact {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub business_impact: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub technical_impact: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affected_users: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cost: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingRecommendation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    #[serde(default)]
    pub remediation_steps: Vec<String>,
    #[serde(default)]
    pub references: Vec<String>,
}

// ==================== 主要 Payload 結構 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingPayload {
    pub finding_id: String,
    pub task_id: String,
    pub scan_id: String,
    pub status: FindingStatus,
    pub vulnerability: Vulnerability,
    pub target: Target,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<FindingEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub impact: Option<FindingImpact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<FindingRecommendation>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ==================== 輔助函數 ====================

fn default_version() -> String {
    "1.0".to_string()
}

fn default_get_method() -> String {
    "GET".to_string()
}

impl FindingPayload {
    /// 創建新的 FindingPayload 實例
    pub fn new(
        finding_id: String,
        task_id: String,
        scan_id: String,
        status: FindingStatus,
        vulnerability: Vulnerability,
        target: Target,
    ) -> Self {
        let now = Utc::now();
        Self {
            finding_id,
            task_id,
            scan_id,
            status,
            vulnerability,
            target,
            strategy: None,
            evidence: None,
            impact: None,
            recommendation: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// 驗證必要字段格式
    pub fn validate(&self) -> Result<(), String> {
        if !self.finding_id.starts_with("finding_") {
            return Err("finding_id must start with 'finding_'".to_string());
        }
        if !self.task_id.starts_with("task_") {
            return Err("task_id must start with 'task_'".to_string());
        }
        if !self.scan_id.starts_with("scan_") {
            return Err("scan_id must start with 'scan_'".to_string());
        }
        Ok(())
    }

    /// 更新時間戳
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

impl Default for FindingStatus {
    fn default() -> Self {
        FindingStatus::New
    }
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Medium
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Confidence::Tentative
    }
}
'''
        return content
    
    def generate_go(self):
        """生成 Go schema"""
        print("\n🐹 生成 Go schema...")
        
        go_config = self.config['generation_config']['go']
        target_dir = Path(go_config['target_dir'])
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成 Go schema
        self._generate_go_structs(target_dir)
        
        print(f"✅ Go schema 已生成至: {target_dir}")
    
    def _generate_go_structs(self, target_dir: Path):
        """生成 Go 結構定義"""
        content = '''// AIVA Go Schema - 自動生成
// 版本: {version}
// 基於 core_schema_sot.yaml 作為單一事實來源
//
// ⚠️  此檔案自動生成，請勿手動修改
// 📅 最後更新: {timestamp}

package schemas

import (
	"time"
)

// ==================== 枚舉類型 ====================

type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
	SeverityInfo     Severity = "info"
)

type Confidence string

const (
	ConfidenceConfirmed Confidence = "confirmed"
	ConfidenceFirm      Confidence = "firm"
	ConfidenceTentative Confidence = "tentative"
)

type FindingStatus string

const (
	FindingStatusNew           FindingStatus = "new"
	FindingStatusConfirmed     FindingStatus = "confirmed"
	FindingStatusFalsePositive FindingStatus = "false_positive"
	FindingStatusFixed         FindingStatus = "fixed"
	FindingStatusIgnored       FindingStatus = "ignored"
)

// ==================== 核心結構定義 ====================

type MessageHeader struct {{
	MessageID     string    `json:"message_id"`
	TraceID       string    `json:"trace_id"`
	CorrelationID *string   `json:"correlation_id,omitempty"`
	SourceModule  string    `json:"source_module"`
	Timestamp     time.Time `json:"timestamp"`
	Version       string    `json:"version"`
}}

type Target struct {{
	URL       string                 `json:"url"`
	Parameter *string                `json:"parameter,omitempty"`
	Method    string                 `json:"method"`
	Headers   map[string]string      `json:"headers"`
	Params    map[string]interface{{}} `json:"params"`
	Body      *string                `json:"body,omitempty"`
}}

type Vulnerability struct {{
	Name        string     `json:"name"`
	CWE         *string    `json:"cwe,omitempty"`
	Severity    Severity   `json:"severity"`
	Confidence  Confidence `json:"confidence"`
	Description *string    `json:"description,omitempty"`
}}

type FindingEvidence struct {{
	Payload           *string  `json:"payload,omitempty"`
	ResponseTimeDelta *float64 `json:"response_time_delta,omitempty"`
	DBVersion         *string  `json:"db_version,omitempty"`
	Request           *string  `json:"request,omitempty"`
	Response          *string  `json:"response,omitempty"`
	Proof             *string  `json:"proof,omitempty"`
}}

type FindingImpact struct {{
	Description     *string  `json:"description,omitempty"`
	BusinessImpact  *string  `json:"business_impact,omitempty"`
	TechnicalImpact *string  `json:"technical_impact,omitempty"`
	AffectedUsers   *int     `json:"affected_users,omitempty"`
	EstimatedCost   *float64 `json:"estimated_cost,omitempty"`
}}

type FindingRecommendation struct {{
	Fix              *string  `json:"fix,omitempty"`
	Priority         *string  `json:"priority,omitempty"`
	RemediationSteps []string `json:"remediation_steps"`
	References       []string `json:"references"`
}}

// ==================== 主要 Payload 結構 ====================

type FindingPayload struct {{
	FindingID      string                  `json:"finding_id"`
	TaskID         string                  `json:"task_id"`
	ScanID         string                  `json:"scan_id"`
	Status         FindingStatus           `json:"status"`
	Vulnerability  Vulnerability           `json:"vulnerability"`
	Target         Target                  `json:"target"`
	Strategy       *string                 `json:"strategy,omitempty"`
	Evidence       *FindingEvidence        `json:"evidence,omitempty"`
	Impact         *FindingImpact          `json:"impact,omitempty"`
	Recommendation *FindingRecommendation  `json:"recommendation,omitempty"`
	Metadata       map[string]interface{{}} `json:"metadata"`
	CreatedAt      time.Time               `json:"created_at"`
	UpdatedAt      time.Time               `json:"updated_at"`
}}

// NewFindingPayload 創建新的 FindingPayload 實例
func NewFindingPayload(
	findingID, taskID, scanID string,
	status FindingStatus,
	vulnerability Vulnerability,
	target Target,
) *FindingPayload {{
	now := time.Now()
	return &FindingPayload{{
		FindingID:      findingID,
		TaskID:         taskID,
		ScanID:         scanID,
		Status:         status,
		Vulnerability:  vulnerability,
		Target:         target,
		Metadata:       make(map[string]interface{{}}),
		CreatedAt:      now,
		UpdatedAt:      now,
	}}
}}

// Touch 更新時間戳
func (fp *FindingPayload) Touch() {{
	fp.UpdatedAt = time.Now()
}}
'''.format(version=self.config.get('version', '1.0.0'), timestamp=self.timestamp)
        
        with open(target_dir / 'schemas.go', 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_python_init(self, target_dir: Path):
        """生成 Python __init__.py"""
        content = '''"""
AIVA Schema 統一導出模組
========================

此模組提供統一的 schema 導出介面
"""

from .base_types import *
from .findings import *

__all__ = [
    # 基礎類型
    "MessageHeader",
    "Target", 
    "Vulnerability",
    
    # 發現相關
    "FindingPayload",
    "FindingEvidence",
    "FindingImpact", 
    "FindingRecommendation",
]
'''
        
        with open(target_dir / '__init__.py', 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='AIVA Schema Generator - 單一事實來源工具')
    parser.add_argument(
        '--sot-file', 
        default='services/aiva_common/core_schema_sot.yaml',
        help='SOT YAML 文件路径'
    )
    parser.add_argument(
        '--lang',
        choices=['python', 'rust', 'go', 'all'],
        default='all',
        help='生成特定語言或全部'
    )
    
    args = parser.parse_args()
    
    sot_path = Path(args.sot_file)
    if not sot_path.exists():
        print(f"❌ SOT 文件不存在: {sot_path}")
        sys.exit(1)
    
    generator = SchemaGenerator(sot_path)
    
    try:
        if args.lang == 'all':
            generator.generate_all()
        elif args.lang == 'python':
            generator.generate_python()
        elif args.lang == 'rust':
            generator.generate_rust() 
        elif args.lang == 'go':
            generator.generate_go()
            
        print(f"\n🎉 Schema 生成完成！")
        print(f"📋 基於 SOT: {sot_path}")
        print(f"🔧 生成語言: {args.lang}")
        
    except Exception as e:
        print(f"❌ 生成失敗: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()