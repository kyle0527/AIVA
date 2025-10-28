// AIVA Rust Schema 模組
// ==========================================
//
// 這個模組提供 AIVA 系統中使用的標準化 Schema 定義
// 基於 core_schema_sot.yaml 自動生成，確保跨語言一致性
//
// 主要結構:
// - MessageHeader: 統一訊息標頭
// - AivaMessage: AIVA 統一訊息格式
// - FunctionTaskPayload: 功能任務載荷
// - FindingPayload: 漏洞發現載荷
// - Target, Vulnerability: 基礎類型

pub mod generated;

// 重新導出主要類型以便使用
pub use generated::{
    // 基礎類型
    MessageHeader,
    Target,
    Vulnerability,
    
    // 訊息通訊
    AivaMessage,
    
    // 任務管理
    FunctionTaskPayload,
    FunctionTaskTarget,
    FunctionTaskContext,
    FunctionTaskTestConfig,
    
    // 發現結果
    FindingPayload,
    FindingEvidence,
    FindingImpact,
    FindingRecommendation,
    
    // 枚舉類型
    Severity,
    Confidence,
    FindingStatus,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_message_header_serialization() {
        let header = MessageHeader {
            message_id: "test-123".to_string(),
            trace_id: "trace-456".to_string(),
            correlation_id: Some("corr-789".to_string()),
            source_module: "test_module".to_string(),
            timestamp: chrono::Utc::now(),
            version: "1.0.0".to_string(),
        };

        let json = serde_json::to_string(&header).unwrap();
        let deserialized: MessageHeader = serde_json::from_str(&json).unwrap();
        
        assert_eq!(header.message_id, deserialized.message_id);
        assert_eq!(header.trace_id, deserialized.trace_id);
        assert_eq!(header.source_module, deserialized.source_module);
    }

    #[test]
    fn test_finding_payload_creation() {
        let finding = FindingPayload {
            finding_id: "find-001".to_string(),
            task_id: "task-001".to_string(),
            scan_id: "scan-001".to_string(),
            status: "new".to_string(),
            vulnerability: Vulnerability {
                name: "SQL Injection".to_string(),
                cwe: Some("CWE-89".to_string()),
                severity: "high".to_string(),
                confidence: "high".to_string(),
                description: Some("SQL injection vulnerability detected".to_string()),
            },
            target: Target {
                url: "https://example.com".to_string(),
                parameter: Some("id".to_string()),
                method: Some("POST".to_string()),
                headers: Some(std::collections::HashMap::new()),
                params: Some(std::collections::HashMap::new()),
                body: None,
            },
            strategy: Some("comprehensive".to_string()),
            evidence: None,
            impact: None,
            recommendation: None,
            metadata: Some(std::collections::HashMap::new()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        assert_eq!(finding.vulnerability.name, "SQL Injection");
        assert_eq!(finding.target.url, "https://example.com");
    }
}