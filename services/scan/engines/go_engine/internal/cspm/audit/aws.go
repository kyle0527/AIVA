package audit

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/sirupsen/logrus"
)

// =====================================================================
// AWS Auditor - CSPM 雲端安全審計
// =====================================================================
// 來源: C:\Users\User\Downloads\新增資料夾 (6)\aws_audit.go
// 用途: 使用 AWS SDK 執行深度配置審計
// =====================================================================

// AWSAuditor 負責執行 AWS 環境的合規性檢查
type AWSAuditor struct {
	ctx    context.Context
	client *s3.Client // 這裡以 S3 為例，實際應包含多個 Service Client
}

// NewAWSAuditor 初始化 AWS 客戶端
// 這裡假設憑證已透過環境變數 (AWS_ACCESS_KEY_ID 等) 注入
// region: AWS 區域 (例如 "us-east-1")
func NewAWSAuditor(ctx context.Context, region string) (*AWSAuditor, error) {
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("unable to load SDK config: %w", err)
	}

	return &AWSAuditor{
		ctx:    ctx,
		client: s3.NewFromConfig(cfg),
	}, nil
}

// AuditS3Buckets 檢查 S3 存儲桶的公開存取權限
// 這比 Rust 只能探測 URL 更深入，因為它檢查的是 ACL 配置
// 返回: (風險 Bucket 列表, 錯誤)
func (a *AWSAuditor) AuditS3Buckets() ([]string, error) {
	var riskBuckets []string

	// 1. 列出所有 Buckets
	output, err := a.client.ListBuckets(a.ctx, &s3.ListBucketsInput{})
	if err != nil {
		return nil, fmt.Errorf("failed to list S3 buckets: %w", err)
	}

	logrus.Infof("[AWS Auditor] Found %d S3 buckets", len(output.Buckets))

	for _, bucket := range output.Buckets {
		name := *bucket.Name
		logrus.Debugf("[AWS Auditor] Auditing bucket: %s", name)

		// 2. 檢查 Bucket ACL 配置
		acl, err := a.client.GetBucketAcl(a.ctx, &s3.GetBucketAclInput{Bucket: &name})
		if err != nil {
			logrus.Warnf("Failed to get ACL for bucket %s: %v", name, err)
			continue
		}

		if isPublicACL(acl) {
			riskBuckets = append(riskBuckets, name)
			logrus.Warnf("[AWS Auditor] RISK: Bucket %s has public ACL", name)
			continue
		}

		// 3. 檢查 Public Access Block 配置
		pab, err := a.client.GetPublicAccessBlock(a.ctx, &s3.GetPublicAccessBlockInput{Bucket: &name})
		if err != nil {
			logrus.Debugf("No Public Access Block for bucket %s (may be default): %v", name, err)
			// 沒有 PAB 配置可能意味著使用預設設置,不一定是風險
		} else if pab.PublicAccessBlockConfiguration == nil ||
			!*pab.PublicAccessBlockConfiguration.BlockPublicAcls ||
			!*pab.PublicAccessBlockConfiguration.BlockPublicPolicy {
			riskBuckets = append(riskBuckets, name)
			logrus.Warnf("[AWS Auditor] RISK: Bucket %s allows public access", name)
		}
	}

	logrus.Infof("[AWS Auditor] Audit completed, found %d risk buckets", len(riskBuckets))
	return riskBuckets, nil
}

// isPublicACL 檢查 ACL 是否包含公開訪問權限
func isPublicACL(acl *s3.GetBucketAclOutput) bool {
	for _, grant := range acl.Grants {
		if grant.Grantee != nil && grant.Grantee.URI != nil {
			uri := *grant.Grantee.URI
			// 檢查是否授權給所有人或已驗證用戶
			if uri == "http://acs.amazonaws.com/groups/global/AllUsers" ||
				uri == "http://acs.amazonaws.com/groups/global/AuthenticatedUsers" {
				return true
			}
		}
	}
	return false
}

// =====================================================================
// 擴展: 其他 AWS 服務審計方法
// =====================================================================

// AuditIAMUsers 檢查 IAM 用戶權限配置
// 檢查項:
// - 是否有未使用的訪問密鑰
// - 密鑰輪換策略
// - 過度授權的策略
func (a *AWSAuditor) AuditIAMUsers() ([]string, error) {
	// TODO: 需要 IAM 客戶端
	logrus.Info("[AWS Auditor] IAM audit not yet implemented")
	return nil, nil
}

// AuditSecurityGroups 檢查 EC2 安全組規則
// 檢查項:
// - 是否有 0.0.0.0/0 開放的高風險端口 (22, 3389, 3306 等)
// - 過於寬鬆的出站規則
func (a *AWSAuditor) AuditSecurityGroups() ([]string, error) {
	// TODO: 需要 EC2 客戶端
	logrus.Info("[AWS Auditor] Security Group audit not yet implemented")
	return nil, nil
}

// AuditCloudTrail 檢查 CloudTrail 日誌配置
// 檢查項:
// - 是否啟用多區域日誌記錄
// - 日誌文件驗證
// - S3 存儲桶加密
func (a *AWSAuditor) AuditCloudTrail() ([]string, error) {
	// TODO: 需要 CloudTrail 客戶端
	logrus.Info("[AWS Auditor] CloudTrail audit not yet implemented")
	return nil, nil
}

// AuditKMSKeys 檢查 KMS 密鑰管理配置
// 檢查項:
// - 密鑰輪換策略
// - 密鑰訪問策略
// - 未使用的密鑰
func (a *AWSAuditor) AuditKMSKeys() ([]string, error) {
	// TODO: 需要 KMS 客戶端
	logrus.Info("[AWS Auditor] KMS audit not yet implemented")
	return nil, nil
}

// RunFullAudit 執行完整的 CIS Benchmark 審計
// 整合所有審計方法,生成綜合報告
func (a *AWSAuditor) RunFullAudit() (map[string][]string, error) {
	results := make(map[string][]string)

	// S3 審計
	if s3Risks, err := a.AuditS3Buckets(); err == nil {
		results["s3"] = s3Risks
	} else {
		logrus.Errorf("S3 audit failed: %v", err)
	}

	// IAM 審計
	if iamRisks, err := a.AuditIAMUsers(); err == nil {
		results["iam"] = iamRisks
	}

	// Security Group 審計
	if sgRisks, err := a.AuditSecurityGroups(); err == nil {
		results["security_groups"] = sgRisks
	}

	// CloudTrail 審計
	if ctRisks, err := a.AuditCloudTrail(); err == nil {
		results["cloudtrail"] = ctRisks
	}

	// KMS 審計
	if kmsRisks, err := a.AuditKMSKeys(); err == nil {
		results["kms"] = kmsRisks
	}

	logrus.Infof("[AWS Auditor] Full audit completed, scanned %d service categories", len(results))
	return results, nil
}
