/**
 * AIVA Common TypeScript - 統一 TypeScript AI 組件庫
 * 
 * 提供與 Python aiva_common 對應的 TypeScript 實現
 * 支持 AIVA 系統中 TypeScript 模組的 AI 功能需求
 * 
 * 架構位置: services/features/common/typescript/aiva_common_ts/
 * 用途: TypeScript 模組的 AI 能力評估和經驗管理
 */

// ============================================================================
// AI 組件導出
// ============================================================================

// Capability Evaluator
export { 
  default as AIVACapabilityEvaluator,
  createCapabilityEvaluator,
  type CapabilityEvidence,
  type EvaluationMetric,
  type BenchmarkTest,
  type CapabilityAssessment,
  type CapabilityEvaluatorConfig,
  EvidenceType,
  MetricType,
  BenchmarkType,
  EvaluationDimension
} from './capability-evaluator';

// Experience Manager
export {
  default as AIVAExperienceManager,
  createExperienceManager,
  type ExperienceSample,
  type LearningSession,
  type ExperienceFilter,
  type ExperienceStatistics,
  type ExperienceManagerConfig,
  SessionType,
  SessionStatus,
  SortBy,
  StorageBackend
} from './experience-manager';

// ============================================================================
// 版本和元數據
// ============================================================================

export const VERSION = '1.0.0';
export const DESCRIPTION = 'AIVA Common TypeScript AI Components';
export const COMPATIBLE_WITH_PYTHON_VERSION = '1.0.0';

// ============================================================================
// 工具函數
// ============================================================================

/**
 * 檢查與 Python aiva_common 的兼容性
 */
export function checkPythonCompatibility(pythonVersion: string): boolean {
  return pythonVersion === COMPATIBLE_WITH_PYTHON_VERSION;
}

/**
 * 獲取 TypeScript AI 組件信息
 */
export function getComponentInfo(): Record<string, any> {
  return {
    name: 'aiva_common_ts',
    version: VERSION,
    description: DESCRIPTION,
    components: [
      'AIVACapabilityEvaluator',
      'AIVAExperienceManager'
    ],
    compatibleWithPython: COMPATIBLE_WITH_PYTHON_VERSION,
    features: [
      'AI 能力評估',
      '經驗樣本管理',
      '強化學習支持',
      '連續監控',
      '質量評估',
      '統計分析'
    ]
  };
}

/**
 * 創建默認的 AI 組件配置
 */
export function createDefaultAIConfig(): {
  capabilityEvaluator: any;
  experienceManager: any;
} {
  return {
    capabilityEvaluator: {
      evaluatorId: `ts_evaluator_${Date.now()}`,
      continuousMonitoring: true,
      benchmarkTimeout: 30000
    },
    experienceManager: {
      managerId: `ts_exp_mgr_${Date.now()}`,
      storageBackend: 'memory',
      deduplicationEnabled: true,
      autoCleanup: true
    }
  };
}

console.log(`[AIVA Common TypeScript] Loaded version ${VERSION}`);