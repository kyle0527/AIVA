/**
 * AIVA TypeScript Capability Evaluator
 * 
 * TypeScript 版本的能力評估器，對應 aiva_common.ai.capability_evaluator
 * 提供與 Python 版本一致的 AI 能力評估功能
 * 
 * 架構位置: services/features/common/typescript/aiva_common_ts/
 * 用途: 支持 TypeScript 模組的 AI 能力評估需求
 */

import { EventEmitter } from 'events';

// ============================================================================
// 基礎類型定義 (對應 Python 的 dataclass) - 與 aiva_common.schemas.capability.CapabilityInfo 一致
// ============================================================================

export interface CapabilityInfo {
  // 基本信息
  id: string;
  name: string;
  description?: string;
  version: string;
  
  // 技術信息
  language: string; // 對應 ProgrammingLanguage enum
  entrypoint: string;
  topic: string;
  
  // 接口定義
  inputs?: Array<{
    name: string;
    type: string;
    required: boolean;
    description?: string;
  }>;
  outputs?: Array<{
    name: string;
    type: string;
    description?: string;
  }>;
  
  // 依賴與前置條件
  prerequisites?: string[];
  dependencies?: string[];
  
  // 元數據
  tags?: string[];
  status: string; // 對應 TaskStatus enum
  
  // 時間戳
  created_at?: Date;
  updated_at?: Date;
}

export interface CapabilityScorecard {
  capability_id: string;
  
  // 7日性能指標
  success_rate_7d: number;
  avg_latency_ms: number;
  availability_7d: number;
  usage_count_7d: number;
  
  // 其他指標
  overall_score: number;
  dimension_scores: Record<string, number>;
  last_updated_at?: Date;
  recommendations: string[];
}

export interface CapabilityEvidence {
  evidenceId: string;
  capabilityId: string;
  evidenceType: EvidenceType;
  data: Record<string, any>;
  confidence: number;
  timestamp: Date;
  source: string;
  metadata: Record<string, any>;
}

export interface EvaluationMetric {
  metricId: string;
  name: string;
  metricType: MetricType;
  value: number;
  unit: string;
  timestamp: Date;
  context: Record<string, any>;
}

export interface BenchmarkTest {
  testId: string;
  name: string;
  testType: BenchmarkType;
  inputData: Record<string, any>;
  expectedOutput: Record<string, any>;
  actualOutput: Record<string, any> | null;
  success: boolean;
  executionTimeMs: number;
  error: string | null;
  metadata: Record<string, any>;
}

export interface CapabilityAssessment {
  assessmentId: string;
  capabilityId: string;
  overallScore: number;
  dimension: EvaluationDimension;
  evidence: CapabilityEvidence[];
  metrics: EvaluationMetric[];
  benchmarkResults: BenchmarkTest[];
  confidence: number;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  lastUpdated: Date;
  validUntil: Date;
}

// ============================================================================
// 枚舉定義 (對應 Python 的 Enum)
// ============================================================================

export enum EvidenceType {
  EXECUTION_LOG = "execution_log",
  PERFORMANCE_METRIC = "performance_metric",
  USER_FEEDBACK = "user_feedback",
  SYSTEM_OBSERVATION = "system_observation",
  BENCHMARK_RESULT = "benchmark_result",
  COMPARATIVE_ANALYSIS = "comparative_analysis"
}

export enum MetricType {
  ACCURACY = "accuracy",
  PERFORMANCE = "performance",
  RELIABILITY = "reliability",
  EFFICIENCY = "efficiency",
  USABILITY = "usability",
  SECURITY = "security"
}

export enum BenchmarkType {
  UNIT_TEST = "unit_test",
  INTEGRATION_TEST = "integration_test",
  PERFORMANCE_TEST = "performance_test",
  STRESS_TEST = "stress_test",
  REGRESSION_TEST = "regression_test",
  COMPARATIVE_TEST = "comparative_test"
}

export enum EvaluationDimension {
  FUNCTIONALITY = "functionality",
  PERFORMANCE = "performance",
  RELIABILITY = "reliability",
  USABILITY = "usability",
  EFFICIENCY = "efficiency",
  MAINTAINABILITY = "maintainability",
  PORTABILITY = "portability",
  SECURITY = "security"
}

// ============================================================================
// 配置類型
// ============================================================================

export interface CapabilityEvaluatorConfig {
  evaluatorId: string;
  dataDirectory: string;
  evidenceRetentionDays: number;
  benchmarkTimeout: number;
  continuousMonitoring: boolean;
  alertThresholds: Record<string, number>;
  integrationEndpoints: Record<string, string>;
}

// ============================================================================
// 主要評估器類別
// ============================================================================

export class AIVACapabilityEvaluator extends EventEmitter {
  private config: CapabilityEvaluatorConfig;
  private evidenceStore: Map<string, CapabilityEvidence[]>;
  private assessmentCache: Map<string, CapabilityAssessment>;
  private benchmarkSuite: Map<string, BenchmarkTest[]>;
  private monitoringActive: boolean;

  constructor(config: CapabilityEvaluatorConfig) {
    super();
    this.config = config;
    this.evidenceStore = new Map();
    this.assessmentCache = new Map();
    this.benchmarkSuite = new Map();
    this.monitoringActive = false;
    
    this.initializeEvaluator();
  }

  // ========================================================================
  // 初始化和配置
  // ========================================================================

  private initializeEvaluator(): void {
    console.log(`[CapabilityEvaluator] Initializing evaluator: ${this.config.evaluatorId}`);
    
    // 設置連續監控
    if (this.config.continuousMonitoring) {
      this.startContinuousMonitoring();
    }
    
    this.emit('evaluator_initialized', {
      evaluatorId: this.config.evaluatorId,
      timestamp: new Date()
    });
  }

  // ========================================================================
  // 證據收集 (對應 Python 的 collect_evidence)
  // ========================================================================

  public async collectEvidence(
    capabilityId: string,
    evidenceType: EvidenceType,
    data: Record<string, any>,
    source: string = "typescript_evaluator"
  ): Promise<string> {
    try {
      const evidence: CapabilityEvidence = {
        evidenceId: this.generateId('evidence'),
        capabilityId,
        evidenceType,
        data: { ...data },
        confidence: this.calculateEvidenceConfidence(data, evidenceType),
        timestamp: new Date(),
        source,
        metadata: {
          collectorVersion: "1.0.0",
          environment: "typescript"
        }
      };

      // 存儲證據
      if (!this.evidenceStore.has(capabilityId)) {
        this.evidenceStore.set(capabilityId, []);
      }
      this.evidenceStore.get(capabilityId)!.push(evidence);

      // 發送事件
      this.emit('evidence_collected', evidence);

      console.log(`[CapabilityEvaluator] Evidence collected: ${evidence.evidenceId} for ${capabilityId}`);
      return evidence.evidenceId;

    } catch (error) {
      console.error(`[CapabilityEvaluator] Error collecting evidence: ${error}`);
      throw error;
    }
  }

  // ========================================================================
  // 基準測試 (對應 Python 的 run_benchmark_tests)
  // ========================================================================

  public async runBenchmarkTests(
    capabilityId: string,
    testSuite?: BenchmarkTest[]
  ): Promise<BenchmarkTest[]> {
    try {
      const tests = testSuite || this.benchmarkSuite.get(capabilityId) || [];
      const results: BenchmarkTest[] = [];

      for (const test of tests) {
        const startTime = Date.now();
        
        try {
          // 執行測試 (這裡是簡化版本，實際應該調用具體的測試邏輯)
          const result = await this.executeBenchmarkTest(test);
          const executionTime = Date.now() - startTime;

          const completedTest: BenchmarkTest = {
            ...test,
            actualOutput: result,
            success: this.validateTestResult(result, test.expectedOutput),
            executionTimeMs: executionTime,
            error: null
          };

          results.push(completedTest);

        } catch (error) {
          const executionTime = Date.now() - startTime;
          
          const failedTest: BenchmarkTest = {
            ...test,
            actualOutput: null,
            success: false,
            executionTimeMs: executionTime,
            error: error instanceof Error ? error.message : String(error)
          };

          results.push(failedTest);
        }
      }

      // 收集基準測試結果作為證據
      await this.collectEvidence(
        capabilityId,
        EvidenceType.BENCHMARK_RESULT,
        {
          totalTests: results.length,
          successfulTests: results.filter(t => t.success).length,
          averageExecutionTime: results.reduce((sum, t) => sum + t.executionTimeMs, 0) / results.length,
          results: results.map(t => ({
            testId: t.testId,
            success: t.success,
            executionTimeMs: t.executionTimeMs
          }))
        },
        'benchmark_runner'
      );

      return results;

    } catch (error) {
      console.error(`[CapabilityEvaluator] Error running benchmark tests: ${error}`);
      throw error;
    }
  }

  // ========================================================================
  // 評估生成 (對應 Python 的 generate_assessment)
  // ========================================================================

  public async generateAssessment(
    capabilityId: string,
    dimension: EvaluationDimension = EvaluationDimension.FUNCTIONALITY
  ): Promise<CapabilityAssessment> {
    try {
      // 獲取證據和指標
      const evidence = this.evidenceStore.get(capabilityId) || [];
      const benchmarkResults = await this.runBenchmarkTests(capabilityId);
      
      // 計算評估分數
      const overallScore = this.calculateOverallScore(evidence, benchmarkResults, dimension);
      
      // 生成評估報告
      const assessment: CapabilityAssessment = {
        assessmentId: this.generateId('assessment'),
        capabilityId,
        overallScore,
        dimension,
        evidence,
        metrics: this.generateMetrics(evidence, benchmarkResults),
        benchmarkResults,
        confidence: this.calculateAssessmentConfidence(evidence, benchmarkResults),
        strengths: this.identifyStrengths(evidence, benchmarkResults),
        weaknesses: this.identifyWeaknesses(evidence, benchmarkResults),
        recommendations: this.generateRecommendations(evidence, benchmarkResults, overallScore),
        lastUpdated: new Date(),
        validUntil: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24小時後過期
      };

      // 緩存評估結果
      this.assessmentCache.set(capabilityId, assessment);

      // 發送事件
      this.emit('assessment_generated', assessment);

      console.log(`[CapabilityEvaluator] Assessment generated for ${capabilityId}: ${overallScore.toFixed(2)}/100`);
      return assessment;

    } catch (error) {
      console.error(`[CapabilityEvaluator] Error generating assessment: ${error}`);
      throw error;
    }
  }

  // ========================================================================
  // 連續監控 (對應 Python 的 start_continuous_monitoring)
  // ========================================================================

  public startContinuousMonitoring(): void {
    if (this.monitoringActive) {
      console.warn('[CapabilityEvaluator] Monitoring already active');
      return;
    }

    this.monitoringActive = true;
    console.log('[CapabilityEvaluator] Starting continuous monitoring...');

    // 設置定期評估 (每10分鐘)
    const monitoringInterval = setInterval(async () => {
      try {
        await this.performPeriodicEvaluation();
      } catch (error) {
        console.error(`[CapabilityEvaluator] Error in periodic evaluation: ${error}`);
      }
    }, 10 * 60 * 1000);

    // 清理函數
    this.once('stop_monitoring', () => {
      clearInterval(monitoringInterval);
      this.monitoringActive = false;
      console.log('[CapabilityEvaluator] Continuous monitoring stopped');
    });

    this.emit('monitoring_started');
  }

  public stopContinuousMonitoring(): void {
    this.emit('stop_monitoring');
  }

  // ========================================================================
  // 輔助方法
  // ========================================================================

  private generateId(prefix: string): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    return `${prefix}_${timestamp}_${random}`;
  }

  private calculateEvidenceConfidence(data: Record<string, any>, type: EvidenceType): number {
    // 簡化的信心度計算
    let baseConfidence = 0.5;
    
    switch (type) {
      case EvidenceType.BENCHMARK_RESULT:
        baseConfidence = 0.9;
        break;
      case EvidenceType.PERFORMANCE_METRIC:
        baseConfidence = 0.8;
        break;
      case EvidenceType.SYSTEM_OBSERVATION:
        baseConfidence = 0.7;
        break;
      case EvidenceType.USER_FEEDBACK:
        baseConfidence = 0.6;
        break;
      default:
        baseConfidence = 0.5;
    }
    
    // 根據數據完整性調整
    const dataCompleteness = Object.keys(data).length / 10; // 假設理想情況下有10個字段
    return Math.min(baseConfidence * (0.5 + Math.min(dataCompleteness, 0.5)), 1.0);
  }

  private async executeBenchmarkTest(test: BenchmarkTest): Promise<Record<string, any>> {
    // 這裡應該實現實際的測試執行邏輯
    // 為了示例，返回模擬結果
    await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
    
    return {
      status: "completed",
      result: Math.random() > 0.2 ? "success" : "failure",
      executedAt: new Date().toISOString(),
      testData: test.inputData
    };
  }

  private validateTestResult(actual: Record<string, any>, expected: Record<string, any>): boolean {
    // 簡化的結果驗證
    if (actual.status !== "completed") return false;
    if (expected.expectedStatus && actual.result !== expected.expectedStatus) return false;
    return true;
  }

  private calculateOverallScore(
    evidence: CapabilityEvidence[],
    benchmarks: BenchmarkTest[],
    _dimension: EvaluationDimension
  ): number {
    if (evidence.length === 0 && benchmarks.length === 0) return 0;

    let score = 0;
    let weights = 0;

    // 基於證據的評分
    if (evidence.length > 0) {
      const evidenceScore = evidence.reduce((sum, e) => sum + e.confidence * 100, 0) / evidence.length;
      score += evidenceScore * 0.4;
      weights += 0.4;
    }

    // 基於基準測試的評分
    if (benchmarks.length > 0) {
      const successRate = benchmarks.filter(b => b.success).length / benchmarks.length;
      score += successRate * 100 * 0.6;
      weights += 0.6;
    }

    return weights > 0 ? score / weights : 0;
  }

  private calculateAssessmentConfidence(
    evidence: CapabilityEvidence[],
    benchmarks: BenchmarkTest[]
  ): number {
    const evidenceConfidence = evidence.length > 0 
      ? evidence.reduce((sum, e) => sum + e.confidence, 0) / evidence.length 
      : 0;
    
    const benchmarkConfidence = benchmarks.length > 0 ? 0.9 : 0;
    
    return Math.max(evidenceConfidence, benchmarkConfidence);
  }

  private generateMetrics(
    _evidence: CapabilityEvidence[],
    benchmarks: BenchmarkTest[]
  ): EvaluationMetric[] {
    const metrics: EvaluationMetric[] = [];

    // 基於基準測試的性能指標
    if (benchmarks.length > 0) {
      const avgExecutionTime = benchmarks.reduce((sum, b) => sum + b.executionTimeMs, 0) / benchmarks.length;
      const successRate = benchmarks.filter(b => b.success).length / benchmarks.length;

      metrics.push({
        metricId: this.generateId('metric'),
        name: 'Average Execution Time',
        metricType: MetricType.PERFORMANCE,
        value: avgExecutionTime,
        unit: 'ms',
        timestamp: new Date(),
        context: { benchmarkCount: benchmarks.length }
      });

      metrics.push({
        metricId: this.generateId('metric'),
        name: 'Success Rate',
        metricType: MetricType.RELIABILITY,
        value: successRate * 100,
        unit: '%',
        timestamp: new Date(),
        context: { totalTests: benchmarks.length, successful: benchmarks.filter(b => b.success).length }
      });
    }

    return metrics;
  }

  private identifyStrengths(evidence: CapabilityEvidence[], benchmarks: BenchmarkTest[]): string[] {
    const strengths: string[] = [];

    if (benchmarks.length > 0) {
      const successRate = benchmarks.filter(b => b.success).length / benchmarks.length;
      if (successRate > 0.8) {
        strengths.push('High benchmark test success rate');
      }

      const avgTime = benchmarks.reduce((sum, b) => sum + b.executionTimeMs, 0) / benchmarks.length;
      if (avgTime < 1000) {
        strengths.push('Fast execution performance');
      }
    }

    if (evidence.length > 5) {
      strengths.push('Rich evidence collection');
    }

    return strengths;
  }

  private identifyWeaknesses(evidence: CapabilityEvidence[], benchmarks: BenchmarkTest[]): string[] {
    const weaknesses: string[] = [];

    if (benchmarks.length > 0) {
      const failureRate = benchmarks.filter(b => !b.success).length / benchmarks.length;
      if (failureRate > 0.2) {
        weaknesses.push('High benchmark failure rate');
      }

      const avgTime = benchmarks.reduce((sum, b) => sum + b.executionTimeMs, 0) / benchmarks.length;
      if (avgTime > 5000) {
        weaknesses.push('Slow execution performance');
      }
    }

    if (evidence.length < 3) {
      weaknesses.push('Limited evidence collection');
    }

    return weaknesses;
  }

  private generateRecommendations(
    evidence: CapabilityEvidence[],
    benchmarks: BenchmarkTest[],
    overallScore: number
  ): string[] {
    const recommendations: string[] = [];

    if (overallScore < 60) {
      recommendations.push('Consider comprehensive capability improvement');
    }

    if (benchmarks.length === 0) {
      recommendations.push('Implement benchmark testing suite');
    }

    if (evidence.length < 5) {
      recommendations.push('Increase evidence collection frequency');
    }

    if (benchmarks.some(b => b.executionTimeMs > 10000)) {
      recommendations.push('Optimize performance for long-running operations');
    }

    return recommendations;
  }

  private async performPeriodicEvaluation(): Promise<void> {
    console.log('[CapabilityEvaluator] Performing periodic evaluation...');
    
    // 檢查所有已緩存的能力
    for (const [capabilityId, assessment] of this.assessmentCache.entries()) {
      // 檢查是否需要重新評估
      if (assessment.validUntil < new Date()) {
        try {
          await this.generateAssessment(capabilityId, assessment.dimension);
          console.log(`[CapabilityEvaluator] Updated assessment for capability: ${capabilityId}`);
        } catch (error) {
          console.error(`[CapabilityEvaluator] Failed to update assessment for ${capabilityId}: ${error}`);
        }
      }
    }

    this.emit('periodic_evaluation_completed', {
      timestamp: new Date(),
      evaluatedCapabilities: this.assessmentCache.size
    });
  }

  // ========================================================================
  // 公開 API 方法
  // ========================================================================

  public getAssessment(capabilityId: string): CapabilityAssessment | null {
    return this.assessmentCache.get(capabilityId) || null;
  }

  public getEvidence(capabilityId: string): CapabilityEvidence[] {
    return this.evidenceStore.get(capabilityId) || [];
  }

  public getAllCapabilities(): string[] {
    return Array.from(new Set([
      ...this.evidenceStore.keys(),
      ...this.assessmentCache.keys()
    ]));
  }

  public getEvaluatorStatus(): Record<string, any> {
    return {
      evaluatorId: this.config.evaluatorId,
      monitoringActive: this.monitoringActive,
      trackedCapabilities: this.getAllCapabilities().length,
      totalEvidence: Array.from(this.evidenceStore.values()).reduce((sum, arr) => sum + arr.length, 0),
      cachedAssessments: this.assessmentCache.size,
      uptime: Date.now() - (this as any).startTime || 0
    };
  }
}

// ============================================================================
// 工廠函數 (對應 Python 的 factory function)
// ============================================================================

export function createCapabilityEvaluator(config?: Partial<CapabilityEvaluatorConfig>): AIVACapabilityEvaluator {
  const defaultConfig: CapabilityEvaluatorConfig = {
    evaluatorId: `ts_evaluator_${Date.now()}`,
    dataDirectory: './data/capability_evaluation',
    evidenceRetentionDays: 30,
    benchmarkTimeout: 30000,
    continuousMonitoring: true,
    alertThresholds: {
      lowPerformance: 60,
      highFailureRate: 0.3,
      slowExecution: 5000
    },
    integrationEndpoints: {}
  };

  const finalConfig = { ...defaultConfig, ...config };
  return new AIVACapabilityEvaluator(finalConfig);
}

// ============================================================================
// 導出
// ============================================================================

export default AIVACapabilityEvaluator;