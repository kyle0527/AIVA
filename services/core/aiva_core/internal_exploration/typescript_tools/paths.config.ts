// paths.config.ts - TypeScript 路徑配置
// 自動從 Python paths.py 生成

import * as path from 'path';
import * as fs from 'fs';

// 獲取專案根目錄
// 當前文件: services/core/aiva_core/internal_exploration/typescript_tools/paths.config.ts
// 向上 5 層到達專案根目錄
const PROJECT_ROOT = path.resolve(__dirname, '../../../..');
const SERVICES_ROOT = path.join(PROJECT_ROOT, 'services');
const INTEGRATION_ROOT = path.join(SERVICES_ROOT, 'integration');
const INTEGRATION_DATA_ROOT = path.join(INTEGRATION_ROOT, 'data');

// Internal Exploration 輸出路徑
export const PATHS = {
    projectRoot: PROJECT_ROOT,
    servicesRoot: SERVICES_ROOT,
    integrationRoot: INTEGRATION_ROOT,
    integrationDataRoot: INTEGRATION_DATA_ROOT,
    
    // Internal Exploration 路徑
    internalExplorationData: path.join(INTEGRATION_DATA_ROOT, 'internal_exploration'),
    analysisResultsDir: path.join(INTEGRATION_DATA_ROOT, 'internal_exploration', 'analysis_results'),
    analysisHistoryDir: path.join(INTEGRATION_DATA_ROOT, 'internal_exploration', 'analysis_history'),
    selfHealingDir: path.join(INTEGRATION_DATA_ROOT, 'internal_exploration', 'self_healing'),
    
    // 其他整合模組路徑
    attackPathsDir: path.join(INTEGRATION_DATA_ROOT, 'attack_paths'),
    experiencesDir: path.join(INTEGRATION_DATA_ROOT, 'experiences'),
    trainingDataDir: path.join(INTEGRATION_DATA_ROOT, 'training'),
};

// 環境變量控制
export const USE_INTEGRATED_PATHS = process.env.AIVA_USE_INTEGRATED_PATHS !== 'false';

// 確保目錄存在
export function ensureDirectories(): void {
    const dirs = [
        PATHS.internalExplorationData,
        PATHS.analysisResultsDir,
        PATHS.analysisHistoryDir,
        PATHS.selfHealingDir,
        PATHS.attackPathsDir,
        PATHS.experiencesDir,
        PATHS.trainingDataDir,
    ];
    
    for (const dir of dirs) {
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
    }
}

// 獲取特定工具的輸出目錄
export function getAnalysisOutputDir(toolName: string = 'typescript'): string {
    if (toolName === 'self_healing') {
        return PATHS.selfHealingDir;
    }
    
    const toolDir = path.join(PATHS.analysisResultsDir, toolName);
    if (!fs.existsSync(toolDir)) {
        fs.mkdirSync(toolDir, { recursive: true });
    }
    return toolDir;
}

// 獲取默認輸出目錄（向後兼容）
export function getDefaultOutputDir(): string {
    if (USE_INTEGRATED_PATHS) {
        ensureDirectories();
        return getAnalysisOutputDir('typescript');
    }
    
    // 向後兼容：使用舊路徑
    return './analysis_output';
}

// 配置摘要
export function getConfigSummary() {
    return {
        projectRoot: PATHS.projectRoot,
        servicesRoot: PATHS.servicesRoot,
        integrationRoot: PATHS.integrationRoot,
        useIntegratedPaths: USE_INTEGRATED_PATHS,
        outputDirectories: {
            internalExploration: PATHS.internalExplorationData,
            analysisResults: PATHS.analysisResultsDir,
            analysisHistory: PATHS.analysisHistoryDir,
            selfHealing: PATHS.selfHealingDir,
            attackPaths: PATHS.attackPathsDir,
            experiences: PATHS.experiencesDir,
            training: PATHS.trainingDataDir,
        }
    };
}
