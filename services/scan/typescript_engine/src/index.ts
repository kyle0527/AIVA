/**
 * AIVA Scan Node - Playwright 動態掃描引擎
 * 日期: 2025-10-13
 * 功能: 使用 Playwright 進行動態網頁掃描 (CLI 模式)
 */

import { chromium, Browser } from 'playwright-core';
import { logger } from './utils/logger.js';
import { ScanService } from './services/scan-service.js';
import { DOMSecurityAnalyzer } from './dom-security-analyzer.js';

interface ScanTask {
  scan_id: string;
  target_url: string;
  max_depth: number;
  max_pages: number;
  enable_javascript: boolean;
}

let browser: Browser | null = null;
let scanService: ScanService | null = null;

async function initialize(): Promise<void> {
  logger.info('🚀 初始化 AIVA Scan Node (CLI Mode)...');

  // 啟動瀏覽器
  logger.info('🌐 啟動 Chromium 瀏覽器...');
  browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  logger.info('✅ 瀏覽器已啟動');

  // 初始化掃描服務
  scanService = new ScanService(browser);
  logger.info('✅ 初始化完成');
}

async function runScan(task: ScanTask): Promise<void> {
  if (!scanService || !browser) {
    throw new Error('ScanService 未初始化');
  }

  logger.info({ scan_id: task.scan_id }, '📥 開始掃描任務');

  // 執行掃描
  const result = await scanService.scan(task);

  // 執行 DOM 安全分析（如果啟用 JavaScript）
  if (task.enable_javascript && browser) {
    logger.info({ scan_id: task.scan_id }, '🔍 執行 DOM 安全分析...');
    const page = await browser.newPage();
    const domAnalyzer = new DOMSecurityAnalyzer(page);

    try {
      const domFindings = await domAnalyzer.analyze(task.target_url);

      // 將 DOM 發現添加到結果中
      if (domFindings.length > 0) {
        logger.info(
          { scan_id: task.scan_id, findings: domFindings.length },
          '✅ DOM 安全分析完成'
        );
        
        // 將發現添加到結果（可以擴展 result 結構）
        (result as any).dom_security_findings = domFindings;
      }
    } catch (error) {
      logger.error({ error }, '❌ DOM 安全分析失敗');
    } finally {
      await page.close();
    }
  }

  logger.info(
    { scan_id: task.scan_id, assets: result.assets.length },
    '✅ 掃描完成'
  );

  // 將結果輸出到 stdout
  console.log(JSON.stringify(result));
}

async function shutdown(): Promise<void> {
  logger.info('🛑 關閉服務...');

  if (browser) {
    await browser.close();
    logger.info('✅ 瀏覽器已關閉');
  }

  process.exit(0);
}

// 優雅關閉
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// 主程序
async function main(): Promise<void> {
  try {
    // 從 stdin 讀取任務
    let inputData = '';
    process.stdin.on('data', chunk => {
      inputData += chunk.toString();
    });

    process.stdin.on('end', async () => {
      try {
        const task: ScanTask = JSON.parse(inputData);
        await initialize();
        await runScan(task);
        await shutdown();
      } catch (error) {
        logger.error({ error }, '❌ 執行掃描失敗');
        process.exit(1);
      }
    });
  } catch (error) {
    logger.error({ error }, '❌ 啟動失敗');
    process.exit(1);
  }
}

main();
