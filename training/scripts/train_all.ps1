# AIVA 訓練快速啟動腳本（PowerShell 版本）

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "🎓 AIVA 知識蒸餾訓練 - 完整流程" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""
Write-Host "此腳本將依序執行：" -ForegroundColor Yellow
Write-Host "  1. 構建安全詞彙表" -ForegroundColor White
Write-Host "  2. 生成蒸餾訓練數據集" -ForegroundColor White
Write-Host "  3. 訓練 AIVA 5M Student Model" -ForegroundColor White
Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

$continue = Read-Host "按 Enter 鍵開始訓練，或輸入 'n' 取消"
if ($continue -eq 'n') {
    Write-Host "❌ 訓練已取消" -ForegroundColor Red
    exit
}

$ErrorActionPreference = "Stop"
$scriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path

try {
    # 步驟 1: 構建詞彙表
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    Write-Host "▶ 步驟 1/3: 構建安全詞彙表" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    
    python "$scriptsDir\build_security_vocabulary.py" `
        --docs-dir "C:\Users\User\Downloads\新增資料夾" `
        --output-dir "$scriptsDir\..\data\security_vocabulary"
    
    if ($LASTEXITCODE -ne 0) { throw "步驟 1 失敗" }
    
    # 步驟 2: 生成數據集
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    Write-Host "▶ 步驟 2/3: 生成蒸餾數據集" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    
    python "$scriptsDir\generate_distillation_dataset.py" `
        --vocab-dir "$scriptsDir\..\data\security_vocabulary" `
        --output-dir "$scriptsDir\..\data\distillation_dataset" `
        --samples-per-type 100
    
    if ($LASTEXITCODE -ne 0) { throw "步驟 2 失敗" }
    
    # 步驟 3: 訓練模型
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    Write-Host "▶ 步驟 3/3: 訓練 Student Model" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    
    python "$scriptsDir\train_student_model.py"
    
    if ($LASTEXITCODE -ne 0) { throw "步驟 3 失敗" }
    
    # 完成
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Green
    Write-Host ("=" * 59) -ForegroundColor Green
    Write-Host "✅ 訓練完成！" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Green
    Write-Host ("=" * 59) -ForegroundColor Green
    Write-Host ""
    Write-Host "📁 輸出位置:" -ForegroundColor Yellow
    Write-Host "  - 詞彙表: $scriptsDir\..\data\security_vocabulary" -ForegroundColor White
    Write-Host "  - 數據集: $scriptsDir\..\data\distillation_dataset" -ForegroundColor White
    Write-Host "  - 模型: $scriptsDir\..\models" -ForegroundColor White
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    Write-Host "下一步:" -ForegroundColor Yellow
    Write-Host "  1. 查看訓練報告: cat ..\data\distillation_dataset\dataset_report.md" -ForegroundColor White
    Write-Host "  2. 部署模型到 services 目錄" -ForegroundColor White
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    
} catch {
    Write-Host ""
    Write-Host "❌ 訓練失敗: $_" -ForegroundColor Red
    Write-Host "請檢查錯誤訊息並修正後重試" -ForegroundColor Yellow
    exit 1
}
