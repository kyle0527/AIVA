# AIVA 多語言環境設置腳本 (現代化版本)
# 日期: 2025-11-25
# 用途: 一次性設置所有語言的開發環境 (使用全域 Python 環境)

param(
    [switch]$SkipPython,
    [switch]$SkipNode,
    [switch]$SkipGo,
    [switch]$SkipRust,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$OriginalLocation = Get-Location

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🔧 AIVA 多語言環境設置" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "使用全域環境策略 - 2025-11-25" -ForegroundColor Gray
Write-Host ""

# 切換到專案根目錄
$ProjectRoot = "C:\D\fold7\AIVA-git"
Set-Location $ProjectRoot

# ============================================
# 1. Python 環境
# ============================================
if (-not $SkipPython) {
    Write-Host "🐍 設置 Python 環境..." -ForegroundColor Cyan
    
    # 檢查 Python
    $pythonInstalled = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonInstalled) {
        Write-Host "   ❌ Python 未安裝" -ForegroundColor Red
        Write-Host "   請安裝 Python 3.9+: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host ""
    } else {
        $pythonVersion = python --version 2>&1
        Write-Host "   ✓ 檢測到: $pythonVersion" -ForegroundColor Green
        
        Write-Host "   📦 升級核心工具..." -ForegroundColor Yellow
        python -m pip install --upgrade pip setuptools wheel --quiet
        
        Write-Host "   📦 安裝 AIVA 套件 (可編輯模式)..." -ForegroundColor Yellow
        python -m pip install -e . --quiet
        
        Write-Host "   📦 安裝開發依賴..." -ForegroundColor Yellow
        if (Test-Path "requirements.txt") {
            python -m pip install -r requirements.txt --quiet
        }
        
        # 安裝 Playwright (如果需要)
        Write-Host "   🎭 安裝 Playwright..." -ForegroundColor Yellow
        python -m pip install playwright --quiet
        python -m playwright install chromium --quiet
        
        Write-Host "   ✅ Python 環境設置完成" -ForegroundColor Green
        Write-Host ""
    }
}

# ============================================
# 2. Node.js 環境
# ============================================
if (-not $SkipNode) {
    Write-Host "🟢 設置 Node.js 環境..." -ForegroundColor Cyan
    
    $nodeInstalled = Get-Command node -ErrorAction SilentlyContinue
    if (-not $nodeInstalled) {
        Write-Host "   ⚠️  Node.js 未安裝 (可選)" -ForegroundColor Yellow
        Write-Host "   如需 TypeScript 掃描引擎，請安裝: https://nodejs.org/" -ForegroundColor Gray
        Write-Host "   建議版本: 20.x LTS" -ForegroundColor Gray
        Write-Host ""
    } else {
        $nodeVersion = node --version 2>&1
        Write-Host "   ✓ 檢測到: Node.js $nodeVersion" -ForegroundColor Green
        
        # 檢查並安裝 TypeScript 掃描引擎依賴
        $tsEnginePath = "services\scan\engines\typescript_engine"
        if (Test-Path $tsEnginePath) {
            Write-Host "   📦 安裝 TypeScript 引擎依賴..." -ForegroundColor Yellow
            Push-Location $tsEnginePath
            npm install --silent 2>$null
            Pop-Location
            Write-Host "   ✅ TypeScript 引擎已就緒" -ForegroundColor Green
        }
        
        Write-Host "   ✅ Node.js 環境設置完成" -ForegroundColor Green
        Write-Host ""
    }
}

# ============================================
# 3. Go 環境
# ============================================
if (-not $SkipGo) {
    Write-Host "🔵 設置 Go 環境..." -ForegroundColor Cyan
    
    $goInstalled = Get-Command go -ErrorAction SilentlyContinue
    if (-not $goInstalled) {
        Write-Host "   ⚠️  Go 未安裝 (可選)" -ForegroundColor Yellow
        Write-Host "   如需 Go 掃描引擎，請安裝: https://go.dev/dl/" -ForegroundColor Gray
        Write-Host "   建議版本: 1.21+" -ForegroundColor Gray
        Write-Host ""
    } else {
        $goVersion = go version 2>&1
        Write-Host "   ✓ 檢測到: $goVersion" -ForegroundColor Green
        
        # 查找所有 Go 模組並安裝依賴
        $goModules = Get-ChildItem -Path "services" -Filter "go.mod" -Recurse -ErrorAction SilentlyContinue
        if ($goModules.Count -gt 0) {
            Write-Host "   📦 發現 $($goModules.Count) 個 Go 模組" -ForegroundColor Yellow
            foreach ($mod in $goModules) {
                $modDir = $mod.Directory.FullName
                $relativePath = $modDir.Replace($ProjectRoot + "\", "")
                Write-Host "      • $relativePath" -ForegroundColor Gray
                
                Push-Location $modDir
                go mod download 2>$null
                go mod tidy 2>$null
                Pop-Location
            }
            Write-Host "   ✅ Go 模組依賴已安裝" -ForegroundColor Green
        }
        
        Write-Host "   ✅ Go 環境設置完成" -ForegroundColor Green
        Write-Host ""
    }
}

# ============================================
# 4. Rust 環境
# ============================================
if (-not $SkipRust) {
    Write-Host "🦀 設置 Rust 環境..." -ForegroundColor Cyan
    
    $cargoInstalled = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargoInstalled) {
        Write-Host "   ⚠️  Rust 未安裝 (可選)" -ForegroundColor Yellow
        Write-Host "   如需 Rust 組件，請安裝: https://rustup.rs/" -ForegroundColor Gray
        Write-Host ""
    } else {
        $rustVersion = rustc --version 2>&1
        Write-Host "   ✓ 檢測到: $rustVersion" -ForegroundColor Green
        
        # 檢查 Cargo.toml
        if (Test-Path "Cargo.toml") {
            Write-Host "   📦 安裝 Rust 依賴..." -ForegroundColor Yellow
            cargo build --release 2>$null
            Write-Host "   ✅ Rust 組件已構建" -ForegroundColor Green
        }
        
        Write-Host "   ✅ Rust 環境設置完成" -ForegroundColor Green
        Write-Host ""
    }
}

# ============================================
# 5. 驗證安裝
# ============================================
Write-Host "✓ 驗證環境設置..." -ForegroundColor Cyan

$validationResults = @()

# Python 驗證
try {
    python -c "import services.core.aiva_core; print('AIVA Core 可用')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $validationResults += @{ Component = "AIVA Core"; Status = "✅" }
    } else {
        $validationResults += @{ Component = "AIVA Core"; Status = "❌" }
    }
} catch {
    $validationResults += @{ Component = "AIVA Core"; Status = "❌" }
}

# 顯示驗證結果
Write-Host ""
Write-Host "驗證結果:" -ForegroundColor Yellow
foreach ($result in $validationResults) {
    Write-Host "   $($result.Status) $($result.Component)" -ForegroundColor $(if ($result.Status -eq "✅") { "Green" } else { "Red" })
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ 環境設置完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "   1. 啟動基礎設施: docker-compose up -d" -ForegroundColor White
Write-Host "   2. 啟動服務: python start_ai_service.py" -ForegroundColor White
Write-Host ""

Set-Location $OriginalLocation
