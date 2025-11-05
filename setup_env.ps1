# AIVA Python 環境設定腳本
# 執行此腳本以設定正確的 PYTHONPATH

$env:PYTHONPATH = "C:\D\fold7\AIVA-git;C:\D\fold7\AIVA-git\services;C:\D\fold7\AIVA-git\services\features;C:\D\fold7\AIVA-git\services\aiva_common;C:\D\fold7\AIVA-git\api;C:\D\fold7\AIVA-git\config"
Write-Host "✅ PYTHONPATH 已設定為:" -ForegroundColor Green
Write-Host $env:PYTHONPATH -ForegroundColor Yellow

# 驗證設定
python -c "import sys; print('\n📋 當前 Python 路徑:'); [print(f'  • {p}') for p in sys.path[:10]]"
