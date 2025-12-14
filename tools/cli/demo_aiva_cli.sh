#!/usr/bin/env bash
# AIVA CLI 完整功能示範腳本

echo "🚀 AIVA CLI 系統功能完整示範"
echo "=================================="
echo ""

echo "1. 📊 查看系統統計資訊"
echo "--------------------------------"
python aiva_cli_implementation.py stats
echo ""

echo "2. 🎯 基本路徑執行 - 到達AI模型管理器"
echo "--------------------------------"
python aiva_cli_implementation.py reach ai_model_manager --dry-run
echo ""

echo "3. 🚀 指定起點執行 - 從生物訓練器到模型訓練器"
echo "--------------------------------"
python aiva_cli_implementation.py reach model_trainer --from=scalable_bio_trainer --dry-run
echo ""

echo "4. 📋 查看多路徑選項"
echo "--------------------------------"
python aiva_cli_implementation.py paths scalable_bio_trainer authz_mapper
echo ""

echo "5. 🎲 選擇特定路徑執行"
echo "--------------------------------"
python aiva_cli_implementation.py reach authz_mapper --from=scalable_bio_trainer --path=1 --dry-run
echo ""

echo "6. 📝 流程詳細資訊查看"
echo "--------------------------------"
python aiva_cli_implementation.py flow show 11
echo ""

echo "7. 🔍 按模組列出流程"
echo "--------------------------------"
python aiva_cli_implementation.py flow list --module=cognitive_core
echo ""

echo "8. ▶️ 執行特定流程"
echo "--------------------------------"
python aiva_cli_implementation.py flow run 11 --dry-run
echo ""

echo "✅ 示範完成！CLI系統運行正常。"
echo ""
echo "💡 更多用法請參考 AIVA_CLI_USAGE_GUIDE.md"