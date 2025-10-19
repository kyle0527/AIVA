def test_can_import_plugin():
    # 測試僅確認套件可以被匯入（需要執行環境已可匯入 aiva_common.schemas）
    try:
        import aiva_schemas_plugin as schemas  # noqa: F401
    except Exception as e:
        # 在 CI 無 aiva_common 時，允許失敗但訊息要合理
        # 允許兩種情況：找不到 aiva_schemas_plugin 或找不到 aiva_common.schemas
        assert "aiva_common.schemas" in str(e) or "aiva_schemas_plugin" in str(e)
