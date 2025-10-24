# aiva-schemas-plugin

本插件將整個專案對 `schemas.py` 的依賴**集中**到單一對外入口：`aiva_schemas_plugin`，
並以**轉接層**的方式直接 re-export `aiva_common.schemas` 中的所有公開成員。

這樣做的優點：
- 專案中所有模組只需要 `from aiva_schemas_plugin import ...`；
- 其他子專案（function/、scan/、integration/ 等目錄下的 `schemas.py`）可以安全移除；
- 後續如需真的把 `aiva_common/schemas.py` 移動到獨立倉庫，僅需在本插件內部調整，不再全倉大規模改動。

> **注意**：現在的實作是「轉接層」（adapter）。也就是說，**資料模型仍定義在 `aiva_common/schemas.py`**。  
> 當你準備把模型真正搬到外部套件時，只要把本插件中的 `__init__.py` 改為內建定義或指向新位置即可。

## 安裝（本地開發）

```bash
pip install -e ./aiva-schemas-plugin
```

## 使用方式（遷移後的統一寫法）

```python
from aiva_schemas_plugin import SomeModel, AnotherType
# 或保留原先的點位存取習慣：
import aiva_schemas_plugin as schemas
item = schemas.SomeModel(...)
```

## 移除其他 `schemas.py` 並批量改寫匯入

請在專案根目錄執行：

```bash
python ./aiva-schemas-plugin/scripts/refactor_imports_and_cleanup.py --repo-root ./services
```

該腳本會：
1. 備份將被修改的檔案；
2. 將所有 `from XXX.schemas import Y`、`from .schemas import Y`、`import XXX.schemas as schemas` 等常見寫法改成 `from aiva_schemas_plugin import Y` 或 `import aiva_schemas_plugin as schemas`；
3. **刪除**除了 `services/aiva_common/schemas.py` 之外的其他 `schemas.py` 檔案（清單內含常見位置）；
4. 輸出修改摘要。

## 測試

```bash
pytest -q
```
