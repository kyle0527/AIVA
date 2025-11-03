## 重複資料模型定義問題

- **Target** -- ✅ **已修正**：原先於 `services/scan/schemas.py` 定義的 `Target` 類別現已棄用，統一使用 `services.aiva_common.schemas.Target` 作為單一來源[\[1\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/schemas.py#L16-L21)。舊的定義僅保留相容性，建議在確認無舊代碼依賴後將其移除。

- **ScanScope** -- ❌ **未解決**：`ScanScope` 目前在 **共用** (`aiva_common.schemas`) 與 **掃描模組** (`services/scan/discovery_schemas.py`) 中各有一套定義，且結構不一致[\[2\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py#L64-L71)[\[3\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L26-L34)。這導致概念重複且可能混淆。建議合併為單一定義（或在其中一處引用另一處），並調整欄位以統一範圍設定的表示方式。

- **APIResponse** -- ✅ **已修正**：已統一使用 `aiva_common.schemas.APIResponse` 作為所有 API 端點的標準回應模型[\[4\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py#L31-L39)。先前各處自行定義回應結構的情況已消除，整個系統皆改為引用統一的 APIResponse 結構[\[5\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/reports/phase_2_execution_report_20251101.md#L48-L56)。建議持續沿用該標準模型，確保未來所有新 API 都遵循一致格式。

- **Finding** -- ❌ **未解決**：漏洞「發現」資料模型仍存在重複定義。掃描模組使用 `VulnerabilityFinding` 類（定義在 `discovery_schemas.py`）[\[6\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L156-L165)來描述漏洞結果，而共用合約中則有對應的 `FindingPayload` 定義[\[7\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/findings.py#L106-L115)。雖然證據(`FindingEvidence`)、影響(`FindingImpact`)、建議(`FindingRecommendation`)等子結構已統一引用共用定義[\[8\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L166-L174)，但核心 Finding 資料模型仍未整合。建議將掃描模組的漏洞發現結構與共用模型統一，或明確區分名稱以避免混淆，逐步消除重複。

- **Asset** -- ❌ **未解決**：資產資訊模型有多處定義且內容不一致。共用架構中 `Asset` 包含資產ID、類型、值等基本欄位[\[9\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py#L72-L80)；但在掃描模組中也定義了 `Asset`（URL、狀態碼、內容類型等）[\[10\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L70-L78)；Web 前端合約亦有自己的 `Asset` 介面定義（帶有風險分數等欄位）[\[11\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/web/contracts/aiva-web-contracts.ts#L126-L134)。目前尚未統一，名稱相同但欄位差異可能導致誤用。建議選定單一權威定義，將其他處的資產模型重命名或重構為該權威模型的一部分，確保各模組對 Asset 的定義一致。

- **Fingerprints** -- ❌ **未解決**：技術指紋資料結構重複定義且內容不一。在共用架構中 `Fingerprints` 僅包含 Web伺服器、框架、語言、WAF 等簡要欄位[\[12\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py#L91-L99)；但掃描模組中 `Fingerprints` 則定義了技術棧、服務、Header、Cookie 等詳細資訊[\[13\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L102-L110)。兩者用途重疊但結構不同，尚未整合。建議統一技術指紋模型，可將簡單欄位與詳細掃描結果合併為單一結構，或至少更名區分，以消除同名異構的情況。

## 重複枚舉名稱/值問題

- **RiskLevel** -- ❌ **未解決**：`RiskLevel` 枚舉的定義與 `Severity` (嚴重程度) 類似，皆使用了Critical/High/Medium/Low/Info等級[\[14\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py#L92-L100)[\[15\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py#L8-L13)。這兩者表意重疊，可能造成混亂。目前兩個枚舉仍並存，尚未合併。建議評估是否可直接以 `Severity` 取代 `RiskLevel`，或在系統中區分明確用途，避免維護兩組意義相近的等級定義。

- **DataFormat** -- ❌ **未解決**：`DataFormat` 枚舉定義了一系列資料格式/MIME類型（如JSON、XML、YAML等）[\[16\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py#L741-L749)。然而在報表或輸出中也存在類似的格式枚舉（例如前端合約的 `ReportFormat` 包含 json、html、xml 等）[\[17\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/web/contracts/aiva-web-contracts.ts#L220-L224)。目前這些格式定義未完全統一。建議將資料格式相關的枚舉集中管理，避免重複定義；可考慮讓 ReportFormat 直接使用 DataFormat 中對應值，或合併為單一枚舉以減少重複。

- **EncodingType** -- ❌ **未解決**：`EncodingType` 枚舉羅列了多種字元與傳輸編碼類型（UTF-8、Base64、URL_ENCODED 等）[\[18\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py#L756-L764)。目前尚未發現有另一處定義完全相同的列表，但需注意系統中是否存在功能重疊的常數或枚舉。例如，如有其他地方單獨定義了部分編碼類型，應予以統一。建議審視專案中對編碼形式的使用情況，將所有編碼相關常數匯總為單一枚舉或工具函式，以免重複。

- **Topic 枚舉別名** -- ❌ **未解決**：訊息佇列主題 (`Topic` 枚舉) 中仍存在多組名稱對應同一數值的情況。例如：`SCAN_START` 與 `TASK_SCAN_START` 皆對應 `"tasks.scan.start"`[\[19\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/modules.py#L250-L258)。這類別名當初為向後相容而保留，但目前仍未清理。建議在確保舊代碼不再使用舊名稱後，移除這些重複的枚舉成員，或至少將舊別名標註為已棄用，以減少未來維護混亂。

## 多語言契約統一問題

- **跨語言合約定義** -- ✅ **已修正**：多語言的資料合約定義現已統一來源。後端在 `aiva_common.schemas` 定義的模式會自動產生對應的 TypeScript/Go 等前端類型檔案[\[20\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/cli_generated/schemas.ts#L6-L14)（例如 `cli_generated/schemas.ts`），確保各語言間使用同一套結構。之前前後端各自維護重複結構的問題已大幅改善[\[21\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/cli_generated/schemas.ts#L8-L16)。未來建議嚴格遵循單一來源原則，由後端模式自動生成前端型別，避免手動維護不一致的合約，並逐步淘汰舊的手動定義檔以防混亂。

## 功能模組重複實作問題

- **Forensic Tools 工具模組** -- ✅ **已修正**：`forensic_tools` 模組重複實作的問題已解決。原本在 **features** 與 **integration** 模組下各有一份 `forensic_tools`，現在已將主要實作統一遷移至 `services/integration/capability/forensic_tools.py`，作為唯一維護的版本[\[22\]](https://github.com/kyle0527/AIVA/blob/b7d658f7d788a5a30b0060e2143f84407cd614e9/services/features/forensic_tools.py#L7-L9)。原 `services/features/forensic_tools.py` 已移至歸檔並加上棄用提示，只作轉發用途[\[23\]](https://github.com/kyle0527/AIVA/blob/b7d658f7d788a5a30b0060e2143f84407cd614e9/services/features/forensic_tools.py#L5-L9)。建議後續徹底移除舊檔案引用，在確認無人使用舊介面後刪除棄用模組，確保只有單一來源的法證工具實現。

------------------------------------------------------------------------

[\[1\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/schemas.py#L16-L21) schemas.py

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/schemas.py>

[\[2\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py#L64-L71) [\[4\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py#L31-L39) [\[9\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py#L72-L80) [\[12\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py#L91-L99) base.py

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/base.py>

[\[3\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L26-L34) [\[6\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L156-L165) [\[8\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L166-L174) [\[10\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L70-L78) [\[13\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py#L102-L110) discovery_schemas.py

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/scan/discovery_schemas.py>

[\[5\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/reports/phase_2_execution_report_20251101.md#L48-L56) phase_2_execution_report_20251101.md

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/reports/phase_2_execution_report_20251101.md>

[\[7\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/findings.py#L106-L115) findings.py

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/schemas/findings.py>

[\[11\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/web/contracts/aiva-web-contracts.ts#L126-L134) [\[17\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/web/contracts/aiva-web-contracts.ts#L220-L224) aiva-web-contracts.ts

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/web/contracts/aiva-web-contracts.ts>

[\[14\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py#L92-L100) [\[15\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py#L8-L13) [\[16\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py#L741-L749) [\[18\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py#L756-L764) common.py

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/common.py>

[\[19\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/modules.py#L250-L258) modules.py

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/services/aiva_common/enums/modules.py>

[\[20\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/cli_generated/schemas.ts#L6-L14) [\[21\]](https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/cli_generated/schemas.ts#L8-L16) schemas.ts

<https://github.com/kyle0527/AIVA/blob/2c1538d3ef2eaa14e7c151ad28fcc7a6cfbea882/cli_generated/schemas.ts>

[\[22\]](https://github.com/kyle0527/AIVA/blob/b7d658f7d788a5a30b0060e2143f84407cd614e9/services/features/forensic_tools.py#L7-L9) [\[23\]](https://github.com/kyle0527/AIVA/blob/b7d658f7d788a5a30b0060e2143f84407cd614e9/services/features/forensic_tools.py#L5-L9) forensic_tools.py

<https://github.com/kyle0527/AIVA/blob/b7d658f7d788a5a30b0060e2143f84407cd614e9/services/features/forensic_tools.py>
