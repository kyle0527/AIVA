# TypeScript Engine 依賴套件完整使用指南

本文檔整合自 node_modules/ 中 **439 個 Markdown 文件**的內容。

生成時間: 2025-11-27

## 📑 目錄

- [概述](#概述)
- [核心運行時依賴](#核心運行時依賴)
- [開發工具依賴](#開發工具依賴)
- [傳遞依賴套件](#傳遞依賴套件)
- [快速參考](#快速參考)
- [完整套件清單](#完整套件清單)

---

## 概述

TypeScript Engine 的依賴結構：

- **總套件數**: 229 個
- **總文檔數**: 439 個 Markdown 文件
- **總大小**: ~100 MB

### 套件分類

#### 核心運行時依賴 (4個)
- `playwright` - 瀏覽器自動化框架
- `amqplib` - RabbitMQ 客戶端
- `pino` - 高性能日誌記錄
- `pino-pretty` - 日誌美化輸出

#### 開發工具依賴 (6個)
- `typescript` - TypeScript 編譯器
- `@types/node` - Node.js 型別定義
- `eslint` - 程式碼檢查工具
- `prettier` - 程式碼格式化
- `tsx` - TypeScript 執行器
- `vitest` - 測試框架

#### 傳遞依賴 (219個)
由上述直接依賴自動引入的底層套件。

---

## 核心運行時依賴

這些套件是程式執行時必需的。

### playwright

**文檔數量**: 4 個

**簡介**: # 🎭 Playwright  ## 📑 目錄  - [[Documentation](https://playwright.dev) | [API reference](https://playwright.dev/docs/api/class-playwright)](#documentationhttpsplaywrightdev-api-referencehttpsplaywrightdevdocsapiclassplaywright) - [Installation](#installation)   - [Using init command](#using-init-command)   - [Manually](#manually) - [Capabilities](#capabilities)   - [Resilient • No flaky tests](#resilient-no-flaky-tests)   - [No trade-offs • No limits](#no-tradeoffs-no-limits)   - [Full isolation • ...

**相關文檔**:
- `README.md` (8975 bytes)
- `generator.md` (3721 bytes)
- `healer.md` (3746 bytes)
- `planner.md` (5099 bytes)

---

### amqplib

**文檔數量**: 4 個

**簡介**: # AMQP 0-9-1 library and client for Node.JS  ## 📑 目錄  - [❤️ Help Support Jack](#help-support-jack) - [RabbitMQ Compatibility](#rabbitmq-compatibility) - [Links](#links) - [Project status](#project-status) - [Callback API example](#callback-api-example) - [Promise/Async API example](#promiseasync-api-example) - [Running tests](#running-tests) - [Test coverage](#test-coverage)  ---   [![NPM version](https://img.shields.io/npm/v/amqplib.svg?style=flat-square)](https://www.npmjs.com/package/amqplib)...

**相關文檔**:
- `README.md` (5598 bytes)
- `README.md` (1864 bytes)
- `README.md` (3427 bytes)
- `bug_report.md` (1200 bytes)

---

### pino

**文檔數量**: 16 個

**簡介**: ![banner](pino-banner.png)  # pino  ## 📑 目錄  - [Documentation](#documentation) - [Install](#install) - [Usage](#usage) - [Essentials](#essentials)   - [Development Formatting](#development-formatting)   - [Transports & Log Processing](#transports-log-processing)   - [Low overhead](#low-overhead)   - [Bundling support](#bundling-support) - [The Team](#the-team)   - [Matteo Collina](#matteo-collina)   - [David Mark Clements](#david-mark-clements)   - [James Sumners](#james-sumners)   - [Thomas Wat...

**相關文檔**:
- `CONTRIBUTING.md` (1330 bytes)
- `README.md` (4870 bytes)
- `SECURITY.md` (2864 bytes)
- `api.md` (53098 bytes)
- `asynchronous.md` (1557 bytes)
- ... 還有 11 個文檔

---

### pino-pretty

**文檔數量**: 2 個

**簡介**: <a id="intro"></a> # pino-pretty  ## 📑 目錄  - [Example](#example) - [Install](#install) - [Usage](#usage)   - [CLI Arguments](#cli-arguments) - [Programmatic Integration](#programmatic-integration)   - [Usage as a stream](#usage-as-a-stream)   - [Usage with Jest](#usage-with-jest)   - [Handling non-serializable options](#handling-nonserializable-options)   - [Options](#options) - [Limitations](#limitations) - [License](#license)  ---   [![NPM Package Version](https://img.shields.io/npm/v/pino-pre...

**相關文檔**:
- `Readme.md` (12948 bytes)
- `help.md` (1240 bytes)

---

## 開發工具依賴

這些套件用於開發過程。

### typescript
**文檔數量**: 2 個

- `README.md` (2920 bytes)
- `SECURITY.md` (2831 bytes)

---

### @types/node
**文檔數量**: 1 個

- `README.md` (1482 bytes)

---

### eslint
**文檔數量**: 3 個

- `README.md` (19555 bytes)
- `README.md` (4278 bytes)
- `README.md` (8087 bytes)

---

### prettier
**文檔數量**: 2 個

- `README.md` (3516 bytes)
- `THIRD-PARTY-NOTICES.md` (276803 bytes)

---

### tsx
**文檔數量**: 1 個

- `README.md` (1373 bytes)

---

### vitest
**文檔數量**: 2 個

- `LICENSE.md` (82984 bytes)
- `README.md` (261 bytes)

---

## 傳遞依賴套件

由直接依賴自動引入的套件。以下列出文檔最多的前 20 個套件：

### 1. @typescript-eslint/eslint-plugin
**文檔數量**: 146 個
**文檔**: README.md, adjacent-overload-signatures.md, array-type.md, await-thenable.md, ban-ts-comment.md
 ... 還有 141 個

### 2. chai
**文檔數量**: 5 個
**文檔**: CODE_OF_CONDUCT.md, CONTRIBUTING.md, History.md, README.md, ReleaseNotes.md

### 3. vite
**文檔數量**: 5 個
**文檔**: LICENSE.md, README.md, LICENSE.md, README.md, README.md

### 4. events
**文檔數量**: 3 個
**文檔**: History.md, Readme.md, security.md

### 5. glob
**文檔數量**: 3 個
**文檔**: README.md, README.md, README.md

### 6. @vitest/runner
**文檔數量**: 3 個
**文檔**: README.md, readme.md, readme.md

### 7. @humanwhocodes/config-array
**文檔數量**: 3 個
**文檔**: README.md, README.md, README.md

### 8. @eslint/eslintrc
**文檔數量**: 3 個
**文檔**: README.md, README.md, README.md

### 9. ajv
**文檔數量**: 2 個
**文檔**: README.md, README.md

### 10. assertion-error
**文檔數量**: 2 個
**文檔**: History.md, README.md

### 11. balanced-match
**文檔數量**: 2 個
**文檔**: LICENSE.md, README.md

### 12. buffer
**文檔數量**: 2 個
**文檔**: AUTHORS.md, README.md

### 13. colorette
**文檔數量**: 2 個
**文檔**: LICENSE.md, README.md

### 14. esbuild
**文檔數量**: 2 個
**文檔**: LICENSE.md, README.md

### 15. fast-glob
**文檔數量**: 2 個
**文檔**: README.md, README.md

### 16. fast-levenshtein
**文檔數量**: 2 個
**文檔**: LICENSE.md, README.md

### 17. fastq
**文檔數量**: 2 個
**文檔**: README.md, SECURITY.md

### 18. mlly
**文檔數量**: 2 個
**文檔**: README.md, README.md

### 19. ms
**文檔數量**: 2 個
**文檔**: license.md, readme.md

### 20. npm-run-path
**文檔數量**: 2 個
**文檔**: readme.md, readme.md

**其他傳遞依賴**: 199 個套件

---

## 快速參考

### 安裝依賴
```bash
cd services/scan/engines/typescript_engine
npm install
```

### 核心套件用途

| 套件 | 用途 | 必要性 |
|------|------|--------|
| `playwright` | 瀏覽器自動化 | ✅ 絕對必要 |
| `amqplib` | RabbitMQ 客戶端 | ✅ 架構必需 |
| `pino` | 日誌記錄 | ⚠️ 建議保留 |
| `pino-pretty` | 日誌美化 | ❌ 僅開發用 |
| `typescript` | TS 編譯器 | ✅ 絕對必要 |
| `@types/node` | Node.js 型別 | ✅ 強烈建議 |
| `eslint` | 程式碼檢查 | ❌ 可選 |
| `prettier` | 程式碼格式化 | ❌ 可選 |
| `tsx` | TS 執行器 | ⚠️ 開發便利 |
| `vitest` | 測試框架 | ⚠️ 建議保留 |

### 常用命令
```bash
npm run dev      # 開發模式
npm run build    # 編譯
npm start        # 執行
npm run lint     # 檢查
npm run format   # 格式化
npm test         # 測試
```

---

## 完整套件清單

共 229 個套件，439 個文檔：

### @esbuild/win32-x64
文檔數: 1 個
- `README.md` (143 bytes)

### @eslint-community/eslint-utils
文檔數: 1 個
- `README.md` (1832 bytes)

### @eslint-community/regexpp
文檔數: 1 個
- `README.md` (6971 bytes)

### @eslint/eslintrc
文檔數: 3 個
- `README.md` (3128 bytes)
- `README.md` (4278 bytes)
- `README.md` (8087 bytes)

### @eslint/js
文檔數: 1 個
- `README.md` (1567 bytes)

### @humanwhocodes/config-array
文檔數: 3 個
- `README.md` (14899 bytes)
- `README.md` (4278 bytes)
- `README.md` (8087 bytes)

### @humanwhocodes/module-importer
文檔數: 1 個
- `README.md` (2269 bytes)

### @humanwhocodes/object-schema
文檔數: 1 個
- `README.md` (5739 bytes)

### @jest/schemas
文檔數: 1 個
- `README.md` (129 bytes)

### @jridgewell/sourcemap-codec
文檔數: 1 個
- `README.md` (10099 bytes)

### @nodelib/fs.scandir
文檔數: 1 個
- `README.md` (5720 bytes)

### @nodelib/fs.stat
文檔數: 1 個
- `README.md` (3700 bytes)

### @nodelib/fs.walk
文檔數: 1 個
- `README.md` (7030 bytes)

### @rollup/rollup-win32-x64-gnu
文檔數: 1 個
- `README.md` (92 bytes)

### @rollup/rollup-win32-x64-msvc
文檔數: 1 個
- `README.md` (94 bytes)

### @sinclair/typebox
文檔數: 1 個
- `readme.md` (82792 bytes)

### @types/amqplib
文檔數: 1 個
- `README.md` (745 bytes)

### @types/estree
文檔數: 1 個
- `README.md` (443 bytes)

### @types/json-schema
文檔數: 1 個
- `README.md` (607 bytes)

### @types/node
文檔數: 1 個
- `README.md` (1482 bytes)

### @types/semver
文檔數: 1 個
- `README.md` (701 bytes)

### @typescript-eslint/eslint-plugin
文檔數: 146 個
- `README.md` (751 bytes)
- `adjacent-overload-signatures.md` (2802 bytes)
- `array-type.md` (5406 bytes)
- `await-thenable.md` (1590 bytes)
- `ban-ts-comment.md` (4914 bytes)
- `ban-tslint-comment.md` (1218 bytes)
- `ban-types.md` (4665 bytes)
- `block-spacing.md` (472 bytes)
- `brace-style.md` (397 bytes)
- `camelcase.md` (429 bytes)
- `class-literal-property-style.md` (3317 bytes)
- `class-methods-use-this.md` (3075 bytes)
- `comma-dangle.md` (830 bytes)
- `comma-spacing.md` (395 bytes)
- `consistent-generic-constructors.md` (2549 bytes)
- `consistent-indexed-object-style.md` (1973 bytes)
- `consistent-type-assertions.md` (4057 bytes)
- `consistent-type-definitions.md` (2242 bytes)
- `consistent-type-exports.md` (2818 bytes)
- `consistent-type-imports.md` (5096 bytes)
- `default-param-last.md` (1261 bytes)
- `dot-notation.md` (2971 bytes)
- `explicit-function-return-type.md` (8170 bytes)
- `explicit-member-accessibility.md` (10659 bytes)
- `explicit-module-boundary-types.md` (6726 bytes)
- `func-call-spacing.md` (435 bytes)
- `indent.md` (508 bytes)
- `init-declarations.md` (407 bytes)
- `key-spacing.md` (460 bytes)
- `keyword-spacing.md` (403 bytes)
- `lines-around-comment.md` (1351 bytes)
- `lines-between-class-members.md` (1779 bytes)
- `max-params.md` (435 bytes)
- `member-delimiter-style.md` (4207 bytes)
- `member-ordering.md` (36975 bytes)
- `method-signature-style.md` (3070 bytes)
- `naming-convention.md` (29743 bytes)
- `no-array-constructor.md` (718 bytes)
- `no-array-delete.md` (1246 bytes)
- `no-base-to-string.md` (2800 bytes)
- `no-confusing-non-null-assertion.md` (1673 bytes)
- `no-confusing-void-expression.md` (4181 bytes)
- `no-dupe-class-members.md` (496 bytes)
- `no-duplicate-enum-values.md` (1643 bytes)
- `no-duplicate-imports.md` (508 bytes)
- `no-duplicate-type-constituents.md` (2277 bytes)
- `no-dynamic-delete.md` (1903 bytes)
- `no-empty-function.md` (3499 bytes)
- `no-empty-interface.md` (1632 bytes)
- `no-explicit-any.md` (4786 bytes)
- `no-extra-non-null-assertion.md` (1110 bytes)
- `no-extra-parens.md` (368 bytes)
- `no-extra-semi.md` (1010 bytes)
- `no-extraneous-class.md` (7724 bytes)
- `no-floating-promises.md` (4312 bytes)
- `no-for-in-array.md` (2401 bytes)
- `no-implied-eval.md` (2955 bytes)
- `no-import-type-side-effects.md` (2780 bytes)
- `no-inferrable-types.md` (2789 bytes)
- `no-invalid-this.md` (508 bytes)
- `no-invalid-void-type.md` (3794 bytes)
- `no-loop-func.md` (401 bytes)
- `no-loss-of-precision.md` (441 bytes)
- `no-magic-numbers.md` (3807 bytes)
- `no-meaningless-void-operator.md` (1920 bytes)
- `no-misused-new.md` (1504 bytes)
- `no-misused-promises.md` (6677 bytes)
- `no-mixed-enums.md` (2126 bytes)
- `no-namespace.md` (3357 bytes)
- `no-non-null-asserted-nullish-coalescing.md` (1846 bytes)
- `no-non-null-asserted-optional-chain.md` (1622 bytes)
- `no-non-null-assertion.md` (1536 bytes)
- `no-parameter-properties.md` (411 bytes)
- `no-redeclare.md` (2119 bytes)
- `no-redundant-type-constituents.md` (3227 bytes)
- `no-require-imports.md` (2058 bytes)
- `no-restricted-imports.md` (2288 bytes)
- `no-shadow.md` (4168 bytes)
- `no-this-alias.md` (2727 bytes)
- `no-throw-literal.md` (2372 bytes)
- `no-type-alias.md` (16691 bytes)
- `no-unnecessary-boolean-literal-compare.md` (5471 bytes)
- `no-unnecessary-condition.md` (4808 bytes)
- `no-unnecessary-qualifier.md` (1142 bytes)
- `no-unnecessary-type-arguments.md` (1563 bytes)
- `no-unnecessary-type-assertion.md` (1747 bytes)
- `no-unnecessary-type-constraint.md` (1403 bytes)
- `no-unsafe-argument.md` (3328 bytes)
- `no-unsafe-assignment.md` (3164 bytes)
- `no-unsafe-call.md` (2052 bytes)
- `no-unsafe-declaration-merging.md` (1637 bytes)
- `no-unsafe-enum-comparison.md` (2293 bytes)
- `no-unsafe-member-access.md` (2208 bytes)
- `no-unsafe-return.md` (3321 bytes)
- `no-unsafe-unary-minus.md` (900 bytes)
- `no-unused-expressions.md` (426 bytes)
- `no-unused-vars.md` (1750 bytes)
- `no-use-before-define.md` (2278 bytes)
- `no-useless-constructor.md` (842 bytes)
- `no-useless-empty-export.md` (1558 bytes)
- `no-useless-template-literals.md` (1411 bytes)
- `no-var-requires.md` (2126 bytes)
- `non-nullable-type-assertion-style.md` (1308 bytes)
- `object-curly-spacing.md` (390 bytes)
- `padding-line-between-statements.md` (1069 bytes)
- `parameter-properties.md` (9961 bytes)
- `prefer-as-const.md` (1524 bytes)
- `prefer-destructuring.md` (2387 bytes)
- `prefer-enum-initializers.md` (1319 bytes)
- `prefer-find.md` (1602 bytes)
- `prefer-for-of.md` (1192 bytes)
- `prefer-function-type.md` (2313 bytes)
- `prefer-includes.md` (2201 bytes)
- `prefer-literal-enum-member.md` (2877 bytes)
- `prefer-namespace-keyword.md` (1631 bytes)
- `prefer-nullish-coalescing.md` (8054 bytes)
- `prefer-optional-chain.md` (9294 bytes)
- `prefer-promise-reject-errors.md` (1197 bytes)
- `prefer-readonly-parameter-types.md` (11773 bytes)
- `prefer-readonly.md` (2751 bytes)
- `prefer-reduce-type-parameter.md` (2373 bytes)
- `prefer-regexp-exec.md` (1342 bytes)
- `prefer-return-this-type.md` (2100 bytes)
- `prefer-string-starts-ends-with.md` (1699 bytes)
- `prefer-ts-expect-error.md` (2370 bytes)
- `promise-function-async.md` (4498 bytes)
- `quotes.md` (430 bytes)
- `README.md` (2986 bytes)
- `require-array-sort-compare.md` (2259 bytes)
- `require-await.md` (611 bytes)
- `restrict-plus-operands.md` (5657 bytes)
- `restrict-template-expressions.md` (4725 bytes)
- `return-await.md` (4792 bytes)
- `semi.md` (536 bytes)
- `sort-type-constituents.md` (3983 bytes)
- `space-before-blocks.md` (895 bytes)
- `space-before-function-paren.md` (441 bytes)
- `space-infix-ops.md` (404 bytes)
- `strict-boolean-expressions.md` (8436 bytes)
- `switch-exhaustiveness-check.md` (6282 bytes)
- `TEMPLATE.md` (710 bytes)
- `triple-slash-reference.md` (3265 bytes)
- `type-annotation-spacing.md` (5926 bytes)
- `typedef.md` (8159 bytes)
- `unbound-method.md` (3521 bytes)
- `unified-signatures.md` (2291 bytes)

### @typescript-eslint/parser
文檔數: 1 個
- `README.md` (856 bytes)

### @typescript-eslint/scope-manager
文檔數: 1 個
- `README.md` (693 bytes)

### @typescript-eslint/type-utils
文檔數: 1 個
- `README.md` (864 bytes)

### @typescript-eslint/types
文檔數: 1 個
- `README.md` (427 bytes)

### @typescript-eslint/typescript-estree
文檔數: 1 個
- `README.md` (786 bytes)

### @typescript-eslint/utils
文檔數: 1 個
- `README.md` (697 bytes)

### @typescript-eslint/visitor-keys
文檔數: 1 個
- `README.md` (361 bytes)

### @ungap/structured-clone
文檔數: 1 個
- `README.md` (4702 bytes)

### @vitest/expect
文檔數: 1 個
- `README.md` (445 bytes)

### @vitest/runner
文檔數: 3 個
- `README.md` (163 bytes)
- `readme.md` (2819 bytes)
- `readme.md` (2857 bytes)

### @vitest/snapshot
文檔數: 1 個
- `README.md` (2558 bytes)

### @vitest/spy
文檔數: 1 個
- `README.md` (63 bytes)

### abort-controller
文檔數: 1 個
- `README.md` (3587 bytes)

### acorn
文檔數: 1 個
- `README.md` (11027 bytes)

### acorn-jsx
文檔數: 1 個
- `README.md` (2013 bytes)

### acorn-walk
文檔數: 1 個
- `README.md` (4567 bytes)

### ajv
文檔數: 2 個
- `README.md` (90456 bytes)
- `README.md` (149 bytes)

### amqplib
文檔數: 4 個
- `README.md` (5598 bytes)
- `README.md` (1864 bytes)
- `README.md` (3427 bytes)
- `bug_report.md` (1200 bytes)

### ansi-regex
文檔數: 1 個
- `readme.md` (2899 bytes)

### ansi-styles
文檔數: 1 個
- `readme.md` (4846 bytes)

### argparse
文檔數: 1 個
- `README.md` (2768 bytes)

### array-union
文檔數: 1 個
- `readme.md` (727 bytes)

### assertion-error
文檔數: 2 個
- `History.md` (527 bytes)
- `README.md` (1788 bytes)

### atomic-sleep
文檔數: 1 個
- `readme.md` (1823 bytes)

### balanced-match
文檔數: 2 個
- `LICENSE.md` (1096 bytes)
- `README.md` (3799 bytes)

### base64-js
文檔數: 1 個
- `README.md` (1234 bytes)

### brace-expansion
文檔數: 1 個
- `README.md` (4536 bytes)

### braces
文檔數: 1 個
- `README.md` (22673 bytes)

### buffer
文檔數: 2 個
- `AUTHORS.md` (2924 bytes)
- `README.md` (18238 bytes)

### buffer-more-ints
文檔數: 1 個
- `README.md` (2627 bytes)

### cac
文檔數: 1 個
- `README.md` (17647 bytes)

### callsites
文檔數: 1 個
- `readme.md` (1978 bytes)

### chai
文檔數: 5 個
- `CODE_OF_CONDUCT.md` (2863 bytes)
- `CONTRIBUTING.md` (10011 bytes)
- `History.md` (37589 bytes)
- `README.md` (8484 bytes)
- `ReleaseNotes.md` (33544 bytes)

### chalk
文檔數: 1 個
- `readme.md` (14122 bytes)

### check-error
文檔數: 1 個
- `README.md` (6441 bytes)

### color-convert
文檔數: 1 個
- `README.md` (2913 bytes)

### color-name
文檔數: 1 個
- `README.md` (373 bytes)

### colorette
文檔數: 2 個
- `LICENSE.md` (1079 bytes)
- `README.md` (4681 bytes)

### confbox
文檔數: 1 個
- `README.md` (5097 bytes)

### cross-spawn
文檔數: 1 個
- `README.md` (4470 bytes)

### dateformat
文檔數: 1 個
- `Readme.md` (11804 bytes)

### debug
文檔數: 1 個
- `README.md` (23018 bytes)

### deep-eql
文檔數: 1 個
- `README.md` (4362 bytes)

### diff-sequences
文檔數: 1 個
- `README.md` (16284 bytes)

### dir-glob
文檔數: 1 個
- `readme.md` (1670 bytes)

### doctrine
文檔數: 1 個
- `README.md` (7088 bytes)

### end-of-stream
文檔數: 1 個
- `README.md` (1779 bytes)

### esbuild
文檔數: 2 個
- `LICENSE.md` (1069 bytes)
- `README.md` (175 bytes)

### escape-string-regexp
文檔數: 1 個
- `readme.md` (1109 bytes)

### eslint
文檔數: 3 個
- `README.md` (19555 bytes)
- `README.md` (4278 bytes)
- `README.md` (8087 bytes)

### eslint-scope
文檔數: 1 個
- `README.md` (2170 bytes)

### eslint-visitor-keys
文檔數: 1 個
- `README.md` (3060 bytes)

### espree
文檔數: 1 個
- `README.md` (11258 bytes)

### esquery
文檔數: 1 個
- `README.md` (2187 bytes)

### esrecurse
文檔數: 1 個
- `README.md` (5425 bytes)

### estraverse
文檔數: 1 個
- `README.md` (5215 bytes)

### estree-walker
文檔數: 1 個
- `README.md` (1732 bytes)

### esutils
文檔數: 1 個
- `README.md` (8616 bytes)

### event-target-shim
文檔數: 1 個
- `README.md` (9784 bytes)

### events
文檔數: 3 個
- `History.md` (3550 bytes)
- `Readme.md` (2438 bytes)
- `security.md` (412 bytes)

### execa
文檔數: 1 個
- `readme.md` (28422 bytes)

### fast-copy
文檔數: 1 個
- `README.md` (15235 bytes)

### fast-deep-equal
文檔數: 1 個
- `README.md` (3547 bytes)

### fast-glob
文檔數: 2 個
- `README.md` (28178 bytes)
- `README.md` (4945 bytes)

### fast-json-stable-stringify
文檔數: 1 個
- `README.md` (3736 bytes)

### fast-levenshtein
文檔數: 2 個
- `LICENSE.md` (1100 bytes)
- `README.md` (3670 bytes)

### fast-redact
文檔數: 1 個
- `readme.md` (12059 bytes)

### fast-safe-stringify
文檔數: 1 個
- `readme.md` (6497 bytes)

### fastq
文檔數: 2 個
- `README.md` (9649 bytes)
- `SECURITY.md` (573 bytes)

### file-entry-cache
文檔數: 1 個
- `README.md` (5543 bytes)

### fill-range
文檔數: 1 個
- `README.md` (7892 bytes)

### find-up
文檔數: 1 個
- `readme.md` (4732 bytes)

### flat-cache
文檔數: 1 個
- `README.md` (3256 bytes)

### flatted
文檔數: 1 個
- `README.md` (4885 bytes)

### fs.realpath
文檔數: 1 個
- `README.md` (881 bytes)

### get-func-name
文檔數: 1 個
- `README.md` (3377 bytes)

### get-stream
文檔數: 1 個
- `readme.md` (11842 bytes)

### get-tsconfig
文檔數: 1 個
- `README.md` (7574 bytes)

### glob
文檔數: 3 個
- `README.md` (16098 bytes)
- `README.md` (4278 bytes)
- `README.md` (8087 bytes)

### glob-parent
文檔數: 1 個
- `README.md` (4497 bytes)

### globals
文檔數: 1 個
- `readme.md` (1716 bytes)

### globby
文檔數: 1 個
- `readme.md` (6231 bytes)

### graphemer
文檔數: 1 個
- `README.md` (5819 bytes)

### has-flag
文檔數: 1 個
- `readme.md` (1802 bytes)

### help-me
文檔數: 1 個
- `README.md` (1117 bytes)

### human-signals
文檔數: 1 個
- `README.md` (5490 bytes)

### ieee754
文檔數: 1 個
- `README.md` (1938 bytes)

### ignore
文檔數: 1 個
- `README.md` (13439 bytes)

### import-fresh
文檔數: 1 個
- `readme.md` (1237 bytes)

### imurmurhash
文檔數: 1 個
- `README.md` (5070 bytes)

### inflight
文檔數: 1 個
- `README.md` (991 bytes)

### inherits
文檔數: 1 個
- `README.md` (1714 bytes)

### is-extglob
文檔數: 1 個
- `README.md` (3756 bytes)

### is-glob
文檔數: 1 個
- `README.md` (7412 bytes)

### is-number
文檔數: 1 個
- `README.md` (6903 bytes)

### is-path-inside
文檔數: 1 個
- `readme.md` (1583 bytes)

### is-stream
文檔數: 1 個
- `readme.md` (1976 bytes)

### isexe
文檔數: 1 個
- `README.md` (1599 bytes)

### joycon
文檔數: 1 個
- `README.md` (4239 bytes)

### js-tokens
文檔數: 1 個
- `README.md` (436 bytes)

### js-yaml
文檔數: 1 個
- `README.md` (8807 bytes)

### json-buffer
文檔數: 1 個
- `README.md` (659 bytes)

### json-schema-traverse
文檔數: 1 個
- `README.md` (2860 bytes)

### keyv
文檔數: 1 個
- `README.md` (17185 bytes)

### levn
文檔數: 1 個
- `README.md` (11020 bytes)

### local-pkg
文檔數: 1 個
- `README.md` (1222 bytes)

### locate-path
文檔數: 1 個
- `readme.md` (2772 bytes)

### lodash.merge
文檔數: 1 個
- `README.md` (446 bytes)

### loupe
文檔數: 1 個
- `README.md` (2358 bytes)

### magic-string
文檔數: 1 個
- `README.md` (14517 bytes)

### merge-stream
文檔數: 1 個
- `README.md` (2059 bytes)

### merge2
文檔數: 1 個
- `README.md` (4476 bytes)

### micromatch
文檔數: 1 個
- `README.md` (41258 bytes)

### mimic-fn
文檔數: 1 個
- `readme.md` (2369 bytes)

### minimatch
文檔數: 1 個
- `README.md` (18321 bytes)

### minimist
文檔數: 1 個
- `README.md` (3609 bytes)

### mlly
文檔數: 2 個
- `README.md` (14928 bytes)
- `README.md` (3149 bytes)

### ms
文檔數: 2 個
- `license.md` (1079 bytes)
- `readme.md` (2138 bytes)

### nanoid
文檔數: 1 個
- `README.md` (1524 bytes)

### natural-compare
文檔數: 1 個
- `README.md` (3385 bytes)

### npm-run-path
文檔數: 2 個
- `readme.md` (3074 bytes)
- `readme.md` (1390 bytes)

### on-exit-leak-free
文檔數: 1 個
- `README.md` (1366 bytes)

### once
文檔數: 1 個
- `README.md` (1847 bytes)

### onetime
文檔數: 1 個
- `readme.md` (2127 bytes)

### optionator
文檔數: 1 個
- `README.md` (16027 bytes)

### p-limit
文檔數: 1 個
- `readme.md` (3340 bytes)

### p-locate
文檔數: 1 個
- `readme.md` (2880 bytes)

### parent-module
文檔數: 1 個
- `readme.md` (1645 bytes)

### path-exists
文檔數: 1 個
- `readme.md` (1629 bytes)

### path-is-absolute
文檔數: 1 個
- `readme.md` (1410 bytes)

### path-key
文檔數: 1 個
- `readme.md` (1532 bytes)

### path-type
文檔數: 1 個
- `readme.md` (1471 bytes)

### pathe
文檔數: 1 個
- `README.md` (2868 bytes)

### pathval
文檔數: 1 個
- `README.md` (4447 bytes)

### picocolors
文檔數: 1 個
- `README.md` (622 bytes)

### picomatch
文檔數: 1 個
- `README.md` (29024 bytes)

### pino
文檔數: 16 個
- `CONTRIBUTING.md` (1330 bytes)
- `README.md` (4870 bytes)
- `SECURITY.md` (2864 bytes)
- `api.md` (53098 bytes)
- `asynchronous.md` (1557 bytes)
- `benchmarks.md` (1136 bytes)
- `browser.md` (7480 bytes)
- `bundling.md` (1894 bytes)
- `ecosystem.md` (5182 bytes)
- `help.md` (11415 bytes)
- `lts.md` (3013 bytes)
- `pretty.md` (939 bytes)
- `redaction.md` (4369 bytes)
- `transports.md` (35870 bytes)
- `web.md` (5413 bytes)
- `sidebar.md` (1214 bytes)

### pino-abstract-transport
文檔數: 1 個
- `README.md` (5186 bytes)

### pino-pretty
文檔數: 2 個
- `Readme.md` (12948 bytes)
- `help.md` (1240 bytes)

### pino-std-serializers
文檔數: 1 個
- `Readme.md` (6902 bytes)

### pkg-types
文檔數: 2 個
- `README.md` (4163 bytes)
- `README.md` (3149 bytes)

### playwright
文檔數: 4 個
- `README.md` (8975 bytes)
- `generator.md` (3721 bytes)
- `healer.md` (3746 bytes)
- `planner.md` (5099 bytes)

### playwright-core
文檔數: 1 個
- `README.md` (120 bytes)

### postcss
文檔數: 1 個
- `README.md` (1179 bytes)

### prelude-ls
文檔數: 1 個
- `README.md` (613 bytes)

### prettier
文檔數: 2 個
- `README.md` (3516 bytes)
- `THIRD-PARTY-NOTICES.md` (276803 bytes)

### pretty-format
文檔數: 2 個
- `README.md` (14717 bytes)
- `readme.md` (4520 bytes)

### process
文檔數: 1 個
- `README.md` (1477 bytes)

### process-warning
文檔數: 1 個
- `README.md` (4629 bytes)

### pump
文檔數: 2 個
- `README.md` (2624 bytes)
- `SECURITY.md` (193 bytes)

### punycode
文檔數: 1 個
- `README.md` (6272 bytes)

### querystringify
文檔數: 1 個
- `README.md` (2551 bytes)

### queue-microtask
文檔數: 1 個
- `README.md` (6629 bytes)

### quick-format-unescaped
文檔數: 1 個
- `readme.md` (2060 bytes)

### react-is
文檔數: 1 個
- `README.md` (2727 bytes)

### readable-stream
文檔數: 1 個
- `README.md` (5606 bytes)

### real-require
文檔數: 2 個
- `LICENSE.md` (1114 bytes)
- `README.md` (1878 bytes)

### requires-port
文檔數: 1 個
- `README.md` (1880 bytes)

### resolve-from
文檔數: 1 個
- `readme.md` (2166 bytes)

### resolve-pkg-maps
文檔數: 1 個
- `README.md` (8827 bytes)

### reusify
文檔數: 2 個
- `README.md` (3392 bytes)
- `SECURITY.md` (573 bytes)

### rimraf
文檔數: 1 個
- `README.md` (3724 bytes)

### rollup
文檔數: 2 個
- `LICENSE.md` (36115 bytes)
- `README.md` (10426 bytes)

### run-parallel
文檔數: 1 個
- `README.md` (3472 bytes)

### safe-buffer
文檔數: 1 個
- `README.md` (21251 bytes)

### safe-stable-stringify
文檔數: 1 個
- `readme.md` (6958 bytes)

### secure-json-parse
文檔數: 2 個
- `LICENSE.md` (1720 bytes)
- `README.md` (4906 bytes)

### semver
文檔數: 1 個
- `README.md` (25719 bytes)

### shebang-command
文檔數: 1 個
- `readme.md` (641 bytes)

### shebang-regex
文檔數: 1 個
- `readme.md` (726 bytes)

### siginfo
文檔數: 1 個
- `README.md` (1319 bytes)

### signal-exit
文檔數: 1 個
- `README.md` (2576 bytes)

### slash
文檔數: 1 個
- `readme.md` (1031 bytes)

### sonic-boom
文檔數: 1 個
- `README.md` (6055 bytes)

### source-map-js
文檔數: 1 個
- `README.md` (29453 bytes)

### split2
文檔數: 1 個
- `README.md` (3381 bytes)

### stackback
文檔數: 1 個
- `README.md` (1990 bytes)

### std-env
文檔數: 1 個
- `README.md` (3004 bytes)

### string_decoder
文檔數: 1 個
- `README.md` (1929 bytes)

### strip-ansi
文檔數: 1 個
- `readme.md` (1764 bytes)

### strip-final-newline
文檔數: 1 個
- `readme.md` (1197 bytes)

### strip-json-comments
文檔數: 1 個
- `readme.md` (2251 bytes)

### strip-literal
文檔數: 1 個
- `README.md` (916 bytes)

### supports-color
文檔數: 1 個
- `readme.md` (2432 bytes)

### thread-stream
文檔數: 1 個
- `README.md` (3501 bytes)

### tinybench
文檔數: 1 個
- `README.md` (13049 bytes)

### tinypool
文檔數: 1 個
- `README.md` (1347 bytes)

### tinyspy
文檔數: 1 個
- `README.md` (492 bytes)

### to-regex-range
文檔數: 1 個
- `README.md` (14162 bytes)

### ts-api-utils
文檔數: 2 個
- `LICENSE.md` (1038 bytes)
- `README.md` (7441 bytes)

### tsx
文檔數: 1 個
- `README.md` (1373 bytes)

### type-check
文檔數: 1 個
- `README.md` (10930 bytes)

### type-detect
文檔數: 1 個
- `README.md` (8041 bytes)

### type-fest
文檔數: 1 個
- `readme.md` (30958 bytes)

### typescript
文檔數: 2 個
- `README.md` (2920 bytes)
- `SECURITY.md` (2831 bytes)

### ufo
文檔數: 1 個
- `README.md` (13587 bytes)

### undici-types
文檔數: 1 個
- `README.md` (455 bytes)

### uri-js
文檔數: 1 個
- `README.md` (7147 bytes)

### url-parse
文檔數: 1 個
- `README.md` (6459 bytes)

### vite
文檔數: 5 個
- `LICENSE.md` (166194 bytes)
- `README.md` (1129 bytes)
- `LICENSE.md` (1069 bytes)
- `README.md` (175 bytes)
- `README.md` (143 bytes)

### vite-node
文檔數: 1 個
- `README.md` (5586 bytes)

### vitest
文檔數: 2 個
- `LICENSE.md` (82984 bytes)
- `README.md` (261 bytes)

### which
文檔數: 1 個
- `README.md` (1434 bytes)

### why-is-node-running
文檔數: 1 個
- `README.md` (2680 bytes)

### word-wrap
文檔數: 1 個
- `README.md` (6759 bytes)

### wrappy
文檔數: 1 個
- `README.md` (685 bytes)

### yocto-queue
文檔數: 1 個
- `readme.md` (2328 bytes)

---

## 📝 備註

- 本文檔整合自 **439 個** Markdown 文件
- 涵蓋 **229 個**套件的完整文檔
- node_modules/ 已在 .gitignore 中，不會提交
- 執行 `npm install` 可完全重建

## 🔗 相關資源

- [Playwright](https://playwright.dev/)
- [RabbitMQ](https://www.rabbitmq.com/)
- [Pino](https://getpino.io/)
- [TypeScript](https://www.typescriptlang.org/)
- [NPM](https://www.npmjs.com/)
