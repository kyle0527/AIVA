from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Type

try:
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover
    BaseModel = object  # type: ignore

def _iter_models_from_module(mod) -> Dict[str, Type]:
    out = {}
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        try:
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
                out[obj.__name__] = obj
        except Exception:
            continue
    return out

def _discover_models() -> Dict[str, Type]:
    # 掃描 aiva_schemas_plugin 及其子模組（common/scan/function/integration/core/ai）
    root = importlib.import_module("aiva_schemas_plugin")
    models = _iter_models_from_module(root)

    for sub in ("common", "scan", "function", "integration", "core", "ai"):
        try:
            m = importlib.import_module(f"aiva_schemas_plugin.{sub}")
        except Exception:
            continue
        models.update(_iter_models_from_module(m))
    return dict(sorted(models.items(), key=lambda kv: kv[0].lower()))

def cmd_list_models(args: argparse.Namespace) -> None:
    models = _discover_models()
    for name in models:
        print(name)

def cmd_export_jsonschema(args: argparse.Namespace) -> None:
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = _discover_models()
    components: Dict[str, Any] = {}
    # 收集每個模型的 JSON Schema
    for name, model in models.items():
        try:
            schema = model.model_json_schema(ref_template="#/$defs/{model}")
        except Exception as e:
            print(f"[warn] schema generation failed for {name}: {e}")
            continue
        components[name] = schema

    # 合併到單一檔，放到 $defs
    bundle = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$defs": components}
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[i] JSON Schema written: {out_path}")

# --- JSON Schema -> TypeScript .d.ts 轉換（簡化版，涵蓋常見情形） ---

TS_BUILTIN = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
}

def _ts_type_from_schema(sch: Dict[str, Any], defs: Dict[str, Any]) -> str:
    if "$ref" in sch:
        ref = sch["$ref"]
        name = ref.split("/")[-1]
        return name

    if "anyOf" in sch:
        return " | ".join(_ts_type_from_schema(x, defs) for x in sch["anyOf"])
    if "oneOf" in sch:
        return " | ".join(_ts_type_from_schema(x, defs) for x in sch["oneOf"])
    if "allOf" in sch:
        # 簡易處理：交集以 & 表示
        return " & ".join(_ts_type_from_schema(x, defs) for x in sch["allOf"])

    t = sch.get("type")
    if isinstance(t, list):
        # union types
        return " | ".join(TS_BUILTIN.get(x, "any") for x in t)

    if t == "array":
        items = sch.get("items", {})
        return f"{_ts_type_from_schema(items, defs)}[]"
    if t == "object" or ("properties" in sch):
        props = sch.get("properties", {})
        required = set(sch.get("required", []))
        entries = []
        for k, v in props.items():
            ts_t = _ts_type_from_schema(v, defs)
            opt = "" if k in required else "?"
            entries.append(f"{k}{opt}: {ts_t};")
        return "{ " + " ".join(entries) + " }"

    if "enum" in sch:
        # 使用字面量 union
        lits = sch["enum"]
        lits_ts = []
        for lit in lits:
            if isinstance(lit, str):
                lits_ts.append(json.dumps(lit))
            elif lit is None:
                lits_ts.append("null")
            else:
                lits_ts.append(str(lit))
        return " | ".join(lits_ts)

    if t in TS_BUILTIN:
        return TS_BUILTIN[t]

    # formats
    if sch.get("format") in {"date-time", "date", "time"}:
        return "string"

    return "any"

def _emit_ts_interface(name: str, sch: Dict[str, Any], defs: Dict[str, Any]) -> str:
    # 嘗試從 $defs 的模型 schema 裡抓出 properties 轉介面；若非 object，則用 type alias。
    body = _ts_type_from_schema(sch, defs)
    if body.strip().startswith("{"):
        return f"export interface {name} {body}\n"
    else:
        return f"export type {name} = {body};\n"

def cmd_gen_ts(args: argparse.Namespace) -> None:
    json_path = Path(args.json).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = json.loads(json_path.read_text(encoding="utf-8"))
    defs = bundle.get("$defs", {})
    # 先宣告所有名稱，避免循環參考時順序問題（此簡版僅輸出順序，不做拓撲排序）
    lines = ["// AUTO-GENERATED from JSON Schema; do not edit manually.\n\n"]
    for name, sch in sorted(defs.items()):
        lines.append(_emit_ts_interface(name, sch, defs))

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[i] TypeScript d.ts written: {out_path}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aiva-contracts", description="AIVA contracts tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-models", help="列出 aiva_schemas_plugin 中可發現的模型")
    sp.set_defaults(func=cmd_list_models)

    sp = sub.add_parser("export-jsonschema", help="輸出合併 JSON Schema")
    sp.add_argument("--out", required=True, help="輸出檔路徑，例如 ./schemas/aiva_schemas.json")
    sp.set_defaults(func=cmd_export_jsonschema)

    sp = sub.add_parser("gen-ts", help="由 JSON Schema 產生 TypeScript .d.ts")
    sp.add_argument("--json", required=True, help="合併 JSON Schema 路徑")
    sp.add_argument("--out", required=True, help="輸出 .d.ts 路徑")
    sp.set_defaults(func=cmd_gen_ts)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
