#!/usr/bin/env python3
from __future__ import annotations

import argparse
import enum
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="輸出 enums.ts 路徑")
    args = parser.parse_args()

    mod = import_module("aiva_common.enums")
    ts_lines: List[str] = ["// AUTO-GENERATED from aiva_common.enums; do not edit.\n\n"]

    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, enum.Enum):
            # 決定 TS enum 的值型別（字串或數字）
            member_lines: List[str] = []
            all_strings = True
            for m in obj:
                val = m.value
                if not isinstance(val, str):
                    all_strings = False
            for m in obj:
                key = m.name
                val = m.value
                if isinstance(val, str):
                    member_lines.append(f"  {key} = {val!r},")
                else:
                    member_lines.append(f"  {key} = {val},")

            ts_lines.append(f"export enum {name} {{")
            ts_lines.extend(member_lines)
            ts_lines.append("}\n")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(ts_lines), encoding="utf-8")
    print(f"[i] TS enums written: {out}")

if __name__ == "__main__":
    main()
