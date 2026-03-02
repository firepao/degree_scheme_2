from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="打包复现与答辩材料目录")
    parser.add_argument("--repro-dir", type=str, default=None, help="指定 repro_* 目录；不传则自动选择最新")
    parser.add_argument("--defense-dir", type=str, default=None, help="指定 defense_* 目录；不传则自动选择最新")
    parser.add_argument("--artifacts-root", type=str, default="artifacts", help="repro 目录根路径")
    parser.add_argument("--deliverables-root", type=str, default="deliverables", help="defense 目录根路径和 zip 输出根路径")
    return parser.parse_args()


def latest_dir(root: Path, pattern: str) -> Path:
    candidates = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"未找到目录: {root}/{pattern}")
    return candidates[-1]


def make_zip(src_dir: Path, out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    zip_path = shutil.make_archive(str(out_base), "zip", root_dir=str(src_dir))
    return Path(zip_path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    artifacts_root = (repo_root / args.artifacts_root).resolve()
    deliverables_root = (repo_root / args.deliverables_root).resolve()
    deliverables_root.mkdir(parents=True, exist_ok=True)

    repro_dir = (repo_root / args.repro_dir).resolve() if args.repro_dir else latest_dir(artifacts_root, "repro_*")
    defense_dir = (repo_root / args.defense_dir).resolve() if args.defense_dir else latest_dir(deliverables_root, "defense_*")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    repro_zip_base = deliverables_root / f"repro_package_{ts}"
    defense_zip_base = deliverables_root / f"defense_package_{ts}"

    repro_zip = make_zip(repro_dir, repro_zip_base)
    defense_zip = make_zip(defense_dir, defense_zip_base)

    manifest = {
        "timestamp": ts,
        "repro_dir": str(repro_dir),
        "defense_dir": str(defense_dir),
        "repro_zip": str(repro_zip),
        "defense_zip": str(defense_zip),
    }

    manifest_path = deliverables_root / f"package_manifest_{ts}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("打包完成：")
    print(f"- Repro zip: {repro_zip}")
    print(f"- Defense zip: {defense_zip}")
    print(f"- Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
