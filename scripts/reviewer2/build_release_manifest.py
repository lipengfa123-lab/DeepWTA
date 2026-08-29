#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--release-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=None)
    args=ap.parse_args()
    root=args.release_root.resolve()
    out=(args.output or root/"SHA256_MANIFEST.csv").resolve()
    summary_path=(root/"RELEASE_SUMMARY.json").resolve()
    rows=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.resolve() not in {out, summary_path}:
            rows.append({"path":p.relative_to(root).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha(p)})
    pd.DataFrame(rows).to_csv(out,index=False)
    summary={
        "release_root":".","file_count":len(rows),
        "npz_input_count":sum(1 for r in rows if "/cases/" in "/"+r["path"] and r["path"].endswith(".npz")),
        "npz_solution_count":sum(1 for r in rows if "/solutions/" in "/"+r["path"] and r["path"].endswith(".npz")),
        "csv_count":sum(1 for r in rows if r["path"].endswith(".csv")),
        "sha256_manifest":out.relative_to(root).as_posix(),
    }
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
