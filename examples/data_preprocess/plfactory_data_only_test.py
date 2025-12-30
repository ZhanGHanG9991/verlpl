#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 JSON -> Parquet 转换脚本（按原始脚本的处理逻辑）：

需求变更点（相对原脚本）：
1) 遍历指定文件夹下所有 *.json；
2) 不区分 train/test；
3) 不做抽样/筛选：每个 JSON 文件内所有数据都转入 parquet；
4) parquet 命名与 json 文件名一致（仅扩展名改为 .parquet）；
5) 自动根据文件名判断 backend：文件名中一定包含 postgres 或 oracle；
6) 保持原脚本的样本构建逻辑：schema_cache、prompt 模板、build_example、datasets.to_parquet、可选 HDFS 同步等。

说明：
- 仍支持 JSON 顶层为 dict 分组（procedure/function/trigger/其它 key）或 list。
- 每个 JSON 文件独立产出一个 parquet（不会合并）。
"""

import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import datasets

import postgres_util
import oracle_util
from verl.utils.hdfs_io import copy, makedirs


PROMPT_TEMPLATE_POSTGRES = """You are an expert in postgres database and PL/SQL programming. Please generate a PL/pgSQL based on the provided database schema information and following the natural language instruction.

Schema Info:
{databases_schema_info}

Instruction:
{ir_text}

Please generate the PL/pgSQL code based on the schema and instruction."""


PROMPT_TEMPLATE_ORACLE = """You are an expert in oracle database and PL/SQL programming. Please generate an Oracle PL/SQL based on the provided database schema information and following the natural language instruction.

Schema Info:
{databases_schema_info}

Instruction:
{ir_text}

Please generate the Oracle PL/SQL code based on the schema and instruction."""


def format_prompt(backend: str, schema_info: str, ir_text: str) -> str:
    template = PROMPT_TEMPLATE_POSTGRES if backend == "postgres" else PROMPT_TEMPLATE_ORACLE
    return template.format(
        databases_schema_info=schema_info.strip(),
        ir_text=ir_text.strip(),
    )


def infer_backend_from_filename(filename: str) -> str:
    lower = filename.lower()
    if "postgres" in lower:
        return "postgres"
    if "oracle" in lower:
        return "oracle"
    raise ValueError(f"文件名不包含 postgres/oracle，无法判断 backend: {filename}")


def load_grouped_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    读取 JSON 文件，支持：
        - { "procedure": [...], "function": [...], ... }
        - [ ... ] （将整体视为 "__all__" key）
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON 文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError(f"{path} 列表元素必须为 JSON 对象。")
        return {"__all__": data}

    if isinstance(data, dict):
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for key, value in data.items():
            if not isinstance(value, list):
                raise ValueError(f"{path} 中 key={key} 的值必须为列表。")
            if not all(isinstance(item, dict) for item in value):
                raise ValueError(f"{path} 中 key={key} 的列表元素必须为 JSON 对象。")
            grouped[key] = value
        return grouped

    raise ValueError(f"{path} 内容须为列表或字典。")


def annotate_records_with_type(groups: Dict[str, List[Dict[str, Any]]], backend: str) -> None:
    for records in groups.values():
        for record in records:
            record["type"] = backend


def build_schema_prompt(
    schema_cache: Dict[str, Any],
    database_name: str,
    tables: List[str],
    backend: str,
) -> str:
    tables = tables or []
    util = postgres_util if backend == "postgres" else oracle_util
    fallback = (
        "Details: Schema information unavailable. Use best judgment based on IR."
        if backend == "postgres"
        else "Details: Oracle schema information unavailable. Use best judgment based on IR."
    )

    if schema_cache and database_name in schema_cache:
        try:
            table_schema_str = util.generate_schema_prompt_from_dict(
                schema_cache[database_name],
                tables if tables else None,
            )
            candidate_tables_str = ", ".join(sorted(tables))
            return f"Tables: {candidate_tables_str}\nDetails:\n{table_schema_str}"
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[Warning] {backend} schema prompt 生成失败: {database_name}: {exc}")

    tables_str = ", ".join(tables) if tables else "N/A"
    return f"Tables: {tables_str}\n{fallback}"


def strip_unused_extra_fields(extra_info: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("summary", "id", "natural_language"):
        extra_info.pop(key, None)
    return extra_info


def build_example(
    example: Dict[str, Any],
    idx: int,
    split: str,
    data_source: str,
    schema_cache: Dict[str, Any],
    backend: str,
) -> Dict[str, Any]:
    ir_text = example.get("ir", "")
    database_name = example.get("database_name", "")
    tables = example.get("tables") or []

    schema_prompt = build_schema_prompt(schema_cache, database_name, tables, backend)
    prompt_text = format_prompt(backend, schema_prompt, ir_text)

    extra_info = {
        "split": split,
        "index": idx,
        "ir": ir_text,
        "database_name": database_name,
        "tables": tables,
        "call_sqls": example.get("call_sqls"),
        "schema_prompt": schema_prompt,
        "type": backend,
    }
    extra_info = strip_unused_extra_fields(extra_info)

    return {
        "data_source": data_source,
        "type": backend,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "sql",
        "reward_model": {"style": "rule", "ground_truth": example.get("plsql", "")},
        "extra_info": extra_info,
    }


def convert_split(
    raw_records: Sequence[Dict[str, Any]],
    split_name: str,
    data_source: str,
    schema_cache: Dict[str, Any],
    backend: str,
) -> datasets.Dataset:
    hf_ds = datasets.Dataset.from_list(list(raw_records))

    def map_fn(example, idx):
        return build_example(
            example=example,
            idx=idx,
            split=split_name,
            data_source=data_source,
            schema_cache=schema_cache,
            backend=backend,
        )

    return hf_ds.map(function=map_fn, with_indices=True, desc=f"{backend}-{split_name}")


def list_json_files(input_dir: str) -> List[str]:
    input_dir = os.path.expanduser(input_dir)
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"input_dir 不是文件夹: {input_dir}")

    files: List[str] = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(".json"):
            files.append(os.path.join(input_dir, name))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="包含多个 JSON 文件的目录")
    parser.add_argument("--local_save_dir", required=True, help="本地输出 parquet 的目录")
    parser.add_argument("--hdfs_dir", default=None, help="可选：同步输出目录到 HDFS")
    parser.add_argument("--split_name", default="all", help="写入 extra_info.split 的名称（默认 all）")
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 与原脚本一致：初始化并加载 schema cache
    postgres_util.initialize_dataset_paths()
    pg_schema_cache = postgres_util.get_database_schema_json()

    oracle_util.initialize_dataset_paths()
    oc_schema_cache = oracle_util.get_database_schema_json()

    schema_caches = {"postgres": pg_schema_cache, "oracle": oc_schema_cache}

    json_paths = list_json_files(args.input_dir)
    if not json_paths:
        raise RuntimeError(f"[ERROR] {args.input_dir} 下未找到任何 .json 文件。")

    for path in json_paths:
        path = os.path.expanduser(path)
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)

        backend = infer_backend_from_filename(base)

        groups = load_grouped_json(path)
        annotate_records_with_type(groups, backend)

        # 不抽样：把所有 group 的 records 全部收集
        all_records: List[Dict[str, Any]] = []
        for records in groups.values():
            all_records.extend(records)

        if not all_records:
            print(f"[Warning] {base} 为空，跳过。")
            continue

        # 与原逻辑一致：convert_split -> Dataset.map(build_example)
        ds = convert_split(
            raw_records=all_records,
            split_name=args.split_name,
            data_source="plfactory",
            schema_cache=schema_caches[backend],
            backend=backend,
        )

        out_path = os.path.join(save_dir, f"{stem}.parquet")
        ds.to_parquet(out_path)
        print(f"[INFO] {base} ({backend}) -> {out_path} (共 {len(ds)} 条)")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=save_dir, dst=args.hdfs_dir)
        print(f"[INFO] 已同步至 HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    main()