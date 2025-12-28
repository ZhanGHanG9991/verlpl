#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLFactory 数据预处理脚本（新版）：

输入：
    - 2 个 Postgres JSON 文件
    - 2 个 Oracle JSON 文件
    均为 { "procedure": [...], "function": [...], "trigger": [...] } 结构
    （如存在额外 key，同样会纳入比例计算）

处理流程：
    1. 为每条样本添加 "type" 字段（postgres/oracle）。
    2. 每个文件独立按 key 比例随机抽取：
        - 750 条训练样本
        - 125 条测试样本（与训练样本无交集）
    3. 合并 4 个文件的抽样结果：
        - 共 3000 条训练样本 -> shuffle -> parquet
        - 共 500 条测试样本  -> shuffle -> parquet
    4. Postgres 样本使用 PROMPT_TEMPLATE（PL/pgSQL），
       Oracle 样本使用 PROMPT_TEMPLATE_ORACLE（PL/SQL）。
"""

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Sequence

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


def allocate_counts(counts: Dict[str, int], sample_size: int) -> Dict[str, int]:
    total = sum(counts.values())
    if sample_size == 0:
        return {key: 0 for key in counts}
    if total == 0:
        raise ValueError("数据为空，无法抽样。")
    if sample_size > total:
        raise ValueError(f"样本数量 {sample_size} 超过可用数据量 {total}。")

    quotas = {key: 0 for key in counts}
    fractions = []
    allocated = 0
    for key, count in counts.items():
        if count == 0:
            fractions.append((0.0, key))
            continue
        raw = count * sample_size / total
        base = min(count, int(math.floor(raw)))
        quotas[key] = base
        allocated += base
        fractions.append((raw - math.floor(raw), key))

    remaining = sample_size - allocated
    sorted_fracs = sorted(fractions, key=lambda x: x[0], reverse=True)

    while remaining > 0:
        progressed = False
        for _, key in sorted_fracs:
            available = counts[key] - quotas[key]
            if available <= 0:
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("数据不足，无法满足按比例抽样要求。")

    return quotas


def stratified_sample_groups(
    groups: Dict[str, List[Dict[str, Any]]],
    train_size: int,
    test_size: int,
    rng: random.Random,
) -> Dict[str, List[Dict[str, Any]]]:
    counts = {key: len(records) for key, records in groups.items()}
    total_available = sum(counts.values())
    required = train_size + test_size
    if total_available < required:
        raise ValueError(f"数据量不足：可用 {total_available}，需 {required}。")

    train_quota = allocate_counts(counts, train_size)
    test_quota = allocate_counts(counts, test_size)

    selections = {"train": [], "test": []}

    for key, records in groups.items():
        need = train_quota.get(key, 0) + test_quota.get(key, 0)
        if need == 0:
            continue
        if len(records) < need:
            raise ValueError(f"key={key} 可用 {len(records)} < 需 {need}。")
        rng.shuffle(records)
        train_cnt = train_quota.get(key, 0)
        test_cnt = test_quota.get(key, 0)
        selections["train"].extend(records[:train_cnt])
        selections["test"].extend(records[train_cnt : train_cnt + test_cnt])

    return selections


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


def split_records_by_backend(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        backend = record.get("type")
        if backend is None:
            raise ValueError("记录缺少 type 字段。")
        buckets.setdefault(backend, []).append(record)
    return buckets


def build_dataset_from_records(
    records_by_backend: Dict[str, List[Dict[str, Any]]],
    split_name: str,
    schema_caches: Dict[str, Dict[str, Any]],
    seed: int,
) -> datasets.Dataset:
    ds_list: List[datasets.Dataset] = []
    for backend, records in records_by_backend.items():
        if not records:
            continue
        data_source = "plfactory"
        ds = convert_split(
            raw_records=records,
            split_name=split_name,
            data_source=data_source,
            schema_cache=schema_caches[backend],
            backend=backend,
        )
        ds_list.append(ds)

    if not ds_list:
        raise ValueError(f"{split_name} 数据为空，无法构建数据集。")

    combined = ds_list[0] if len(ds_list) == 1 else datasets.concatenate_datasets(ds_list)
    return combined.shuffle(seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postgres_jsons", nargs=2, required=True, help="两个 Postgres JSON 文件路径")
    parser.add_argument("--oracle_jsons", nargs=2, required=True, help="两个 Oracle JSON 文件路径")
    parser.add_argument("--local_save_dir", default="/workspace/opt/projects/verlpl/examples/results")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_output", default="plfactory_rl_train.parquet")
    parser.add_argument("--test_output", default="plfactory_rl_test.parquet")
    parser.add_argument("--train_per_file", type=int, default=750)
    parser.add_argument("--test_per_file", type=int, default=125)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    postgres_util.initialize_dataset_paths()
    pg_schema_cache = postgres_util.get_database_schema_json()

    oracle_util.initialize_dataset_paths()
    oc_schema_cache = oracle_util.get_database_schema_json()

    file_specs = []
    for idx, path in enumerate(args.postgres_jsons, start=1):
        groups = load_grouped_json(path)
        file_specs.append(
            {
                "path": os.path.expanduser(path),
                "backend": "postgres",
                "label": f"postgres_{idx}",
                "groups": groups,
            }
        )

    for idx, path in enumerate(args.oracle_jsons, start=1):
        groups = load_grouped_json(path)
        file_specs.append(
            {
                "path": os.path.expanduser(path),
                "backend": "oracle",
                "label": f"oracle_{idx}",
                "groups": groups,
            }
        )

    train_records_all: List[Dict[str, Any]] = []
    test_records_all: List[Dict[str, Any]] = []

    for spec in file_specs:
        annotate_records_with_type(spec["groups"], spec["backend"])
        selections = stratified_sample_groups(
            groups=spec["groups"],
            train_size=args.train_per_file,
            test_size=args.test_per_file,
            rng=rng,
        )
        train_records_all.extend(selections["train"])
        test_records_all.extend(selections["test"])
        print(
            f"[INFO] {spec['label']} ({spec['backend']}) -> "
            f"train {len(selections['train'])}, test {len(selections['test'])}"
        )

    expected_train = args.train_per_file * len(file_specs)
    expected_test = args.test_per_file * len(file_specs)
    if len(train_records_all) != expected_train:
        raise RuntimeError(f"训练样本应为 {expected_train} 条，实际 {len(train_records_all)} 条。")
    if len(test_records_all) != expected_test:
        raise RuntimeError(f"测试样本应为 {expected_test} 条，实际 {len(test_records_all)} 条。")

    schema_caches = {"postgres": pg_schema_cache, "oracle": oc_schema_cache}

    train_dataset = build_dataset_from_records(
        split_records_by_backend(train_records_all),
        split_name="train",
        schema_caches=schema_caches,
        seed=args.seed,
    )
    test_dataset = build_dataset_from_records(
        split_records_by_backend(test_records_all),
        split_name="test",
        schema_caches=schema_caches,
        seed=args.seed,
    )

    train_path = os.path.join(save_dir, args.train_output)
    test_path = os.path.join(save_dir, args.test_output)

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"[INFO] 训练数据写入: {train_path} (共 {len(train_dataset)} 条，预期 {expected_train})")
    print(f"[INFO] 测试数据写入: {test_path} (共 {len(test_dataset)} 条，预期 {expected_test})")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=save_dir, dst=args.hdfs_dir)
        print(f"[INFO] 已同步至 HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    main()