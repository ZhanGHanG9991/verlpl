#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the PLFactory PL/SQL dataset (JSON input) to Parquet format.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import datasets

from examples.data_preprocess import postgres_util
from verl.utils.hdfs_io import copy, makedirs


PROMPT_TEMPLATE = """You are an expert in PostgreSQL database and PL/pgSQL programming. Your task is to generate valid PL/pgSQL code based on the provided Intermediate Representation (IR) and Database Schema.

Database Context: PostgreSQL
Schema Information (Tables and Columns):
{databases_schema_info}

Intermediate Representation (IR) describing the logic:
{ir_text}

Please generate the corresponding PostgreSQL PL/SQL code.
IMPORTANT: You MUST wrap your final PL/SQL code strictly within <start-plsql> and <end-plsql> tags.
Example format:
<start-plsql>
CREATE OR REPLACE FUNCTION ...
<end-plsql>

Your response:"""


def format_prompt_for_local_model(databases_schema_info: str, ir_text: str) -> str:
    """Format the prompt using schema info and IR text."""
    return PROMPT_TEMPLATE.format(
        databases_schema_info=databases_schema_info.strip(),
        ir_text=ir_text.strip(),
    )


def load_json_dataset(json_path: str) -> datasets.Dataset:
    """Load a local JSON array into a Hugging Face Dataset."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file does not exist: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be an array of objects.")

    return datasets.Dataset.from_list(data)


def build_schema_prompt(
    schema_cache: Dict[str, Any],
    database_name: str,
    tables: List[str],
) -> str:
    """Generate schema prompt string using postgres_util helpers."""
    if schema_cache and database_name in schema_cache:
        schema_entry = schema_cache[database_name]
        try:
            return postgres_util.generate_schema_prompt_from_dict(
                schema_entry,
                tables if tables else None,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[Warning] Failed to generate schema prompt for "
                f"{database_name}: {exc}"
            )

    tables_str = ", ".join(tables) if tables else "N/A"
    return (
        f"Tables: {tables_str}\n"
        "Details: Schema information unavailable. Use best judgment based on IR."
    )


def build_example(
    example: Dict[str, Any],
    idx: int,
    split: str,
    data_source: str,
    schema_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Transform a raw record into VERL-compliant sample."""
    ir_text = example.get("ir", "")
    database_name = example.get("database_name", "")
    tables = example.get("tables") or []
    schema_prompt = build_schema_prompt(schema_cache, database_name, tables)
    prompt_text = format_prompt_for_local_model(schema_prompt, ir_text)

    ground_truth_plsql = example.get("plsql", "")

    extra_info = {
        "split": split,
        "index": idx,
        "ir": ir_text,
        "database_name": database_name,
        "tables": tables,
        "call_sqls": example.get("call_sqls"),
        "summary": example.get("summary"),
        "id": example.get("id"),
        "natural_language": example.get("natural_language"),
        "schema_prompt": schema_prompt,
    }

    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "ability": "sql",
        "reward_model": {"style": "rule", "ground_truth": ground_truth_plsql},
        "extra_info": extra_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_path",
        required=True,
        help="Path to the local JSON dataset file.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Dataset name used by postgres_util to locate schema metadata.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/home/zhanghang/opt/projects/researchprojects/plfactory/data",
        help="Directory to save the preprocessed Parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Destination HDFS directory (optional).",
    )
    args = parser.parse_args()

    json_path = os.path.expanduser(args.local_dataset_path)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    data_source = "plfactory"

    # Initialize postgres_util paths and load schema cache
    postgres_util.initialize_dataset_paths(args.dataset_name)
    schema_cache = postgres_util.get_database_schema_json()

    base_dataset = load_json_dataset(json_path)

    split_dataset = base_dataset.train_test_split(test_size=1 / 7, seed=42)
    train_raw = split_dataset["train"]
    test_raw = split_dataset["test"]

    def make_map_fn(split_name: str):
        def _inner(example, idx):
            return build_example(
                example,
                idx,
                split_name,
                data_source,
                schema_cache,
            )

        return _inner

    train_dataset = train_raw.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_raw.map(function=make_map_fn("test"), with_indices=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()