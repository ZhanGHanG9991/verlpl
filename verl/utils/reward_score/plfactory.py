import os
import re
import json
import subprocess
import threading
import hashlib
import time
import random
from pathlib import Path
from typing import Dict, List, Optional

import psycopg
from psycopg import sql as psql
from psycopg import errors
import pandas as pd
import oracledb

from .pl_setting import pg_config, oc_config


# ============================================================
# 配置部分
# ============================================================

pg_conn_info = (
    f"host={pg_config['host']} "
    f"user={pg_config['user']} "
    f"password={pg_config['password']} "
    f"dbname={pg_config['dbname']} "
    f"port={pg_config['port']}"
)
pg_host = pg_config["host"]
pg_port = pg_config["port"]
pg_user = pg_config["user"]
pg_password = pg_config["password"]

oc_conn_info = (
    f"host={oc_config['host']} "
    f"user={oc_config['user']} "
    f"password={oc_config['password']} "
    f"service_name={oc_config['service_name']} "
    f"port={oc_config['port']}"
)
oc_host = oc_config["host"]
oc_port = oc_config["port"]
oc_user = oc_config["user"]
oc_password = oc_config["password"]
oc_service_name = oc_config["service_name"]

pg_input_path = "/workspace/opt/projects/verlpl/examples/datasets/train/database/postgresql"
oc_input_path = "/workspace/opt/projects/verlpl/examples/datasets/train/database/oracle"


# ============================================================
# Worker ID 生成
# ============================================================

def _get_worker_id() -> str:
    """
    生成短且唯一的 worker 标识符。
    格式: "w" + 8位十六进制 (共9字符)
    添加时间戳和随机数避免同一线程多次调用时冲突
    """
    pid = os.getpid()
    tid = threading.get_ident()
    timestamp = time.time_ns()
    rand = random.randint(0, 0xFFFF)
    hash_input = f"{pid}_{tid}_{timestamp}_{rand}".encode()
    hash_hex = hashlib.md5(hash_input).hexdigest()[:8]
    return f"w{hash_hex}"


def _log(worker_id: str, message: str, level: str = "INFO"):
    """
    统一的日志输出函数，带有 worker_id 前缀便于多线程追踪。
    
    输出格式: [worker_id][level] message
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    # 使用分隔线包裹多行内容，便于观察
    if "\n" in message:
        separator = "=" * 60
        print(f"[{worker_id}][{timestamp}][{level}] {separator}")
        for line in message.split("\n"):
            print(f"[{worker_id}][{timestamp}][{level}] {line}")
        print(f"[{worker_id}][{timestamp}][{level}] {separator}")
    else:
        print(f"[{worker_id}][{timestamp}][{level}] {message}")


def _truncate_sql(sql: str, max_lines: int = 0, max_chars: int = 0) -> str:
    """
    格式化 SQL 用于日志输出。
    
    参数:
        sql: SQL 字符串
        max_lines: 最大行数，0 表示不限制
        max_chars: 最大字符数，0 表示不限制
    """
    if not sql:
        return "<empty>"
    
    result = sql
    
    # 如果设置了行数限制
    if max_lines > 0:
        lines = result.split("\n")
        if len(lines) > max_lines:
            result = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    
    # 如果设置了字符数限制
    if max_chars > 0 and len(result) > max_chars:
        result = result[:max_chars] + f"... ({len(sql) - max_chars} more chars)"
    
    return result


# ============================================================
# 辅助函数
# ============================================================

def _normalize_plsql_block(plsql_code: str) -> str:
    if not plsql_code:
        return ""
    cleaned = plsql_code.strip()
    cleaned = re.sub(r"\n/\s*$", "\n", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def _strip_think_blocks_impl(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _extract_plsql_content_impl(text: str, strip_fn) -> str:
    if not text:
        return ""
    
    cleaned = strip_fn(text).strip()
    TAG_VARIANT = r"(?:pl)?(?:pg)?sql"
    
    # 方法1: Markdown 代码块 (优先级最高)
    md_pattern = r"```(?:sql|plsql|pgsql|plpgsql|pl/?sql|pl/?pgsql)?\s*\n?(.*?)```"
    match_md = re.search(md_pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if match_md:
        content = match_md.group(1).strip()
        if content:
            return _normalize_plsql_ending(content)
    
    # 方法2: XML 标签提取
    start_tag_pattern = rf"<start-{TAG_VARIANT}>"
    match_start = re.search(start_tag_pattern, cleaned, re.IGNORECASE)
    if match_start:
        content_start = match_start.end()
        remaining = cleaned[content_start:]
        end_tag_pattern = rf"</?(?:end|start)-{TAG_VARIANT}>"
        match_end = re.search(end_tag_pattern, remaining, re.IGNORECASE)
        if match_end:
            content = remaining[:match_end.start()].strip()
        else:
            content = remaining.strip()
        if content:
            return _normalize_plsql_ending(content)
    
    # 方法3: 基于关键字截断
    keyword_pattern = r'(?:^|\n)\s*((?:CREATE(?:\s+OR\s+REPLACE)?|DECLARE|BEGIN|DO)\s+.*)'
    match_keyword = re.search(keyword_pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if match_keyword:
        content = match_keyword.group(1).strip()
        return _normalize_plsql_ending(content)
    
    return _normalize_plsql_ending(cleaned)


def _normalize_plsql_ending(content: str) -> str:
    if not content:
        return ""
    content = content.strip()
    TAG_VARIANT = r"(?:pl)?(?:pg)?sql"
    content = re.sub(rf'\s*</?(?:end-|start-)?{TAG_VARIANT}>\s*$', '', content, flags=re.IGNORECASE).strip()
    content = re.sub(r'\s*```\s*$', '', content).strip()
    if re.search(r'\$\$\s*LANGUAGE\s+\w+\s*$', content, re.IGNORECASE):
        content = content.rstrip() + ';'
    return content


def _analyze_plsql_objects_impl(plsql_code: str) -> List[tuple]:
    objects = []
    patterns = [
        ("function", r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+"?(\w+)"?'),
        ("procedure", r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+"?(\w+)"?'),
        ("trigger", r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+"?(\w+)"?'),
    ]
    for obj_type, pattern in patterns:
        matches = re.finditer(pattern, plsql_code, re.IGNORECASE)
        objects.extend((obj_type, match.group(1)) for match in matches)
    return objects


def _get_plsql_type_impl(plsql_code: str) -> str:
    objects = _analyze_plsql_objects_impl(plsql_code)
    for obj_type, _ in objects:
        if obj_type == "trigger":
            return "trigger"
    for obj_type, _ in objects:
        return obj_type
    return "unknown"


# ============================================================
# PostgreSQL 部分
# ============================================================

def pg_analyze_plsql_objects(plsql_code: str) -> List[tuple]:
    return _analyze_plsql_objects_impl(plsql_code)


def pg_get_plsql_type(plsql_code: str) -> str:
    return _get_plsql_type_impl(plsql_code)


def pg_strip_think_blocks(text: str) -> str:
    return _strip_think_blocks_impl(text)


def pg_extract_plsql_content(text: str) -> str:
    return _extract_plsql_content_impl(text, pg_strip_think_blocks)


def pg_execute_sql(database_conn_info: str, sql: str):
    normalized = _normalize_plsql_block(sql)
    if not normalized:
        return
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = %s;", (2 * 1000,))
            cur.execute(normalized)


def pg_fetch_query_results(database_conn_info: str, query: str):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
    return result


def pg_get_all_user_tables(database_name: str) -> List[str]:
    conn_db_info = (
        f"host={pg_host} dbname={database_name} user={pg_user} password={pg_password}"
    )
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY tablename;
                """
            )
            result = cur.fetchall()
    return [table_name[0] for table_name in result]


def pg_get_important_system_tables() -> List[str]:
    return [
        "pg_indexes",
        "pg_constraints",
        "pg_triggers",
        "pg_sequences",
        "pg_views",
        "pg_user_mappings",
        "pg_policies",
        "pg_rules",
    ]


def pg_fetch_system_table_data(database_conn_info: str, system_table: str):
    try:
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_name = '{system_table}'
                        AND table_schema IN ('pg_catalog', 'information_schema')
                    ) OR EXISTS (
                        SELECT 1
                        FROM information_schema.views
                        WHERE table_name = '{system_table}'
                        AND table_schema IN ('pg_catalog', 'information_schema')
                    );
                """
                )
                if not cur.fetchone()[0]:
                    return None
                cur.execute(f'SELECT * FROM "{system_table}" ORDER BY 1;')
                result = cur.fetchall()
                return result
    except Exception as exc:
        print(f"Warning: Could not fetch data from system table {system_table}: {exc}")
        return None


def pg_recreate_database(conn_info: str, db_name: str, maintenance_db: str = "postgres"):
    """重新创建单个数据库"""
    dsn = psycopg.conninfo.make_conninfo(conn_info, dbname=maintenance_db)
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                (db_name,),
            )
            ident = psql.Identifier(db_name)
            cur.execute(psql.SQL("DROP DATABASE IF EXISTS {}").format(ident))
            try:
                cur.execute(psql.SQL("CREATE DATABASE {}").format(ident))
            except psycopg.errors.DuplicateDatabase:
                pass


def pg_import_database(host, port, user, password, dbname, input_file):
    """从 SQL 文件导入数据"""
    command = f"psql -h {host} -p {port} -U {user} -d {dbname} -v ON_ERROR_STOP=1 -f {input_file}"
    env = {**os.environ, "PGPASSWORD": password}
    
    print(f"Executing: {command}")
    
    process = subprocess.run(
        command, 
        env=env, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True 
    )
    
    if process.returncode != 0:
        error_msg = f"PSQL Import Failed for {dbname}!\nReturncode: {process.returncode}\nStderr:\n{process.stderr}\nStdout:\n{process.stdout}"
        print(error_msg) 
        raise RuntimeError(error_msg)
    else:
        print(f"Successfully imported {dbname}.")
        if process.stderr:
            print(f"Warnings/Notices:\n{process.stderr}")


def pg_restore_database(worker_db_name: str, original_db_name: str):
    """
    恢复数据库（无锁版本）。
    
    参数:
        worker_db_name: worker 专用的数据库名
        original_db_name: 原始数据库名（用于查找 SQL 文件）
    """
    input_file = os.path.join(pg_input_path, f"{original_db_name}.sql")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"SQL dump not found: {input_file} (original db: {original_db_name})"
        )
    
    pg_recreate_database(pg_conn_info, worker_db_name)
    pg_import_database(
        pg_host, pg_port, pg_user, pg_password, 
        worker_db_name, input_file
    )


def pg_cleanup_worker_database(worker_db_name: str):
    """清理 worker 数据库"""
    try:
        dsn = psycopg.conninfo.make_conninfo(pg_conn_info, dbname="postgres")
        with psycopg.connect(dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    (worker_db_name,),
                )
                ident = psql.Identifier(worker_db_name)
                cur.execute(psql.SQL("DROP DATABASE IF EXISTS {}").format(ident))
        print(f"Cleaned up worker database: {worker_db_name}")
    except Exception as exc:
        print(f"Warning: Failed to cleanup worker database {worker_db_name}: {exc}")


def pg_capture_table_state(database_conn_info: str, table_names: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
    """批量抓取多个表的数据状态，复用单个连接以优化性能。"""
    snapshots = {}
    try:
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                for table in table_names:
                    try:
                        cur.execute(f'SELECT * FROM "{table}" ORDER BY 1;')
                        result = cur.fetchall()
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                        snapshots[table] = pd.DataFrame(result, columns=columns)
                    except Exception as e:
                        print(f"Warning: Failed to capture table {table}: {e}")
                        snapshots[table] = None
    except Exception as e:
        print(f"Warning: Database connection failed: {e}")
        # 连接失败，只将未处理的表设为 None，保留已成功获取的数据
        for table in table_names:
            if table not in snapshots:
                snapshots[table] = None
    return snapshots


def pg_compare_plsql_function(
    worker_db_name: str, 
    original_db_name: str,
    plsql1: str, 
    plsql2: str, 
    call_plsqls: List[str]
):
    """比较两个 PL/SQL 函数"""
    try:
        database_conn_info = (
            f"host={pg_host} user={pg_user} password={pg_password} dbname={worker_db_name}"
        )
        
        # ========== 第一轮 ==========
        _log(worker_db_name, f"[Round 1] Restoring database from {original_db_name}.sql...")
        pg_restore_database(worker_db_name, original_db_name)
        
        _log(worker_db_name, f"[Round 1] Executing solution PLSQL (function)...")
        pg_execute_sql(database_conn_info, plsql1)
        
        function_results1 = {}
        _log(worker_db_name, f"[Round 1] Executing {len(call_plsqls)} call statement(s)...")
        for i, call_plsql in enumerate(call_plsqls):
            try:
                _log(worker_db_name, f"[Round 1] Executing call [{i+1}]: {call_plsql[:100]}...")
                result = pg_fetch_query_results(database_conn_info, call_plsql)
                function_results1[i] = {"sql": call_plsql, "result": pd.DataFrame(result)}
                _log(worker_db_name, f"[Round 1] Call [{i+1}] returned {len(result)} rows")
            except Exception as exc:
                _log(worker_db_name, f"[Round 1] Call [{i+1}] failed: {exc}", level="WARN")
                function_results1[i] = {"sql": call_plsql, "result": None, "error": str(exc)}

        # ========== 第二轮 ==========
        _log(worker_db_name, f"[Round 2] Restoring database from {original_db_name}.sql...")
        pg_restore_database(worker_db_name, original_db_name)
        
        _log(worker_db_name, f"[Round 2] Executing ground truth PLSQL (function)...")
        pg_execute_sql(database_conn_info, plsql2)
        
        function_results2 = {}
        _log(worker_db_name, f"[Round 2] Executing {len(call_plsqls)} call statement(s)...")
        for i, call_plsql in enumerate(call_plsqls):
            try:
                _log(worker_db_name, f"[Round 2] Executing call [{i+1}]: {call_plsql[:100]}...")
                result = pg_fetch_query_results(database_conn_info, call_plsql)
                function_results2[i] = {"sql": call_plsql, "result": pd.DataFrame(result)}
                _log(worker_db_name, f"[Round 2] Call [{i+1}] returned {len(result)} rows")
            except Exception as exc:
                _log(worker_db_name, f"[Round 2] Call [{i+1}] failed: {exc}", level="WARN")
                function_results2[i] = {"sql": call_plsql, "result": None, "error": str(exc)}

        # ========== 比较 ==========
        total_calls = len(call_plsqls)
        if total_calls == 0:
            _log(worker_db_name, f"[Compare] No call statements, returning 1.0")
            return 1.0

        _log(worker_db_name, f"[Compare] Comparing {total_calls} call results...")
        score = 0.0
        for i in range(total_calls):
            df1 = function_results1.get(i, {}).get("result")
            df2 = function_results2.get(i, {}).get("result")

            if df1 is None and df2 is None:
                # 两者都执行失败，视为匹配（都失败）
                _log(worker_db_name, f"[Compare] Call [{i+1}]: both None (both failed), +1")
                score += 1
            elif df1 is not None and df2 is not None:
                try:
                    if df1.equals(df2):
                        _log(worker_db_name, f"[Compare] Call [{i+1}]: MATCH, +1")
                        score += 1
                    else:
                        _log(worker_db_name, f"[Compare] Call [{i+1}]: MISMATCH (df1 shape: {df1.shape}, df2 shape: {df2.shape}), +0")
                except Exception as e:
                    _log(worker_db_name, f"[Compare] Call [{i+1}]: comparison error {e}, +0", level="WARN")
            else:
                # 一方为 None 另一方不为 None，不加分（不匹配）
                _log(worker_db_name, f"[Compare] Call [{i+1}]: one None, other not (solution={'None' if df1 is None else 'OK'}, ground_truth={'None' if df2 is None else 'OK'}), +0")

        final_score = score / total_calls
        _log(worker_db_name, f"[Compare] Final score: {score}/{total_calls} = {final_score}")
        return final_score
        
    except Exception as exc:
        _log(worker_db_name, f"Error in pg_compare_plsql_function: {exc}", level="ERROR")
        import traceback
        _log(worker_db_name, f"Traceback:\n{traceback.format_exc()}", level="ERROR")
        return 0.0


def pg_compare_plsql(
    worker_db_name: str,
    original_db_name: str,
    plsql1: str, 
    plsql2: str, 
    call_plsqls: List[str]
):
    """比较两个 PL/SQL 对表的影响"""
    try:
        database_conn_info = (
            f"host={pg_host} user={pg_user} password={pg_password} dbname={worker_db_name}"
        )

        # ========== 第一轮 ==========
        _log(worker_db_name, f"[Round 1] Restoring database from {original_db_name}.sql...")
        pg_restore_database(worker_db_name, original_db_name)

        all_user_tables = pg_get_all_user_tables(worker_db_name)
        _log(worker_db_name, f"[Round 1] Found {len(all_user_tables)} user tables: {all_user_tables}")

        _log(worker_db_name, f"[Round 1] Executing solution PLSQL...")
        pg_execute_sql(database_conn_info, plsql1)
        _log(worker_db_name, f"[Round 1] Executing {len(call_plsqls)} call statement(s)...")
        for i, call in enumerate(call_plsqls):
            _log(worker_db_name, f"[Round 1] Executing call [{i+1}]: {call[:100]}...")
            pg_execute_sql(database_conn_info, call)

        user_tables_results1 = pg_capture_table_state(database_conn_info, all_user_tables)

        # ========== 第二轮 ==========
        _log(worker_db_name, f"[Round 2] Restoring database from {original_db_name}.sql...")
        pg_restore_database(worker_db_name, original_db_name)
        
        _log(worker_db_name, f"[Round 2] Executing ground truth PLSQL...")
        pg_execute_sql(database_conn_info, plsql2)
        _log(worker_db_name, f"[Round 2] Executing {len(call_plsqls)} call statement(s)...")
        for i, call in enumerate(call_plsqls):
            _log(worker_db_name, f"[Round 2] Executing call [{i+1}]: {call[:100]}...")
            pg_execute_sql(database_conn_info, call)

        user_tables_results2 = pg_capture_table_state(database_conn_info, all_user_tables)

        # ========== 比较 ==========
        _log(worker_db_name, f"[Compare] Comparing table states...")
        for table in all_user_tables:
            df1 = user_tables_results1.get(table)
            df2 = user_tables_results2.get(table)
            if df1 is None and df2 is None:
                _log(worker_db_name, f"[Compare] Table {table}: both None, skip")
                continue
            elif df1 is not None and df2 is not None:
                try:
                    if not df1.equals(df2):
                        _log(worker_db_name, f"[Compare] Table {table}: MISMATCH! Returning 0.0")
                        _log(worker_db_name, f"[Compare] df1 shape: {df1.shape}, df2 shape: {df2.shape}")
                        return 0.0
                    else:
                        _log(worker_db_name, f"[Compare] Table {table}: MATCH")
                except Exception as e:
                    _log(worker_db_name, f"[Compare] Table {table}: comparison error {e}, returning 0.0", level="ERROR")
                    return 0.0
            else:
                _log(worker_db_name, f"[Compare] Table {table}: one is None, other is not. Returning 0.0")
                return 0.0

        _log(worker_db_name, f"[Compare] All tables match! Returning 1.0")
        return 1.0
        
    except Exception as exc:
        _log(worker_db_name, f"Error in pg_compare_plsql: {exc}", level="ERROR")
        import traceback
        _log(worker_db_name, f"Traceback:\n{traceback.format_exc()}", level="ERROR")
        return 0.0


def pg_compute_score(solution_str, ground_truth, extra_info, format_score=0.0, score=1.0):
    """计算 PostgreSQL PL/SQL 的语义分数"""
    if not solution_str or not ground_truth or not extra_info:
        return 0.0

    original_db_name = extra_info.get("database_name")
    call_sqls = extra_info.get("call_sqls", []) or []
    if not original_db_name:
        return 0.0

    if len(call_sqls) == 0:
        return 0.0

    worker_db_name = _get_worker_id()
    
    # 提取 SQL 内容
    solution_sql = pg_extract_plsql_content(solution_str)
    ground_truth_sql = pg_extract_plsql_content(ground_truth)
    plsql_type = pg_get_plsql_type(solution_sql)
    
    # 详细日志输出
    _log(worker_db_name, f"=== PostgreSQL Score Computation Start ===")
    _log(worker_db_name, f"Original DB: {original_db_name}")
    _log(worker_db_name, f"PL/SQL Type: {plsql_type}")
    _log(worker_db_name, f"Call SQLs Count: {len(call_sqls)}")
    _log(worker_db_name, f"Solution SQL (extracted):\n{_truncate_sql(solution_sql)}")
    _log(worker_db_name, f"Ground Truth SQL (extracted):\n{_truncate_sql(ground_truth_sql)}")
    _log(worker_db_name, f"Call SQLs: {call_sqls}")

    try:
        if plsql_type == "function":
            _log(worker_db_name, "Comparing as FUNCTION...")
            semantic_score = pg_compare_plsql_function(
                worker_db_name=worker_db_name,
                original_db_name=original_db_name,
                plsql1=solution_sql,
                plsql2=ground_truth_sql,
                call_plsqls=call_sqls,
            )
        else:
            _log(worker_db_name, f"Comparing as {plsql_type.upper()} (procedure/trigger)...")
            semantic_score = 0
            for idx, call in enumerate(call_sqls):
                _log(worker_db_name, f"Processing call [{idx+1}/{len(call_sqls)}]: {call[:100]}...")
                per_semantic = pg_compare_plsql(
                    worker_db_name=worker_db_name,
                    original_db_name=original_db_name,
                    plsql1=solution_sql,
                    plsql2=ground_truth_sql,
                    call_plsqls=[call]
                )
                _log(worker_db_name, f"Call [{idx+1}] score: {per_semantic}")
                semantic_score += per_semantic
            semantic_score /= len(call_sqls)
        
        _log(worker_db_name, f"Final semantic score: {semantic_score}")
    except Exception as exc:
        _log(worker_db_name, f"ERROR: {exc}", level="ERROR")
        import traceback
        _log(worker_db_name, f"Traceback:\n{traceback.format_exc()}", level="ERROR")
        return 0.0
    finally:
        # 清理 worker 数据库，避免资源泄漏
        _log(worker_db_name, f"Cleaning up worker database...")
        pg_cleanup_worker_database(worker_db_name)
        _log(worker_db_name, f"=== PostgreSQL Score Computation End ===")

    return semantic_score if semantic_score is not None else 0.0


# ============================================================
# Oracle 部分
# ============================================================

def oc_analyze_plsql_objects(plsql_code: str) -> List[tuple]:
    return _analyze_plsql_objects_impl(plsql_code)


def oc_get_plsql_type(plsql_code: str) -> str:
    return _get_plsql_type_impl(plsql_code)


def oc_strip_think_blocks(text: str) -> str:
    return _strip_think_blocks_impl(text)


def oc_extract_plsql_content(text: str) -> str:
    return _extract_plsql_content_impl(text, oc_strip_think_blocks)


def oc_get_connection(user: str = oc_user, password: str = oc_password):
    return oracledb.connect(
        user=user,
        password=password,
        host=oc_host,
        port=oc_port,
        service_name=oc_service_name,
    )


def oc_split_sql_statements(sql_text: str) -> List[str]:
    statements = []
    buffer = []
    in_single = False
    in_double = False
    for char in sql_text:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        if char == ";" and not in_single and not in_double:
            statement = "".join(buffer).strip()
            if statement and not statement.startswith("--"):
                statements.append(statement)
            buffer = []
        else:
            buffer.append(char)
    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def oc_execute_sql_statements_in_schema(sql_file: str, schema_name: str, password: str):
    with open(sql_file, "r", encoding="utf-8") as sql_handle:
        sql_content = sql_handle.read()

    statements = oc_split_sql_statements(sql_content)
    if not statements:
        raise ValueError(f"No valid SQL statements found in {sql_file}")

    with oc_get_connection(user=schema_name, password=password) as schema_conn:
        cursor = schema_conn.cursor()
        success = 0
        for idx, statement in enumerate(statements, start=1):
            stmt = statement.strip()
            if not stmt or stmt.startswith("--"):
                continue
            try:
                cursor.execute(stmt)
                success += 1
            except oracledb.Error as exc:
                error_msg = str(exc).lower()
                if (
                    "name is already used by an existing object" in error_msg
                    and "create table" in stmt.lower()
                ):
                    table_match = re.search(
                        r'CREATE\s+TABLE\s+"?(\w+)"?', stmt, re.IGNORECASE
                    )
                    if table_match:
                        table_name = table_match.group(1)
                        try:
                            cursor.execute(
                                f'DROP TABLE "{table_name}" CASCADE CONSTRAINTS'
                            )
                            cursor.execute(stmt)
                            success += 1
                            continue
                        except Exception as drop_exc:
                            print(f"Failed to recreate table {table_name}: {drop_exc}")
                elif "does not exist" in error_msg and "drop" in stmt.lower():
                    continue
                else:
                    print(f"Warning: Failed to execute SQL [{idx}] in {schema_name}: {exc}")
                    continue
        schema_conn.commit()
    return success > 0


def oc_kill_user_sessions(cursor, username: str) -> int:
    """Kill all active sessions for a user. Returns number of killed sessions."""
    try:
        cursor.execute(
            "SELECT sid, serial# FROM v$session WHERE username = :username",
            username=username.upper()
        )
        sessions = cursor.fetchall()
        
        killed = 0
        for sid, serial in sessions:
            try:
                cursor.execute(f"ALTER SYSTEM KILL SESSION '{sid},{serial}' IMMEDIATE")
                killed += 1
            except oracledb.Error as e:
                print(f"Warning: Failed to kill session {sid},{serial}: {e}")
        return killed
    except oracledb.Error as e:
        print(f"Warning: Failed to query sessions for user {username}: {e}")
        return 0


def oc_recreate_schema_from_sql(worker_schema_name: str, original_schema_name: str):
    """重建 Oracle schema"""
    sql_file = os.path.join(oc_input_path, f"{original_schema_name.lower()}.sql")
    
    if not os.path.exists(sql_file):
        raise FileNotFoundError(
            f"Oracle SQL dump not found: {sql_file} (original schema: {original_schema_name})"
        )

    password = f"{worker_schema_name.lower()}_pwd"

    with oc_get_connection() as admin_conn:
        with admin_conn.cursor() as cursor:
            # 1. 尝试删除旧用户（带重试）
            max_retries = 3
            for attempt in range(max_retries):
                oc_kill_user_sessions(cursor, worker_schema_name)
                try:
                    cursor.execute(f"DROP USER {worker_schema_name} CASCADE")
                    break
                except oracledb.Error as exc:
                    error_msg = str(exc).lower()
                    if "does not exist" in error_msg:
                        break
                    elif "cannot drop a user that is currently connected" in error_msg:
                        if attempt < max_retries - 1:
                            print(f"User {worker_schema_name} still connected, retry {attempt + 1}/{max_retries}...")
                            time.sleep(0.5)
                        else:
                            raise RuntimeError(f"Failed to drop user {worker_schema_name} after {max_retries} attempts")
                    else:
                        raise

            # 2. 创建新用户
            cursor.execute(f"""
                CREATE USER {worker_schema_name} IDENTIFIED BY {password}
                DEFAULT TABLESPACE USERS
                TEMPORARY TABLESPACE TEMP
            """)
            cursor.execute(f"GRANT CONNECT, RESOURCE TO {worker_schema_name}")
            cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {worker_schema_name}")
        
        admin_conn.commit()

    # 3. 导入数据
    oc_execute_sql_statements_in_schema(sql_file, worker_schema_name, password)
    
    return password


def oc_restore_schema(worker_schema_name: str, original_schema_name: str):
    """恢复 Oracle schema（无锁版本）"""
    return oc_recreate_schema_from_sql(
        worker_schema_name.upper(), 
        original_schema_name.upper()
    )


def oc_cleanup_worker_schema(worker_schema_name: str):
    """清理 worker schema"""
    try:
        with oc_get_connection() as admin_conn:
            with admin_conn.cursor() as cursor:
                oc_kill_user_sessions(cursor, worker_schema_name)
                cursor.execute(f"DROP USER {worker_schema_name} CASCADE")
            admin_conn.commit()
        print(f"Cleaned up worker schema: {worker_schema_name}")
    except Exception as exc:
        print(f"Warning: Failed to cleanup worker schema {worker_schema_name}: {exc}")


def oc_execute_plsql(schema_name: str, plsql_code: str):
    normalized = _normalize_plsql_block(plsql_code)
    if not normalized:
        return
    with oc_get_connection() as conn:
        with conn.cursor() as cur:
            cur.call_timeout = 2 * 1000
            cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name.upper()}")
            cur.execute(normalized)
        conn.commit()


def oc_run_call_statements(schema_name: str, call_plsqls: List[str]) -> Dict[int, dict]:
    results: Dict[int, dict] = {}
    with oc_get_connection() as conn:
        with conn.cursor() as cur:
            cur.call_timeout = 2 * 1000
            cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name.upper()}")
            for idx, call_sql in enumerate(call_plsqls):
                try:
                    cur.execute(call_sql)
                    if cur.description:
                        rows = cur.fetchall()
                        columns = [desc[0] for desc in cur.description]
                        df = pd.DataFrame(rows, columns=columns)
                    else:
                        df = pd.DataFrame()
                    results[idx] = {"sql": call_sql, "result": df}
                except Exception as exc:
                    results[idx] = {"sql": call_sql, "result": None, "error": str(exc)}
    return results


def oc_get_all_user_tables(schema_name: str) -> List[str]:
    with oc_get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM all_tables
                WHERE owner = :schema_name
                AND table_name NOT LIKE 'BIN$%'
                ORDER BY table_name
                """,
                schema_name=schema_name.upper(),
            )
            rows = cur.fetchall()
    return [row[0] for row in rows]


def oc_get_important_system_tables() -> List[str]:
    return ["all_constraints", "all_triggers", "all_sequences", "all_views"]


def oc_fetch_system_table_data(system_table: str):
    try:
        with oc_get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT CASE
                        WHEN EXISTS (
                            SELECT 1 FROM all_tables WHERE table_name = :tbl
                        ) THEN 1
                        WHEN EXISTS (
                            SELECT 1 FROM all_views WHERE view_name = :tbl
                        ) THEN 1
                        ELSE 0
                    END
                    FROM dual
                    """,
                    tbl=system_table.upper(),
                )
                if not cur.fetchone()[0]:
                    return None
                cur.execute(
                    f"SELECT * FROM {system_table} WHERE ROWNUM <= 100 ORDER BY 1"
                )
                return cur.fetchall()
    except Exception as exc:
        print(f"Warning: Could not fetch Oracle system table {system_table}: {exc}")
        return None


def oc_capture_table_state(schema_name: str, table_names: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
    snapshots: Dict[str, Optional[pd.DataFrame]] = {}
    with oc_get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name.upper()}")
            for table in table_names:
                try:
                    cur.execute(f'SELECT * FROM "{table}" ORDER BY 1')
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                    snapshots[table] = pd.DataFrame(rows, columns=columns)
                except Exception:
                    snapshots[table] = None
    return snapshots


def oc_compare_plsql_function(
    worker_schema_name: str, 
    original_schema_name: str,
    plsql1: str, 
    plsql2: str, 
    call_plsqls: List[str]
):
    """比较两个 Oracle PL/SQL 函数"""
    try:
        worker_upper = worker_schema_name.upper()
        call_plsqls = call_plsqls or []

        # ========== 第一轮 ==========
        _log(worker_schema_name, f"[Round 1] Restoring schema from {original_schema_name}.sql...")
        oc_restore_schema(worker_schema_name, original_schema_name)
        
        _log(worker_schema_name, f"[Round 1] Executing solution PLSQL (function)...")
        oc_execute_plsql(worker_upper, plsql1)
        
        _log(worker_schema_name, f"[Round 1] Executing {len(call_plsqls)} call statement(s)...")
        results1 = oc_run_call_statements(worker_upper, call_plsqls)
        for i, res in results1.items():
            if res.get("result") is not None:
                _log(worker_schema_name, f"[Round 1] Call [{i+1}] returned {len(res['result'])} rows")
            else:
                _log(worker_schema_name, f"[Round 1] Call [{i+1}] failed: {res.get('error', 'unknown')}", level="WARN")

        # ========== 第二轮 ==========
        _log(worker_schema_name, f"[Round 2] Restoring schema from {original_schema_name}.sql...")
        oc_restore_schema(worker_schema_name, original_schema_name)
        
        _log(worker_schema_name, f"[Round 2] Executing ground truth PLSQL (function)...")
        oc_execute_plsql(worker_upper, plsql2)
        
        _log(worker_schema_name, f"[Round 2] Executing {len(call_plsqls)} call statement(s)...")
        results2 = oc_run_call_statements(worker_upper, call_plsqls)
        for i, res in results2.items():
            if res.get("result") is not None:
                _log(worker_schema_name, f"[Round 2] Call [{i+1}] returned {len(res['result'])} rows")
            else:
                _log(worker_schema_name, f"[Round 2] Call [{i+1}] failed: {res.get('error', 'unknown')}", level="WARN")

        # ========== 比较 ==========
        total_calls = len(call_plsqls)
        if total_calls == 0:
            _log(worker_schema_name, f"[Compare] No call statements, returning 1.0")
            return 1.0

        _log(worker_schema_name, f"[Compare] Comparing {total_calls} call results...")
        score = 0.0
        for idx in range(total_calls):
            df1 = results1.get(idx, {}).get("result")
            df2 = results2.get(idx, {}).get("result")
            if df1 is None and df2 is None:
                # 两者都执行失败，视为匹配（都失败）
                _log(worker_schema_name, f"[Compare] Call [{idx+1}]: both None (both failed), +1")
                score += 1
            elif df1 is not None and df2 is not None:
                try:
                    if df1.equals(df2):
                        _log(worker_schema_name, f"[Compare] Call [{idx+1}]: MATCH, +1")
                        score += 1
                    else:
                        _log(worker_schema_name, f"[Compare] Call [{idx+1}]: MISMATCH (df1 shape: {df1.shape}, df2 shape: {df2.shape}), +0")
                except Exception as e:
                    _log(worker_schema_name, f"[Compare] Call [{idx+1}]: comparison error {e}, +0", level="WARN")
            else:
                # 一方为 None 另一方不为 None，不加分（不匹配）
                _log(worker_schema_name, f"[Compare] Call [{idx+1}]: one None, other not (solution={'None' if df1 is None else 'OK'}, ground_truth={'None' if df2 is None else 'OK'}), +0")
        
        final_score = score / total_calls
        _log(worker_schema_name, f"[Compare] Final score: {score}/{total_calls} = {final_score}")
        return final_score
        
    except Exception as exc:
        _log(worker_schema_name, f"Error in oc_compare_plsql_function: {exc}", level="ERROR")
        import traceback
        _log(worker_schema_name, f"Traceback:\n{traceback.format_exc()}", level="ERROR")
        return 0.0


def oc_compare_plsql(
    worker_schema_name: str,
    original_schema_name: str,
    plsql1: str, 
    plsql2: str, 
    call_plsqls: List[str]
):
    """比较两个 Oracle PL/SQL 对表的影响"""
    try:
        worker_upper = worker_schema_name.upper()
        call_plsqls = call_plsqls or []

        # ========== 第一轮 ==========
        _log(worker_schema_name, f"[Round 1] Restoring schema from {original_schema_name}.sql...")
        oc_restore_schema(worker_schema_name, original_schema_name)

        all_user_tables = oc_get_all_user_tables(worker_upper)
        _log(worker_schema_name, f"[Round 1] Found {len(all_user_tables)} user tables: {all_user_tables}")
        
        _log(worker_schema_name, f"[Round 1] Executing solution PLSQL...")
        try:
            oc_execute_plsql(worker_upper, plsql1)
        except Exception as e:
            _log(worker_schema_name, f"[Round 1] Failed to execute solution PLSQL: {e}", level="ERROR")
            return 0.0
        
        _log(worker_schema_name, f"[Round 1] Executing {len(call_plsqls)} call statement(s)...")
        round1_call_failed = False
        for i, call in enumerate(call_plsqls):
            _log(worker_schema_name, f"[Round 1] Executing call [{i+1}]: {call[:100]}...")
            try:
                oc_execute_plsql(worker_upper, call)
            except Exception as e:
                _log(worker_schema_name, f"[Round 1] Call [{i+1}] failed: {e}", level="WARN")
                round1_call_failed = True

        user_tables_results1 = oc_capture_table_state(worker_upper, all_user_tables)

        # ========== 第二轮 ==========
        _log(worker_schema_name, f"[Round 2] Restoring schema from {original_schema_name}.sql...")
        oc_restore_schema(worker_schema_name, original_schema_name)
        
        _log(worker_schema_name, f"[Round 2] Executing ground truth PLSQL...")
        oc_execute_plsql(worker_upper, plsql2)
        _log(worker_schema_name, f"[Round 2] Executing {len(call_plsqls)} call statement(s)...")
        round2_call_failed = False
        for i, call in enumerate(call_plsqls):
            _log(worker_schema_name, f"[Round 2] Executing call [{i+1}]: {call[:100]}...")
            try:
                oc_execute_plsql(worker_upper, call)
            except Exception as e:
                _log(worker_schema_name, f"[Round 2] Call [{i+1}] failed: {e}", level="WARN")
                round2_call_failed = True

        user_tables_results2 = oc_capture_table_state(worker_upper, all_user_tables)

        # 如果 solution 执行失败但 ground_truth 成功，返回 0
        if round1_call_failed and not round2_call_failed:
            _log(worker_schema_name, f"[Compare] Solution calls failed but ground truth succeeded, returning 0.0")
            return 0.0

        # ========== 比较 ==========
        _log(worker_schema_name, f"[Compare] Comparing table states...")
        for table in all_user_tables:
            df1 = user_tables_results1.get(table)
            df2 = user_tables_results2.get(table)
            if df1 is None and df2 is None:
                _log(worker_schema_name, f"[Compare] Table {table}: both None, skip")
                continue
            elif df1 is not None and df2 is not None:
                try:
                    if not df1.equals(df2):
                        _log(worker_schema_name, f"[Compare] Table {table}: MISMATCH! Returning 0.0")
                        _log(worker_schema_name, f"[Compare] df1 shape: {df1.shape}, df2 shape: {df2.shape}")
                        return 0.0
                    else:
                        _log(worker_schema_name, f"[Compare] Table {table}: MATCH")
                except Exception as e:
                    _log(worker_schema_name, f"[Compare] Table {table}: comparison error {e}, returning 0.0", level="ERROR")
                    return 0.0
            else:
                _log(worker_schema_name, f"[Compare] Table {table}: one is None, other is not. Returning 0.0")
                return 0.0

        _log(worker_schema_name, f"[Compare] All tables match! Returning 1.0")
        return 1.0
        
    except Exception as exc:
        _log(worker_schema_name, f"Error in oc_compare_plsql: {exc}", level="ERROR")
        import traceback
        _log(worker_schema_name, f"Traceback:\n{traceback.format_exc()}", level="ERROR")
        return 0.0


def oc_compute_score(solution_str, ground_truth, extra_info, format_score=0.0, score=1.0):
    """计算 Oracle PL/SQL 的语义分数"""
    if not solution_str or not ground_truth or not extra_info:
        return 0.0

    original_schema_name = extra_info.get("database_name")
    call_sqls = extra_info.get("call_sqls", []) or []
    if not original_schema_name:
        return 0.0

    if len(call_sqls) == 0:
        return 0.0

    worker_schema_name = _get_worker_id()
    
    # 提取 SQL 内容
    solution_sql = oc_extract_plsql_content(solution_str)
    ground_truth_sql = oc_extract_plsql_content(ground_truth)
    plsql_type = oc_get_plsql_type(solution_sql)
    
    # 详细日志输出
    _log(worker_schema_name, f"=== Oracle Score Computation Start ===")
    _log(worker_schema_name, f"Original Schema: {original_schema_name}")
    _log(worker_schema_name, f"PL/SQL Type: {plsql_type}")
    _log(worker_schema_name, f"Call SQLs Count: {len(call_sqls)}")
    _log(worker_schema_name, f"Solution SQL (extracted):\n{_truncate_sql(solution_sql)}")
    _log(worker_schema_name, f"Ground Truth SQL (extracted):\n{_truncate_sql(ground_truth_sql)}")
    _log(worker_schema_name, f"Call SQLs: {call_sqls}")

    try:
        if plsql_type == "function":
            _log(worker_schema_name, "Comparing as FUNCTION...")
            semantic_score = oc_compare_plsql_function(
                worker_schema_name=worker_schema_name,
                original_schema_name=original_schema_name,
                plsql1=solution_sql,
                plsql2=ground_truth_sql,
                call_plsqls=call_sqls,
            )
        else:
            _log(worker_schema_name, f"Comparing as {plsql_type.upper()} (procedure/trigger)...")
            semantic_score = 0
            for idx, call in enumerate(call_sqls):
                _log(worker_schema_name, f"Processing call [{idx+1}/{len(call_sqls)}]: {call[:100]}...")
                per_semantic = oc_compare_plsql(
                    worker_schema_name=worker_schema_name,
                    original_schema_name=original_schema_name,
                    plsql1=solution_sql,
                    plsql2=ground_truth_sql,
                    call_plsqls=[call]
                )
                _log(worker_schema_name, f"Call [{idx+1}] score: {per_semantic}")
                semantic_score += per_semantic
            semantic_score /= len(call_sqls)
        
        _log(worker_schema_name, f"Final semantic score: {semantic_score}")
    except Exception as exc:
        _log(worker_schema_name, f"ERROR: {exc}", level="ERROR")
        import traceback
        _log(worker_schema_name, f"Traceback:\n{traceback.format_exc()}", level="ERROR")
        return 0.0
    finally:
        # 清理 worker schema，避免资源泄漏
        _log(worker_schema_name, f"Cleaning up worker schema...")
        oc_cleanup_worker_schema(worker_schema_name.upper())
        _log(worker_schema_name, f"=== Oracle Score Computation End ===")

    return semantic_score if semantic_score is not None else 0.0


# ============================================================
# 统一入口
# ============================================================

def compute_score(solution_str, ground_truth, extra_info, format_score=0.0, score=1.0):
    backend_type = (extra_info or {}).get("type", "postgres")
    if backend_type == "oracle":
        return oc_compute_score(solution_str, ground_truth, extra_info, format_score, score)
    return pg_compute_score(solution_str, ground_truth, extra_info, format_score, score)