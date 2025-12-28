import os
import re
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import fcntl
import psycopg
from psycopg import sql as psql
from psycopg import errors
import pandas as pd
import oracledb

from .pl_setting import pg_config, oc_config


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

LOCK_ROOT = Path(os.environ.get("PLFACTORY_DB_LOCK_DIR", "/tmp/plfactory_db_locks"))
LOCK_ROOT.mkdir(parents=True, exist_ok=True)


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
    tag_pattern = r"<start-plsql>\s*(.*?)\s*(?:</end-plsql>|</start-plsql>|<end-plsql>)"
    match = re.search(tag_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    open_tag_pattern = r"<start-plsql>\s*(.*)"
    match_open = re.search(open_tag_pattern, text, re.DOTALL | re.IGNORECASE)
    if match_open:
        return match_open.group(1).strip()

    md_pattern = r"```(?:sql|plsql|pl/?sql)?\s*\n?(.*?)```"
    match_md = re.search(md_pattern, text, re.DOTALL | re.IGNORECASE)
    if match_md:
        return match_md.group(1).strip()

    return cleaned


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


@contextmanager
def pg_database_restore_lock(dbname: str):
    lock_path = LOCK_ROOT / f"pg_{dbname}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: Could not fetch data from system table {system_table}: {exc}")
        return None


def pg_recreate_databases(
    conn_info: str, databases: List[str], maintenance_db: str = "postgres"
):
    dsn = psycopg.conninfo.make_conninfo(conn_info, dbname=maintenance_db)
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for db_name in databases:
                if db_name == maintenance_db:
                    continue
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
    # 加上 -v ON_ERROR_STOP=1，遇到第一个 SQL 错误就停止并返回非零状态码
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
        # 即使成功，也可以打印 stderr，因为可能有 NOTICE/WARNING
        if process.stderr:
            print(f"Warnings/Notices:\n{process.stderr}")


def pg_restore_databases(conn_info, host, port, user, password, database_names):
    for dbname in database_names:
        with pg_database_restore_lock(dbname):
            pg_recreate_databases(conn_info, [dbname])
            if pg_input_path is None:
                raise ValueError("input_path is not configured for database restore.")
            input_file = os.path.join(pg_input_path, f"{dbname}.sql")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"SQL dump not found: {input_file}")
            pg_import_database(host, port, user, password, dbname, input_file)


def pg_compare_plsql_function(database_name, plsql1, plsql2, call_plsqls):
    try:
        database_conn_info = (
            f"host={pg_host} user={pg_user} password={pg_password} dbname={database_name}"
        )
        print(f"Restoring database {database_name}...")
        pg_restore_databases(
            pg_conn_info, pg_host, pg_port, pg_user, pg_password, [database_name]
        )
        print("Executing first PLSQL...")
        pg_execute_sql(database_conn_info, plsql1)
        print("Executing first call statements...")
        function_results1 = {}
        for i, call_plsql in enumerate(call_plsqls):
            try:
                result = pg_fetch_query_results(database_conn_info, call_plsql)
                function_results1[i] = {
                    "sql": call_plsql,
                    "result": pd.DataFrame(result),
                }
            except Exception as exc:  # pylint: disable=broad-except
                function_results1[i] = {
                    "sql": call_plsql,
                    "result": None,
                    "error": str(exc),
                }

        pg_restore_databases(
            pg_conn_info, pg_host, pg_port, pg_user, pg_password, [database_name]
        )
        print("Executing second PLSQL...")
        pg_execute_sql(database_conn_info, plsql2)
        print("Executing second call statements...")
        function_results2 = {}
        for i, call_plsql in enumerate(call_plsqls):
            try:
                result = pg_fetch_query_results(database_conn_info, call_plsql)
                function_results2[i] = {
                    "sql": call_plsql,
                    "result": pd.DataFrame(result),
                }
            except Exception as exc:  # pylint: disable=broad-except
                function_results2[i] = {
                    "sql": call_plsql,
                    "result": None,
                    "error": str(exc),
                }

        total_calls = len(call_plsqls)
        if total_calls == 0:
            return 1.0

        score = 0.0
        for i in range(total_calls):
            res1 = function_results1.get(i, {})
            res2 = function_results2.get(i, {})
            df1 = res1.get("result")
            df2 = res2.get("result")

            if df1 is None and df2 is None:
                score += 1
            elif df1 is not None and df2 is not None:
                try:
                    if df1.equals(df2):
                        score += 1
                except Exception:  # pylint: disable=broad-except
                    pass

        return score / total_calls
    except Exception:  # pylint: disable=broad-except
        print("Error in pg_compare_plsql_function")
        return 0.0


def pg_compare_plsql(schema_name, plsql1, plsql2, call_plsqls):
    try:
        database_name = schema_name
        database_conn_info = (
            f"host={pg_host} user={pg_user} password={pg_password} dbname={database_name}"
        )

        print(f"Restoring database {database_name}...")
        pg_restore_databases(
            pg_conn_info, pg_host, pg_port, pg_user, pg_password, [database_name]
        )

        all_user_tables = pg_get_all_user_tables(database_name)

        print("Executing first PLSQL...")
        pg_execute_sql(database_conn_info, plsql1)
        print("Executing first call statements...")
        for call in call_plsqls:
            pg_execute_sql(database_conn_info, call)

        user_tables_results1 = {}
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = pg_fetch_query_results(database_conn_info, select_query)
                user_tables_results1[table] = pd.DataFrame(result)
            except Exception:  # pylint: disable=broad-except
                user_tables_results1[table] = None

        pg_restore_databases(
            pg_conn_info, pg_host, pg_port, pg_user, pg_password, [database_name]
        )
        print("Executing second PLSQL...")
        pg_execute_sql(database_conn_info, plsql2)
        print("Executing second call statements...")
        for call in call_plsqls:
            pg_execute_sql(database_conn_info, call)

        user_tables_results2 = {}
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = pg_fetch_query_results(database_conn_info, select_query)
                user_tables_results2[table] = pd.DataFrame(result)
            except Exception:  # pylint: disable=broad-except
                user_tables_results2[table] = None

        for table in all_user_tables:
            df1 = user_tables_results1.get(table)
            df2 = user_tables_results2.get(table)
            if df1 is None and df2 is None:
                continue
            elif df1 is not None and df2 is not None:
                try:
                    if not df1.equals(df2):
                        return 0.0
                except Exception:  # pylint: disable=broad-except
                    return 0.0
            else:
                return 0.0

        return 1.0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error in pg_compare_plsql: {exc}")
        return 0.0


def pg_compute_score(solution_str, ground_truth, extra_info, format_score=0.0, score=1.0):
    if not solution_str or not ground_truth or not extra_info:
        return 0.0

    database_name = extra_info.get("database_name")
    call_sqls = extra_info.get("call_sqls", []) or []
    if not database_name:
        return 0.0

    solution_sql = pg_extract_plsql_content(solution_str)
    ground_truth_sql = pg_extract_plsql_content(ground_truth)

    plsql_type = pg_get_plsql_type(solution_sql)

    try:
        if plsql_type == "function":
            semantic_score = pg_compare_plsql_function(
                database_name=database_name,
                plsql1=solution_sql,
                plsql2=ground_truth_sql,
                call_plsqls=call_sqls,
            )
        else:
            semantic_score = 0
            for call in call_sqls:
                per_semantic = pg_compare_plsql(
                    schema_name=database_name,
                    plsql1=solution_sql,
                    plsql2=ground_truth_sql,
                    call_plsqls=[call]
                )
                semantic_score += per_semantic
            semantic_score /= len(call_sqls)
    except Exception:  # pylint: disable=broad-except
        print("compute_score error")
        return 0.0

    if semantic_score is None:
        return 0.0

    return semantic_score


@contextmanager
def oc_database_restore_lock(schema_name: str):
    lock_path = LOCK_ROOT / f"oc_{schema_name.lower()}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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
            except oracledb.Error as exc:  # pylint: disable=broad-except
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
                        except Exception as drop_exc:  # pylint: disable=broad-except
                            print(
                                f"Failed to recreate table {table_name}: {drop_exc}"
                            )
                elif "does not exist" in error_msg and "drop" in stmt.lower():
                    continue
                else:
                    print(
                        f"Warning: Failed to execute SQL [{idx}] in {schema_name}: {exc}"
                    )
                    continue
        schema_conn.commit()
    return success > 0


def oc_recreate_schema_from_sql(schema_name: str):
    if not oc_input_path:
        raise ValueError("Oracle SQL dump directory is not configured.")
    sql_file = os.path.join(oc_input_path, f"{schema_name.lower()}.sql")
    if not os.path.exists(sql_file):
        raise FileNotFoundError(f"Oracle SQL dump not found: {sql_file}")

    password = f"{schema_name.lower()}_pwd"

    with oc_get_connection() as admin_conn:
        cursor = admin_conn.cursor()
        try:
            cursor.execute(f"DROP USER {schema_name} CASCADE")
        except oracledb.Error as exc:  # pylint: disable=broad-except
            if "does not exist" not in str(exc).lower():
                raise
        cursor.execute(
            f"""
            CREATE USER {schema_name} IDENTIFIED BY {password}
            DEFAULT TABLESPACE USERS
            TEMPORARY TABLESPACE TEMP
            """
        )
        cursor.execute(f"GRANT CONNECT, RESOURCE TO {schema_name}")
        cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {schema_name}")
        admin_conn.commit()

    oc_execute_sql_statements_in_schema(sql_file, schema_name, password)
    return password


def oc_restore_schema(schema_name: str):
    schema_upper = schema_name.upper()
    with oc_database_restore_lock(schema_upper):
        return oc_recreate_schema_from_sql(schema_upper)


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
                except Exception as exc:  # pylint: disable=broad-except
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
    except Exception as exc:  # pylint: disable=broad-except
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
                except Exception:  # pylint: disable=broad-except
                    snapshots[table] = None
    return snapshots


def oc_compare_plsql_function(schema_name, plsql1, plsql2, call_plsqls):
    try:
        schema_upper = schema_name.upper()
        call_plsqls = call_plsqls or []

        print(f"Restoring Oracle schema {schema_upper}...")
        oc_restore_schema(schema_upper)
        print("Executing first Oracle PLSQL...")
        oc_execute_plsql(schema_upper, plsql1)
        results1 = oc_run_call_statements(schema_upper, call_plsqls)

        oc_restore_schema(schema_upper)
        print("Executing second Oracle PLSQL...")
        oc_execute_plsql(schema_upper, plsql2)
        results2 = oc_run_call_statements(schema_upper, call_plsqls)

        total_calls = len(call_plsqls)

        score = 0.0
        for idx in range(total_calls):
            df1 = results1.get(idx, {}).get("result")
            df2 = results2.get(idx, {}).get("result")
            if df1 is None and df2 is None:
                score += 1
            elif df1 is not None and df2 is not None:
                try:
                    if df1.equals(df2):
                        score += 1
                except Exception:  # pylint: disable=broad-except
                    pass
        return score / total_calls
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error in oc_compare_plsql_function: {exc}")
        return 0.0


def oc_compare_plsql(schema_name, plsql1, plsql2, call_plsqls):
    try:
        schema_upper = schema_name.upper()
        call_plsqls = call_plsqls or []

        print(f"Restoring Oracle schema {schema_upper}...")
        oc_restore_schema(schema_upper)

        all_user_tables = oc_get_all_user_tables(schema_upper)
        
        print("Running first Oracle PLSQL...")
        oc_execute_plsql(schema_upper, plsql1)
        for call in call_plsqls:
            oc_execute_plsql(schema_upper, call)

        user_tables_results1 = oc_capture_table_state(schema_upper, all_user_tables)

        oc_restore_schema(schema_upper)
        print("Running second Oracle PLSQL...")
        oc_execute_plsql(schema_upper, plsql2)
        for call in call_plsqls:
            oc_execute_plsql(schema_upper, call)

        user_tables_results2 = oc_capture_table_state(schema_upper, all_user_tables)

        for table in all_user_tables:
            df1 = user_tables_results1.get(table)
            df2 = user_tables_results2.get(table)
            if df1 is None and df2 is None:
                continue
            elif df1 is not None and df2 is not None:
                try:
                    if not df1.equals(df2):
                        return 0.0
                except Exception:  # pylint: disable=broad-except
                    return 0.0
            else:
                return 0.0

        return 1.0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error in oc_compare_plsql: {exc}")
        return 0.0


def oc_compute_score(solution_str, ground_truth, extra_info, format_score=0.0, score=1.0):
    if not solution_str or not ground_truth or not extra_info:
        return 0.0

    database_name = extra_info.get("database_name")
    call_sqls = extra_info.get("call_sqls", []) or []
    if not database_name:
        return 0.0

    if len(call_sqls) == 0:
        return 0.0

    solution_sql = oc_extract_plsql_content(solution_str)
    ground_truth_sql = oc_extract_plsql_content(ground_truth)
    plsql_type = oc_get_plsql_type(solution_sql)

    try:
        if plsql_type == "function":
            semantic_score = oc_compare_plsql_function(
                schema_name=database_name,
                plsql1=solution_sql,
                plsql2=ground_truth_sql,
                call_plsqls=call_sqls,
            )
        else:
            semantic_score = 0
            for call in call_sqls:
                per_semantic = oc_compare_plsql(
                    schema_name=database_name,
                    plsql1=solution_sql,
                    plsql2=ground_truth_sql,
                    call_plsqls=[call]
                )
                semantic_score += per_semantic
            semantic_score /= len(call_sqls)
    except Exception as exc:
        print(f"oc_compute_score error: {exc}")
        return 0.0

    if semantic_score is None:
        return 0.0

    return semantic_score


def compute_score(solution_str, ground_truth, extra_info, format_score=0.0, score=1.0):
    backend_type = (extra_info or {}).get("type", "postgres")
    if backend_type == "oracle":
        return oc_compute_score(solution_str, ground_truth, extra_info, format_score, score)
    return pg_compute_score(solution_str, ground_truth, extra_info, format_score, score)