import os
import re
import json
import subprocess
from typing import List, Optional

import psycopg
from psycopg import sql as psql
from psycopg import errors
import pandas as pd

from pl_setting import pg_config, get_dataset_config


conn_info = f"host={pg_config['host']} user={pg_config['user']} password={pg_config['password']} dbname={pg_config['dbname']} port={pg_config['port']}"
host = pg_config['host']
port = pg_config['port']
user = pg_config['user']
password = pg_config['password']

input_path = None
db_schema_graph_path = None
db_schema_dict_path = None


def analyze_plsql_objects(plsql_code):
    objects = []
    patterns = [
        ('function', r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+"?(\w+)"?'),
        ('procedure', r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+"?(\w+)"?'),
        ('trigger', r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+"?(\w+)"?')
    ]
    for obj_type, pattern in patterns:
        matches = re.finditer(pattern, plsql_code, re.IGNORECASE)
        objects.extend((obj_type, match.group(1)) for match in matches)
    return objects


def get_plsql_type(plsql_code):
    objects = analyze_plsql_objects(plsql_code)
    for obj_type, _ in objects:
        if obj_type == 'trigger':
            return 'trigger'
    for obj_type, _ in objects:
        return obj_type
    return 'unknown'


def execute_sql(database_conn_info, sql):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {2 * 1000};")
            cur.execute(sql)


def fetch_query_results(database_conn_info, query):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            return result


def get_all_user_tables(database_name):
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY tablename;
            """)
            result = cur.fetchall()
            return [table_name[0] for table_name in result]


def get_important_system_tables():
    return [
        'pg_indexes',
        'pg_constraints',
        'pg_triggers',
        'pg_sequences',
        'pg_views',
        'pg_user_mappings',
        'pg_policies',
        'pg_rules'
    ]


def fetch_system_table_data(database_conn_info, system_table):
    try:
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
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
                """)
                if not cur.fetchone()[0]:
                    return None
                cur.execute(f"SELECT * FROM {system_table} ORDER BY 1;")
                result = cur.fetchall()
                return result
    except Exception as e:
        print(f"Warning: Could not fetch data from system table {system_table}: {e}")
        return None


def recreate_databases(conn_info, databases, maintenance_db="postgres"):
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
                cur.execute(psql.SQL("CREATE DATABASE {}").format(ident))


def import_database(host, port, user, password, dbname, input_file):
    command = f"psql -h {host} -p {port} -U {user} -d {dbname} -f {input_file}"
    env = {**os.environ, "PGPASSWORD": password}
    subprocess.run(command, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def restore_databases(conn_info, host, port, user, password, database_names):
    try:
        recreate_databases(conn_info, database_names)
        for dbname in database_names:
            input_file = os.path.join(input_path, f"{dbname}.sql")
            import_database(host, port, user, password, dbname.lower(), input_file)
    except Exception as e:
        print(f"Error restoring databases {database_names}: {e}")


def compare_plsql_function(database_name, plsql1, plsql2, call_plsqls):
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name}"""

        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql1)
        function_results1 = {}
        for i, call_plsql in enumerate(call_plsqls):
            try:
                result = fetch_query_results(database_conn_info, call_plsql)
                function_results1[i] = {
                    'sql': call_plsql,
                    'result': pd.DataFrame(result)
                }
            except Exception as e:
                function_results1[i] = {
                    'sql': call_plsql,
                    'result': None,
                    'error': str(e)
                }

        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql2)
        function_results2 = {}
        for i, call_plsql in enumerate(call_plsqls):
            try:
                result = fetch_query_results(database_conn_info, call_plsql)
                function_results2[i] = {
                    'sql': call_plsql,
                    'result': pd.DataFrame(result)
                }
            except Exception as e:
                function_results2[i] = {
                    'sql': call_plsql,
                    'result': None,
                    'error': str(e)
                }

        total_calls = len(call_plsqls)
        if total_calls == 0:
            return 1.0

        score = 0.0
        for i in range(total_calls):
            res1 = function_results1.get(i, {})
            res2 = function_results2.get(i, {})
            df1 = res1.get('result')
            df2 = res2.get('result')

            if df1 is None and df2 is None:
                score += 1
            elif df1 is not None and df2 is not None:
                try:
                    if df1.equals(df2):
                        score += 1
                except Exception:
                    pass

        return score / total_calls
    except Exception as e:
        print("Error in compare_plsql_function")
        print(e)
        return 0.0


def compare_plsql(database_name, plsql1, plsql2, call_plsqls, include_system_tables):
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name}"""

        all_user_tables = get_all_user_tables(database_name)
        important_system_tables = get_important_system_tables() if include_system_tables else []

        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql1)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        user_tables_results1 = {}
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                user_tables_results1[table] = pd.DataFrame(result)
            except Exception:
                user_tables_results1[table] = None

        system_tables_results1 = {}
        for sys_table in important_system_tables:
            system_tables_results1[sys_table] = fetch_system_table_data(database_conn_info, sys_table)

        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql2)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        user_tables_results2 = {}
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                user_tables_results2[table] = pd.DataFrame(result)
            except Exception:
                user_tables_results2[table] = None

        system_tables_results2 = {}
        for sys_table in important_system_tables:
            system_tables_results2[sys_table] = fetch_system_table_data(database_conn_info, sys_table)

        score_user = 0.0
        for table in all_user_tables:
            df1 = user_tables_results1.get(table)
            df2 = user_tables_results2.get(table)
            if df1 is None and df2 is None:
                score_user += 1
            elif df1 is not None and df2 is not None:
                try:
                    if df1.equals(df2):
                        score_user += 1
                except Exception:
                    pass

        score_system = 0.0
        for sys_table in important_system_tables:
            result1 = system_tables_results1.get(sys_table)
            result2 = system_tables_results2.get(sys_table)
            if result1 is None and result2 is None:
                score_system += 1
            elif result1 is not None and result2 is not None and result1 == result2:
                score_system += 1

        total_targets = len(all_user_tables) + len(important_system_tables)
        if total_targets == 0:
            return 1.0

        return (score_user + score_system) / total_targets
    except Exception as e:
        print(f"Error in compare_plsql: {e}")
        return 0.0


def compute_score(solution_str, ground_truth, extra_info, format_score=0.0, score=1.0):
    if not solution_str or not ground_truth or not extra_info:
        return 0.0

    database_name = extra_info.get("database_name")
    call_sqls = extra_info.get("call_sqls", [])
    if not database_name:
        return 0.0

    plsql_type = get_plsql_type(solution_str)

    try:
        if plsql_type == 'function':
            semantic_score = compare_plsql_function(
                database_name=database_name,
                plsql1=solution_str,
                plsql2=ground_truth,
                call_plsqls=call_sqls
            )
        else:
            semantic_score = compare_plsql(
                database_name=database_name,
                plsql1=solution_str,
                plsql2=ground_truth,
                call_plsqls=call_sqls,
                include_system_tables=True
            )
    except Exception as e:
        print(f"compute_score error: {e}")
        return 0.0

    if semantic_score is None:
        return 0.0

    semantic_score = max(0.0, min(1.0, float(semantic_score)))
    return format_score + semantic_score * (score - format_score)