import os
import psycopg
from psycopg import sql as psql
from psycopg import errors
import subprocess
import pandas as pd
import sqlparse
from sqlparse import sql, tokens
from typing import List, Optional, Tuple, Dict, Set
import re
import json
from dataclasses import dataclass, field
from enum import Enum

from tqdm import tqdm
from pl_settings import pg_config, get_dataset_config

# Connection info
conn_info = f"host={pg_config['host']} user={pg_config['user']} password={pg_config['password']} dbname={pg_config['dbname']} port={pg_config['port']}"
host = pg_config['host']
port = pg_config['port']
user = pg_config['user']
password = pg_config['password']

# 这些变量需要通过 initialize_dataset_paths() 函数来设置
input_path = None
db_schema_graph_path = None
db_schema_dict_path = None

def initialize_dataset_paths(dataset_name: str):
    """
    根据数据集名称初始化路径配置
    
    Args:
        dataset_name: 数据集名称
    """
    global input_path, db_schema_graph_path, db_schema_dict_path
    
    dataset_config = get_dataset_config(dataset_name)
    input_path = "/workspace/opt/projects/researchprojects/plfactory/experiments/database/spider/postgres"
    db_schema_graph_path = "/workspace/opt/projects/researchprojects/plfactory/experiments/schema/spider/postgres_db_schema_graph.json"
    db_schema_dict_path = "/workspace/opt/projects/researchprojects/plfactory/experiments/schema/spider/postgres_db_schema_dict.json"

def get_tables_info(database_name):
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    tables_info = {}
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT tablename
                            FROM pg_catalog.pg_tables
                            WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';""")
            result = cur.fetchall()
            table_names = [table_name[0] for table_name in result]

            for table_name in table_names:
                cur.execute(f"""SELECT column_name, data_type
                                FROM information_schema.columns
                                WHERE table_name = '{table_name}';""")
                result = cur.fetchall()
                tables_info[table_name] = [item for item in result]
                
    return tables_info

def get_database_schema(database_name):
    """获取数据库schema信息，返回符合DatabaseSchema类型的字典
    
    Returns:
        Dict包含:
        - table_names: List[str] - 所有表名列表
        - tables: Dict[str, List[str]] - 表名到列名列表的映射
    """
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            # 获取所有表名
            cur.execute(f"""SELECT tablename
                            FROM pg_catalog.pg_tables
                            WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';""")
            result = cur.fetchall()
            table_names = [table_name[0] for table_name in result]
            
            # 获取每个表的列名
            tables = {}
            for table_name in table_names:
                cur.execute(f"""SELECT column_name
                                FROM information_schema.columns
                                WHERE table_name = '{table_name}'
                                ORDER BY ordinal_position;""")
                result = cur.fetchall()
                tables[table_name] = [column_name[0] for column_name in result]
    
    return {
        'table_names': table_names,
        'tables': tables
    }

def get_detailed_database_schema(database_name, sample_limit=3):
    """获取详细的数据库schema信息，适合text2sql任务
    
    Args:
        database_name: 数据库名
        sample_limit: 每表采样数据行数
    
    Returns:
        dict: 包含详细schema信息的字典，结构如下：
            {
                'database_name': str,
                'tables': {
                    'table1': {
                        'columns': [
                            {
                                'name': str,
                                'data_type': str,
                                'is_nullable': str,
                                'constraint_type': str,
                                'comment': str,
                                'examples': list
                            }
                        ],
                        'sample_data': list,
                        'column_names': list
                    }
                },
                'relationships': list,
                'formatted_string': str  # 保持原有格式的字符串
            }
    """
    conn_db_info = f"host={host} dbname={database_name} user={user} password={password}"
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            # 获取所有表名
            cur.execute("""
                SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY tablename;
            """)
            table_names = [row[0] for row in cur.fetchall()]
            
            schema_dict = {
                'database_name': database_name,
                'tables': {},
                'relationships': []
            }
            
            for table_name in table_names:                
                # 获取列详细信息
                cur.execute("""
                    SELECT 
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        CASE WHEN tc.constraint_type = 'PRIMARY KEY' THEN 'PRIMARY KEY' ELSE NULL END as primary_key,
                        col_description(pgc.oid, c.ordinal_position) as column_comment
                    FROM information_schema.columns c
                    LEFT JOIN information_schema.key_column_usage kcu
                        ON c.table_name = kcu.table_name 
                        AND c.column_name = kcu.column_name
                    LEFT JOIN information_schema.table_constraints tc
                        ON kcu.table_name = tc.table_name 
                        AND kcu.constraint_name = tc.constraint_name
                        AND tc.constraint_type = 'PRIMARY KEY'
                    LEFT JOIN pg_class pgc ON pgc.relname = c.table_name
                    WHERE c.table_name = %s
                    ORDER BY c.ordinal_position;
                """, (table_name,))
                
                columns_info = cur.fetchall()
                
                # 数据采样
                cur.execute(f"""
                    SELECT * FROM "{table_name}" LIMIT {sample_limit};
                """)
                sample_data = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]
                
                # 为每个列收集样例数据
                column_examples = {}
                if sample_data:
                    for i, row in enumerate(sample_data):
                        for j, col_name in enumerate(column_names):
                            if j < len(row):
                                if col_name not in column_examples:
                                    column_examples[col_name] = []
                                value_example = str(row[j])
                                if len(value_example) > 30:
                                    value_example = value_example[:30] + "..."
                                else:
                                    value_example = row[j]
                                if value_example not in column_examples[col_name]:
                                    column_examples[col_name].append(value_example)
                
                # 构建表结构字典
                table_info = {
                    'columns': [],
                    'sample_data': sample_data,
                    'column_names': column_names
                }
                
                for col in columns_info:
                    column_name, data_type, is_nullable, constraint_type, col_comment = col
                    
                    # 添加到字典
                    column_dict = {
                        'name': column_name,
                        'data_type': data_type,
                        'is_nullable': is_nullable,
                        'constraint_type': constraint_type,
                        'comment': col_comment,
                        'examples': column_examples.get(column_name, [])[:5]  # 最多5个样例
                    }
                    table_info['columns'].append(column_dict)
                    
                # 保存表信息到字典
                schema_dict['tables'][table_name] = table_info
              
            # 外键关系部分
            all_relationships = []
            
            for table_name in table_names:
                cur.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
                """, (table_name,))
                
                relationships = cur.fetchall()
                if relationships:
                    for rel in relationships:
                        col_name, foreign_table, foreign_col = rel
                        relationship_str = f"{table_name}.{col_name} = {foreign_table}.{foreign_col}"
                        all_relationships.append(relationship_str)
                        
                        # 添加到字典
                        schema_dict['relationships'].append({
                            'source_table': table_name,
                            'source_column': col_name,
                            'target_table': foreign_table,
                            'target_column': foreign_col
                        })
                        
    return schema_dict

# 辅助函数：从字典重新生成特定表的prompt
def generate_schema_prompt_from_dict(schema_dict, table_names=None):
    """从schema字典生成特定表的prompt字符串
    
    Args:
        schema_dict: get_detailed_database_schema返回的字典
        table_names: 要包含的表名列表，如果为None则包含所有表
    
    Returns:
        str: 格式化的schema描述字符串
    """
    if table_names is None:
        table_names = list(schema_dict['tables'].keys())
    
    formatted_parts = []
    
    for table_name in sorted(table_names):
        if table_name not in schema_dict['tables']:
            continue
            
        table_info = schema_dict['tables'][table_name]
        
        table_schema = f"Table: {table_name}\n"
        table_schema += "Columns:\n"
        
        for col in table_info['columns']:
            col_info = f"  - {col['name']} ({col['data_type']})"
            
            if col['constraint_type'] == 'PRIMARY KEY':
                col_info += " PRIMARY KEY"
            
            if col['is_nullable'] == 'NO':
                col_info += " NOT NULL"
            
            if col['examples']:
                col_info += f" examples: {col['examples']}"
            
            if col['comment']:
                col_info += f" - {col['comment']}"
            
            table_schema += col_info + "\n"
        
        formatted_parts.append(table_schema)
    
    # 添加关系信息
    relationship_summary = "\nTable Relationships:\n"
    if schema_dict['relationships']:
        relationships = []
        for rel in schema_dict['relationships']:
            if rel['source_table'] in table_names and rel['target_table'] in table_names:
                rel_str = f"{rel['source_table']}.{rel['source_column']} = {rel['target_table']}.{rel['target_column']}"
                relationships.append(rel_str)
        
        if relationships:
            relationship_summary += "\n".join(sorted(list(set(relationships))))
        else:
            relationship_summary += "No relationships between selected tables."
    else:
        relationship_summary += "No foreign key relationships found."
    
    formatted_parts.append(relationship_summary)
    
    return "\n".join(formatted_parts) + "\n"

def _get_foreign_key_relations(database_name):
    """获取数据库中所有表的外键关系
    
    Args:
        database_name: 数据库名称
        
    Returns:
        Dict[str, List[str]]: 字典，键为表名，值为与该表有外键关联的其他表名列表
    """
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    foreign_key_relations = {}
    
    try:
        with psycopg.connect(conn_db_info) as conn:
            with conn.cursor() as cur:
                # 获取所有用户表
                cur.execute("""
                    SELECT tablename
                    FROM pg_catalog.pg_tables
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY tablename;
                """)
                tables = [row[0] for row in cur.fetchall()]
                
                # 初始化所有表的外键关系列表
                for table in tables:
                    foreign_key_relations[table] = []
                
                # 查询所有外键关系
                cur.execute("""
                    SELECT
                        tc.table_name AS from_table,
                        ccu.table_name AS to_table
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY tc.table_name;
                """)
                
                for from_table, to_table in cur.fetchall():
                    # 添加双向关系
                    if to_table not in foreign_key_relations[from_table]:
                        foreign_key_relations[from_table].append(to_table)
                    if from_table not in foreign_key_relations[to_table]:
                        foreign_key_relations[to_table].append(from_table)
                
    except Exception as e:
        print(f"Error getting foreign key relations for database {database_name}: {e}")
        return {}
    
    return foreign_key_relations

def get_database_schema_graph():
    """获取所有PostgreSQL数据库的schema图谱
    
    Returns:
        Dict: 格式为 {
            "数据库名1": {
                "tables": ["table1", "table2"],
                "table1": [与table1有外键相连的table名]
            },
            ...
        }
    """
    if os.path.exists(db_schema_graph_path):
        # 如果文件已存在，直接加载返回
        with open(db_schema_graph_path, 'r', encoding='utf-8') as f:
            db_schema_graph = json.load(f)
        return db_schema_graph
    
    # 在线构建Dict
    db_schema_graph = {}
    
    # 获取input_path目录下的所有.sql文件
    if not os.path.exists(input_path):
        print(f"Warning: input_path {input_path} does not exist")
        return db_schema_graph
    
    sql_files = [f for f in os.listdir(input_path) if f.endswith('.sql')]
    
    for sql_file in sql_files[:10]:
        # 从文件名中提取数据库名（去除.sql后缀）
        database_name = sql_file[:-4]
        
        try:
            print(f"Processing database: {database_name}")
            
            # 获取所有用户表
            all_tables = get_all_user_tables(database_name)
            
            # 获取外键关系
            foreign_key_relations = _get_foreign_key_relations(database_name)
            
            # 构建该数据库的schema图
            db_schema_graph[database_name] = {
                "tables": all_tables
            }
            
            # 添加每个表的外键关系
            for table in all_tables:
                db_schema_graph[database_name][table] = foreign_key_relations.get(table, [])
                
        except Exception as e:
            print(f"Error processing database {database_name}: {e}")
            continue
    
    # 保存至db_schema_graph_path
    try:
        os.makedirs(os.path.dirname(db_schema_graph_path), exist_ok=True)
        with open(db_schema_graph_path, 'w', encoding='utf-8') as f:
            json.dump(db_schema_graph, f, ensure_ascii=False, indent=2)
        print(f"Database schema graph saved to {db_schema_graph_path}")
    except Exception as e:
        print(f"Error saving database schema graph: {e}")
    
    return db_schema_graph


def get_database_schema_json():
    """获取所有 PostgreSQL 数据库的 schema JSON

    """
    if os.path.exists(db_schema_dict_path):
        # 如果文件已存在，直接加载返回
        with open(db_schema_dict_path, 'r', encoding='utf-8') as f:
            db_schema_dict = json.load(f)
        return db_schema_dict
    
    # 在线构建Dict
    db_schema_dict = {}
    
    # 获取input_path目录下的所有.sql文件
    if not os.path.exists(input_path):
        print(f"Warning: input_path {input_path} does not exist")
        return db_schema_dict
    
    sql_files = [f for f in os.listdir(input_path) if f.endswith('.sql')]
    
    for sql_file in tqdm(sql_files):
        # 从文件名中提取数据库名（去除.sql后缀）
        database_name = sql_file[:-4]
        
        try:
            print(f"Processing database: {database_name}")
            db_schema_dict[database_name] = get_detailed_database_schema(database_name)
                
        except Exception as e:
            print(f"Error processing database {database_name}: {e}")
            continue
    
    # 保存至 db_schema_dict
    try:
        os.makedirs(os.path.dirname(db_schema_dict), exist_ok=True)
        with open(db_schema_dict_path, 'w', encoding='utf-8') as f:
            json.dump(db_schema_dict, f, ensure_ascii=False, indent=2)
        print(f"Database schema dict saved to {db_schema_dict}")
    except Exception as e:
        print(f"Error saving database schema dict: {e}")
    
    return db_schema_dict

def get_all_user_tables(database_name):
    """获取数据库中所有用户表的名称"""
    conn_db_info = f"""host={host} dbname={database_name} user={user} password={password}"""
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            # 获取所有用户表（排除系统表）
            cur.execute("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY tablename;
            """)
            result = cur.fetchall()
            return [table_name[0] for table_name in result]

def get_important_system_tables():
    """Return a list of system tables that are important for PostgreSQL"""
    return [
        'pg_indexes',               # 索引信息
        'pg_constraints',           # 约束信息
        'pg_triggers',              # 触发器信息
        'pg_sequences',             # 序列信息
        'pg_views',                 # 视图信息
        'pg_user_mappings',         # 用户映射
        'pg_policies',              # 行级安全策略
        'pg_rules'                  # 规则信息
    ]

def fetch_system_table_data(database_conn_info, system_table):
    """获取系统表数据，处理可能的权限或存在性问题"""
    try:
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                # 检查表是否存在
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
                
                # 尝试查询系统表数据
                cur.execute(f"SELECT * FROM {system_table} ORDER BY 1;")
                result = cur.fetchall()
                return result
    except Exception as e:
        print(f"Warning: Could not fetch data from system table {system_table}: {e}")
        return None

# 删除再创建数据库的函数
def recreate_databases(conn_info, databases, maintenance_db="postgres"):
    dsn = psycopg.conninfo.make_conninfo(conn_info, dbname=maintenance_db)

    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for db_name in databases:
                if db_name == maintenance_db:
                    continue

                # 断开目标数据库的所有连接
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
    
    process = subprocess.run(command, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def restore_databases(conn_info, host, port, user, password, database_names):
    try:
        # 删除并重建数据库
        recreate_databases(conn_info, database_names)

        # 导入数据
        for dbname in database_names:
            input_file = os.path.join(input_path, f"{dbname}.sql")
            import_database(host, port, user, password, dbname.lower(), input_file)
    except Exception as e:
        print(f"Error restoring databases {database_names}: {e}")

def cleanup_plsql_objects(plsql_code, database_name):
    """清理PL/SQL代码中可能创建的函数、存储过程和触发器"""
    try:
        database_conn_info = f"host={host} user={user} password={password} dbname={database_name} port={port}"
        
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                # 分析PL/SQL代码，识别创建的对象
                objects_to_drop = analyze_plsql_objects(plsql_code)
                
                # 清理识别到的对象
                for obj_type, obj_name in objects_to_drop:
                    if obj_type == 'function':
                        cur.execute(f"DROP FUNCTION IF EXISTS {obj_name} CASCADE")
                    elif obj_type == 'procedure':
                        cur.execute(f"DROP PROCEDURE IF EXISTS {obj_name} CASCADE")
                    elif obj_type == 'trigger':
                        cur.execute(f"DROP TRIGGER IF EXISTS {obj_name} CASCADE")
                        
    except Exception as e:
        return str(e)
    return None


def get_plsql_type(plsql_code):
    objects = analyze_plsql_objects(plsql_code)
    for obj_type, _ in objects:
        if obj_type == 'trigger':
            return 'trigger'
    for obj_type, _ in objects:
        return obj_type
    return 'unknown'
    
def analyze_plsql_objects(plsql_code):
    """分析PL/SQL代码，识别创建的对象"""
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

def execute_sql(database_conn_info, sql):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            # 设置查询超时
            cur.execute(f"SET statement_timeout = {2 * 1000};")  # timeout单位为毫秒
            cur.execute(sql)

def check_plsql_executability(generated_plsql, call_plsqls, database_name):
    execution_error = None
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name} port={port}"""
        restore_databases(conn_info, host, port, user, password, [database_name])
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                # 设置查询超时
                cur.execute(f"SET statement_timeout = {2 * 1000};")  # timeout单位为毫秒
                cur.execute(generated_plsql)
                for call in call_plsqls:
                    cur.execute(call)
    except errors.Error as e:  # 捕获PostgreSQL特定的错误
        execution_error = str(e.sqlstate) + ":" + str(e)
    except Exception as e:
        execution_error = str(e)
    
    return execution_error

def fetch_query_results(database_conn_info, query):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            return result


def will_change_data(database_name, plsql_code, call_plsqls, include_system_tables=False):
    """
    判断执行PL/SQL代码和调用列表中的SQL时是否会改变数据库数据
    
    Args:
        database_name: 数据库名称
        plsql_code: 要检查的PL/SQL代码
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否检查系统表的变化
    
    Returns:
        dict: 包含详细变化信息的结果字典
    """
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name} port={port}"""
        
        # 获取所有用户表
        all_user_tables = get_all_user_tables(database_name)
        print(f"Monitoring {len(all_user_tables)} user tables for changes")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Monitoring {len(important_system_tables)} system tables for changes")
        
        # 执行前备份所有表的数据
        before_execution_data = {}
        
        # 备份用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                before_execution_data[table] = pd.DataFrame(result) if result is not None else None
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table} before execution: {e}")
                before_execution_data[table] = None
        
        # 备份系统表数据
        system_tables_before = {}
        for sys_table in important_system_tables:
            system_tables_before[sys_table] = fetch_system_table_data(database_conn_info, sys_table)
        
        # 执行PL/SQL代码和调用语句
        execute_sql(database_conn_info, plsql_code)
        print("Executed PL/SQL code")
        for call in call_plsqls:
            execute_sql(database_conn_info, call)
        
        # 执行后获取所有表的数据
        after_execution_data = {}
        
        # 获取执行后的用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                after_execution_data[table] = pd.DataFrame(result) if result is not None else None
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table} after execution: {e}")
                after_execution_data[table] = None
        
        # 获取执行后的系统表数据
        system_tables_after = {}
        for sys_table in important_system_tables:
            system_tables_after[sys_table] = fetch_system_table_data(database_conn_info, sys_table)
        
        # 比较数据变化
        changed_user_tables = []
        changed_system_tables = []
        
        # 比较用户表数据
        for table in all_user_tables:
            data_before = before_execution_data.get(table)
            data_after = after_execution_data.get(table)
            
            if data_before is None and data_after is None:
                continue
            elif data_before is None or data_after is None:
                changed_user_tables.append(table)
            elif not data_before.equals(data_after):
                changed_user_tables.append(table)
        
        # 比较系统表数据
        if include_system_tables:
            for sys_table in important_system_tables:
                data_before = system_tables_before.get(sys_table)
                data_after = system_tables_after.get(sys_table)
                
                if data_before is None and data_after is None:
                    continue
                elif data_before is None or data_after is None:
                    changed_system_tables.append(sys_table)
                elif data_before != data_after:
                    changed_system_tables.append(sys_table)
        
        # 判断是否有数据变化
        has_data_changes = len(changed_user_tables) > 0 or len(changed_system_tables) > 0
        
        result = {
            'will_change_data': has_data_changes,
            'changed_user_tables': changed_user_tables,
            'changed_system_tables': changed_system_tables,
            'total_user_tables_monitored': len(all_user_tables),
            'total_system_tables_monitored': len(important_system_tables),
            'user_tables_changed_count': len(changed_user_tables),
            'system_tables_changed_count': len(changed_system_tables),
            'changes_detailed': {
                'user_tables': changed_user_tables,
                'system_tables': changed_system_tables
            }
        }
        
        print(f"Data change analysis completed:")
        print(f"Will change data: {has_data_changes}")
        print(f"User tables changed: {len(changed_user_tables)}/{len(all_user_tables)}")
        print(f"System tables changed: {len(changed_system_tables)}/{len(important_system_tables)}")
        
        if changed_user_tables:
            print(f"Changed user tables: {changed_user_tables}")
        if changed_system_tables:
            print(f"Changed system tables: {changed_system_tables}")
        
        return result
    
    except Exception as e:
        print(f"Error in will_change_data: {e}")
        return {
            'will_change_data': None,  # 表示无法确定
            'error': str(e),
            'changed_user_tables': [],
            'changed_system_tables': [],
            'total_user_tables_monitored': 0,
            'total_system_tables_monitored': 0,
            'user_tables_changed_count': 0,
            'system_tables_changed_count': 0
        }


# 简化版本：只返回布尔值
def will_change_data_simple(database_name, plsql_code, call_plsqls, include_system_tables=False):
    """
    简化版本：只返回是否会改变数据的布尔值
    
    Returns:
        bool: True表示会改变数据，False表示不会改变数据，None表示检查出错
    """
    try:
        result = will_change_data(database_name, plsql_code, call_plsqls, include_system_tables)
        return result.get('will_change_data')
    except Exception as e:
        print(f"Error in will_change_data_simple: {e}")
        return None

def compare_plsql(database_name, plsql1, plsql2, call_plsqls, include_system_tables):
    """
    比较两个PL/SQL代码的执行结果
    
    Args:
        database_name: 数据库名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否包含系统表比较
    
    Returns:
        True or False
    """
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name}"""
        
        # 获取所有用户表
        all_user_tables = get_all_user_tables(database_name)
        print(f"Found {len(all_user_tables)} user tables to compare: {all_user_tables}")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Will compare {len(important_system_tables)} system tables")
        
        # 第一次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql1)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        # 收集第一次执行后的数据
        user_tables_results1 = {}
        system_tables_results1 = {}
        
        # 获取所有用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                user_tables_results1[table] = pd.DataFrame(result)
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table}: {e}")
                user_tables_results1[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results1[sys_table] = fetch_system_table_data(database_conn_info, sys_table)
        
        # 第二次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql2)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        # 收集第二次执行后的数据
        user_tables_results2 = {}
        system_tables_results2 = {}
        
        # 获取所有用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                user_tables_results2[table] = pd.DataFrame(result)
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table}: {e}")
                user_tables_results2[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results2[sys_table] = fetch_system_table_data(database_conn_info, sys_table)

        # 比较用户表数据
        user_tables_same = True
        user_tables_diff = []
        
        for table in all_user_tables:
            df1 = user_tables_results1.get(table)
            df2 = user_tables_results2.get(table)
            
            if df1 is None and df2 is None:
                continue
            elif df1 is None or df2 is None:
                user_tables_same = False
                user_tables_diff.append(table)
            elif not df1.equals(df2):
                user_tables_same = False
                user_tables_diff.append(table)
        
        # 比较系统表数据
        system_tables_same = True
        system_tables_diff = []
        
        if include_system_tables:
            for sys_table in important_system_tables:
                result1 = system_tables_results1.get(sys_table)
                result2 = system_tables_results2.get(sys_table)
                
                if result1 is None and result2 is None:
                    continue
                elif result1 is None or result2 is None:
                    system_tables_same = False
                    system_tables_diff.append(sys_table)
                elif result1 != result2:
                    system_tables_same = False
                    system_tables_diff.append(sys_table)
        
        # 综合结果
        overall_same = user_tables_same and system_tables_same
        
        result = {
            'overall_same': overall_same,
            'user_tables_same': user_tables_same,
            'system_tables_same': system_tables_same,
            'user_tables_compared': len(all_user_tables),
            'system_tables_compared': len(important_system_tables),
            'user_tables_diff': user_tables_diff,
            'system_tables_diff': system_tables_diff
        }

        print(result)
        
        return result.get('overall_same', False)
    
    except Exception as e:
        print(f"Error in compare_plsql: {e}")
        return False


def compare_plsql_function(database_name, plsql1, plsql2, call_plsqls):
    """
    比较两个PL/SQL函数代码的执行结果
    
    Args:
        database_name: 数据库名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
    
    Returns:
        True or False
    """
    try:
        database_conn_info = f"""host={host} user={user} password={password} dbname={database_name}"""

        # 第一次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql1)

        # 收集第一次执行的结果
        function_results1 = {}
        for i, call_plsql in enumerate(call_plsqls):
            try:
                result = fetch_query_results(database_conn_info, call_plsql)
                function_results1[i] = {
                    'sql': call_plsql,
                    'result': pd.DataFrame(result)
                }
            except Exception as e:
                print(f"Warning: Could not execute call statement {i}: {call_plsql}")
                print(f"Error: {e}")
                function_results1[i] = {
                    'sql': call_plsql,
                    'result': None,
                    'error': str(e)
                }
        
        # 第二次执行
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
                print(f"Warning: Could not execute call statement {i}: {call_plsql}")
                print(f"Error: {e}")
                function_results2[i] = {
                    'sql': call_plsql,
                    'result': None,
                    'error': str(e)
                }

        function_same = True
        function_diff = []

        for i in range(len(call_plsqls)):
            res1 = function_results1[i]
            res2 = function_results2[i]

            # 处理结果都为None的情况
            if res1.get('result') is None and res2.get('result') is None:
                continue

            # 处理一个结果为None的情况
            if res1.get('result') is None or res2.get('result') is None:
                function_same = False
                res1_is_none = res1.get('result') is None
                res2_is_none = res2.get('result') is None
                
                if res1_is_none and not res2_is_none:
                    reason = 'res1 is None, res2 is not None'
                elif res2_is_none and not res1_is_none:
                    reason = 'res2 is None, res1 is not None'
                else:
                    reason = 'One result is None'
                
                print(f"Function result difference at index {i}: {reason}")
                function_diff.append({
                    'index': i,
                    'sql': call_plsqls[i],
                    'reason': reason,
                    'result1_is_none': res1_is_none,
                    'result2_is_none': res2_is_none
                })
                continue

            # 比较DataFrame
            try:
                if not res1.get('result').equals(res2.get('result')):
                    function_same = False
                    function_diff.append({
                        'index': i,
                        'sql': call_plsqls[i],
                        'reason': 'DataFrame comparison failed',
                        'result1_shape': res1.get('result').shape,
                        'result2_shape': res2.get('result').shape
                    })
            except Exception as e:
                function_same = False
                function_diff.append({
                    'index': i,
                    'sql': call_plsqls[i],
                    'reason': f'Comparison error: {str(e)}'
                })
        
        # 输出详细比较结果
        print(f"Function comparison result: {function_same}")
        if function_diff:
            print(f"Differences found in {len(function_diff)} call statements:")
            for diff in function_diff:
                print(f"  Index {diff['index']}: {diff['sql'][:100]}...")
                print(f"    Reason: {diff['reason']}")
                if 'result1_is_none' in diff:
                    print(f"    result1_is_none: {diff['result1_is_none']}, result2_is_none: {diff['result2_is_none']}")
                if 'result1_shape' in diff:
                    print(f"    result1_shape: {diff['result1_shape']}, result2_shape: {diff['result2_shape']}")
        
        return function_same
    except Exception as e:
        print("Error in compare_plsql_function")
        print(e)
        return False
    
"""
PostgreSQL PL/pgSQL Semantic Equivalence Checker

This module provides tools for comparing two PL/pgSQL code blocks to determine if they are
semantically equivalent, even when they differ in:
- Whitespace and formatting
- Variable names (identifiers)
- Parameter names
- Cursor names  
- Code structure spacing

The tool uses a hybrid approach:
1. Text preprocessing to normalize syntax and formatting
2. Abstract Syntax Tree (AST) parsing using sqlparse
3. Semantic comparison of normalized AST structures

Supports:
- Procedures (CREATE OR REPLACE PROCEDURE)
- Functions (CREATE OR REPLACE FUNCTION)
- Triggers (CREATE FUNCTION + CREATE TRIGGER - two statements)

Usage:
    from plpgsql_semantic_checker import is_plpgsql_semantically_equivalent
    
    result = is_plpgsql_semantically_equivalent(code1, code2)
    
Integration:
    Replace the existing code in postgres_util.py from line 1082 onwards with this code.
    The main entry points are:
    - is_exact_match(plpgsql1, plpgsql2) -> bool
    - debug_semantic_equivalence(plpgsql1, plpgsql2) -> bool (with debug output)
    - get_plsql_type(plsql_code) -> PLpgSQLType
"""


class PLpgSQLType(Enum):
    """Enumeration of PL/pgSQL object types"""
    PROCEDURE = "procedure"
    FUNCTION = "function"
    TRIGGER = "trigger"
    UNKNOWN = "unknown"


@dataclass
class ASTNode:
    """Abstract Syntax Tree Node for PL/pgSQL"""
    node_type: str
    value: Optional[str] = None
    children: List['ASTNode'] = field(default_factory=list)
    
    def __repr__(self):
        if self.value and not self.children:
            return f"{self.node_type}({self.value})"
        elif self.children:
            children_str = ', '.join(str(child) for child in self.children)
            if self.value:
                return f"{self.node_type}({self.value})[{children_str}]"
            else:
                return f"{self.node_type}[{children_str}]"
        else:
            return self.node_type
    
    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        return (self.node_type == other.node_type and 
                self.value == other.value and 
                self.children == other.children)
    
    def to_dict(self) -> dict:
        """Convert AST node to dictionary for debugging"""
        return {
            'type': self.node_type,
            'value': self.value,
            'children': [child.to_dict() for child in self.children]
        }


def get_plsql_type(plsql_code: str) -> PLpgSQLType:
    """
    Determine the type of PL/pgSQL code (procedure, function, or trigger)
    
    For triggers, there will be both a FUNCTION and a TRIGGER creation.
    
    Args:
        plsql_code: The PL/pgSQL code to analyze
        
    Returns:
        PLpgSQLType enum value indicating the type
    """
    code_upper = plsql_code.upper()
    
    # Check for trigger first (has both CREATE TRIGGER and CREATE FUNCTION)
    has_trigger = bool(re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+', code_upper))
    has_function = bool(re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+', code_upper))
    has_procedure = bool(re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+', code_upper))
    
    if has_trigger:
        return PLpgSQLType.TRIGGER
    elif has_procedure:
        return PLpgSQLType.PROCEDURE
    elif has_function:
        return PLpgSQLType.FUNCTION
    else:
        return PLpgSQLType.UNKNOWN


class IdentifierMapper:
    """
    Maps user-defined identifiers to abstract placeholders while preserving
    semantic structure. Different identifier categories get different placeholder prefixes.
    """
    
    def __init__(self):
        self.param_map: Dict[str, str] = {}
        self.var_map: Dict[str, str] = {}
        self.cursor_map: Dict[str, str] = {}
        self.label_map: Dict[str, str] = {}
        self.counter = {'param': 0, 'var': 0, 'cursor': 0, 'label': 0}
        
        # System objects that should NOT be abstracted
        self.system_objects = {
            'FOUND', 'RECORD', 'SQLSTATE', 'SQLERRM', 'ROW_COUNT',
            'CURRENT_USER', 'SESSION_USER', 'CURRENT_TIMESTAMP', 'NOW',
            'TG_NAME', 'TG_WHEN', 'TG_LEVEL', 'TG_OP', 'TG_RELID',
            'TG_TABLE_NAME', 'TG_TABLE_SCHEMA', 'TG_NARGS', 'TG_ARGV',
            'NEW', 'OLD', 'NULL', 'TRUE', 'FALSE', 'STRICT'
        }
        
        # PL/pgSQL keywords that should not be abstracted
        self.keywords = {
            'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'TRIGGER',
            'LANGUAGE', 'PLPGSQL', 'AS', 'DECLARE', 'BEGIN', 'END',
            'IF', 'THEN', 'ELSE', 'ELSIF', 'ELSEIF', 'CASE', 'WHEN',
            'WHILE', 'FOR', 'LOOP', 'EXIT', 'CONTINUE', 'RETURN', 'RETURNS',
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'UPDATE', 'SET',
            'DELETE', 'VALUES', 'AND', 'OR', 'NOT', 'IN', 'EXISTS',
            'CURSOR', 'OPEN', 'CLOSE', 'FETCH', 'NEXT', 'PRIOR',
            'EXCEPTION', 'RAISE', 'NOTICE', 'WARNING', 'ERROR',
            'COMMIT', 'ROLLBACK', 'PERFORM', 'EXECUTE', 'USING',
            'VOLATILE', 'STABLE', 'IMMUTABLE', 'SECURITY', 'DEFINER', 'INVOKER',
            'AFTER', 'BEFORE', 'INSTEAD', 'OF', 'ON', 'EACH', 'ROW', 'STATEMENT',
            'EXECUTE', 'PROCEDURE', 'FUNCTION', 'FOR', 'REFERENCES',
            'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'NATURAL',
            'GROUP', 'BY', 'ORDER', 'ASC', 'DESC', 'LIMIT', 'OFFSET',
            'HAVING', 'UNION', 'INTERSECT', 'EXCEPT', 'ALL', 'DISTINCT',
            'IS', 'BETWEEN', 'LIKE', 'ILIKE', 'SIMILAR', 'TO',
            'CURRENT', 'OF', 'CONCAT', 'COALESCE', 'NULLIF',
            'CAST', 'ARRAY', 'ANY', 'SOME', 'DEFAULT',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'LENGTH', 'TRIM',
            'UPPER', 'LOWER', 'SUBSTRING', 'POSITION', 'OVERLAY',
            'MOD', 'ABS', 'CEIL', 'FLOOR', 'ROUND', 'TRUNC',
            'REVERSE', 'GREATEST', 'LEAST',
            # Data types
            'TEXT', 'VARCHAR', 'CHAR', 'CHARACTER', 'INTEGER', 'INT', 'INT4',
            'BIGINT', 'INT8', 'SMALLINT', 'INT2', 'DECIMAL', 'NUMERIC',
            'REAL', 'FLOAT4', 'DOUBLE', 'PRECISION', 'FLOAT8',
            'BOOLEAN', 'BOOL', 'DATE', 'TIME', 'TIMESTAMP', 'TIMESTAMPTZ',
            'INTERVAL', 'UUID', 'JSON', 'JSONB', 'BYTEA',
            'SERIAL', 'BIGSERIAL', 'SMALLSERIAL', 'MONEY',
            'VARYING', 'ZONE', 'WITH', 'WITHOUT',
            '%TYPE', '%ROWTYPE', 'ROWTYPE', 'TYPE',
            # Additional keywords
            'GET', 'DIAGNOSTICS', 'STRICT', 'SCROLL', 'NO',
            'FOREACH', 'SLICE', 'REVERSE', 'ALIAS', 'CONSTANT',
            'COLLATE', 'CONSTRAINT', 'CHECK', 'UNIQUE', 'PRIMARY', 'KEY',
            'FOREIGN', 'DEFERRABLE', 'INITIALLY', 'DEFERRED', 'IMMEDIATE'
        }
    
    def is_system_identifier(self, name: str) -> bool:
        """Check if identifier is a system object or keyword"""
        upper_name = name.upper().strip('"')
        return upper_name in self.system_objects or upper_name in self.keywords
    
    def map_parameter(self, name: str) -> str:
        """Map a parameter name to abstract placeholder"""
        if self.is_system_identifier(name):
            return name.upper()
        lower_name = name.lower().strip('"')
        if lower_name not in self.param_map:
            self.param_map[lower_name] = f"<PARAM_{self.counter['param']}>"
            self.counter['param'] += 1
        return self.param_map[lower_name]
    
    def map_variable(self, name: str) -> str:
        """Map a variable name to abstract placeholder"""
        if self.is_system_identifier(name):
            return name.upper()
        lower_name = name.lower().strip('"')
        if lower_name not in self.var_map:
            self.var_map[lower_name] = f"<VAR_{self.counter['var']}>"
            self.counter['var'] += 1
        return self.var_map[lower_name]
    
    def map_cursor(self, name: str) -> str:
        """Map a cursor name to abstract placeholder"""
        if self.is_system_identifier(name):
            return name.upper()
        lower_name = name.lower().strip('"')
        if lower_name not in self.cursor_map:
            self.cursor_map[lower_name] = f"<CURSOR_{self.counter['cursor']}>"
            self.counter['cursor'] += 1
        return self.cursor_map[lower_name]
    
    def map_label(self, name: str) -> str:
        """Map a label name to abstract placeholder"""
        if self.is_system_identifier(name):
            return name.upper()
        lower_name = name.lower().strip('"')
        if lower_name not in self.label_map:
            self.label_map[lower_name] = f"<LABEL_{self.counter['label']}>"
            self.counter['label'] += 1
        return self.label_map[lower_name]
    
    def get_mapped(self, name: str) -> Optional[str]:
        """Get mapping for a name if it exists in any category"""
        lower_name = name.lower().strip('"')
        if lower_name in self.param_map:
            return self.param_map[lower_name]
        if lower_name in self.var_map:
            return self.var_map[lower_name]
        if lower_name in self.cursor_map:
            return self.cursor_map[lower_name]
        if lower_name in self.label_map:
            return self.label_map[lower_name]
        return None


class PLpgSQLNormalizer:
    """
    Normalizes PL/pgSQL code by:
    1. Extracting and parsing the structure
    2. Abstracting user-defined identifiers
    3. Normalizing whitespace and formatting
    """
    
    def __init__(self):
        self.mapper = IdentifierMapper()
        self.declared_vars: Set[str] = set()
        self.declared_cursors: Set[str] = set()
        self.declared_params: Set[str] = set()
    
    def extract_function_body(self, code: str) -> Tuple[str, str, str]:
        """
        Extract the header, body, and footer from a PL/pgSQL function/procedure.
        
        Returns:
            (header, body, footer) - e.g., ("CREATE ... AS ", body_content, " LANGUAGE plpgsql;")
        """
        # Find $$ delimited body - handle both AS $$ and $$ LANGUAGE positions
        # Pattern 1: AS $$ ... $$ (LANGUAGE at end)
        dollar_match = re.search(r'AS\s*\$\$(.+?)\$\$', code, re.IGNORECASE | re.DOTALL)
        if dollar_match:
            body = dollar_match.group(1)
            start_pos = code.index('AS') if 'AS' in code.upper() else dollar_match.start()
            # Find AS position
            as_match = re.search(r'\bAS\b', code, re.IGNORECASE)
            if as_match:
                header = code[:as_match.end()].strip()
            else:
                header = code[:dollar_match.start()].strip()
            end_pos = dollar_match.end()
            footer = code[end_pos:].strip()
            return header, body.strip(), footer
        
        # Try $tag$ delimiter
        tag_match = re.search(r'AS\s*\$(\w+)\$(.+?)\$\1\$', code, re.IGNORECASE | re.DOTALL)
        if tag_match:
            tag = tag_match.group(1)
            body = tag_match.group(2)
            as_match = re.search(r'\bAS\b', code, re.IGNORECASE)
            if as_match:
                header = code[:as_match.end()].strip()
            else:
                header = code[:tag_match.start()].strip()
            end_pos = tag_match.end()
            footer = code[end_pos:].strip()
            return header, body.strip(), footer
        
        return code, "", ""
    
    def extract_parameters(self, header: str) -> List[Tuple[str, str]]:
        """Extract parameters from function/procedure header"""
        params = []
        
        # Match parameter list in parentheses
        paren_match = re.search(r'\(\s*([^)]*)\s*\)', header)
        if not paren_match:
            return params
        
        param_str = paren_match.group(1).strip()
        if not param_str:
            return params
        
        # Split by comma, handling nested parentheses
        param_parts = self._split_params(param_str)
        
        for part in param_parts:
            part = part.strip()
            if not part:
                continue
            
            # Remove IN/OUT/INOUT modifiers
            part = re.sub(r'\b(IN|OUT|INOUT)\b\s*', '', part, flags=re.IGNORECASE).strip()
            
            # Extract parameter name and type
            tokens = part.split()
            if len(tokens) >= 2:
                param_name = tokens[0]
                param_type = ' '.join(tokens[1:])
                params.append((param_name, param_type))
                self.declared_params.add(param_name.lower().strip('"'))
        
        return params
    
    def _split_params(self, param_str: str) -> List[str]:
        """Split parameter string by comma, respecting parentheses"""
        params = []
        current = ""
        depth = 0
        
        for char in param_str:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                params.append(current.strip())
                current = ""
                continue
            current += char
        
        if current.strip():
            params.append(current.strip())
        
        return params
    
    def extract_declarations(self, body: str) -> List[Tuple[str, str, str]]:
        """
        Extract DECLARE section declarations.
        Returns list of (type, name, definition)
        type can be: 'variable', 'cursor', 'record', 'alias'
        """
        declarations = []
        
        # Find DECLARE section
        declare_match = re.search(r'\bDECLARE\b(.+?)\bBEGIN\b', body, re.IGNORECASE | re.DOTALL)
        if not declare_match:
            return declarations
        
        declare_section = declare_match.group(1)
        
        # Split by semicolon to get individual declarations
        decl_parts = declare_section.split(';')
        
        for part in decl_parts:
            part = part.strip()
            if not part:
                continue
            
            # Check for cursor declaration
            cursor_match = re.match(
                r'(\w+)\s+CURSOR\s+(?:(?:\(\s*[^)]*\s*\)\s+)?FOR\s+)?(.+)',
                part, re.IGNORECASE | re.DOTALL
            )
            if cursor_match:
                cursor_name = cursor_match.group(1)
                cursor_def = cursor_match.group(2)
                declarations.append(('cursor', cursor_name, cursor_def))
                self.declared_cursors.add(cursor_name.lower())
                continue
            
            # Check for record/rowtype
            record_match = re.match(r'(\w+)\s+(RECORD|%ROWTYPE|\w+%ROWTYPE)', part, re.IGNORECASE)
            if record_match:
                var_name = record_match.group(1)
                var_type = record_match.group(2)
                declarations.append(('record', var_name, var_type))
                self.declared_vars.add(var_name.lower())
                continue
            
            # Check for alias
            alias_match = re.match(r'(\w+)\s+ALIAS\s+FOR\s+(.+)', part, re.IGNORECASE)
            if alias_match:
                alias_name = alias_match.group(1)
                alias_target = alias_match.group(2)
                declarations.append(('alias', alias_name, alias_target))
                self.declared_vars.add(alias_name.lower())
                continue
            
            # Regular variable declaration
            var_match = re.match(r'(\w+)\s+(.+)', part, re.IGNORECASE)
            if var_match:
                var_name = var_match.group(1)
                var_def = var_match.group(2)
                # Skip if it's a keyword
                if not self.mapper.is_system_identifier(var_name):
                    declarations.append(('variable', var_name, var_def))
                    self.declared_vars.add(var_name.lower())
        
        return declarations
    
    def normalize_header(self, header: str) -> str:
        """Normalize the function/procedure header"""
        # Normalize whitespace
        header = re.sub(r'\s+', ' ', header).strip()
        
        # Normalize CREATE OR REPLACE
        header = re.sub(r'CREATE\s+OR\s+REPLACE', 'CREATE OR REPLACE', header, flags=re.IGNORECASE)
        
        # Extract and normalize parameters
        params = self.extract_parameters(header)
        
        # Rebuild parameter list with abstracted names
        if params:
            param_strs = []
            for name, ptype in params:
                abstract_name = self.mapper.map_parameter(name)
                # Normalize type
                ptype_normalized = self._normalize_type(ptype)
                param_strs.append(f"{abstract_name} {ptype_normalized}")
            
            # Replace original parameter list
            new_param_list = ', '.join(param_strs)
            header = re.sub(r'\([^)]*\)', f'({new_param_list})', header, count=1)
        
        # Normalize function/procedure name to abstract placeholder
        header = re.sub(
            r'(PROCEDURE|FUNCTION)\s+(\w+)',
            lambda m: f"{m.group(1).upper()} <PROC_NAME>",
            header,
            flags=re.IGNORECASE
        )
        
        # Normalize RETURNS clause
        header = re.sub(r'RETURNS\s+(\w+)', lambda m: f"RETURNS {self._normalize_type(m.group(1))}", 
                       header, flags=re.IGNORECASE)
        
        # Normalize TRIGGER return type
        header = re.sub(r'RETURNS\s+TRIGGER', 'RETURNS TRIGGER', header, flags=re.IGNORECASE)
        
        return header.upper()
    
    def _normalize_type(self, type_str: str) -> str:
        """Normalize a PostgreSQL data type"""
        type_str = type_str.strip().upper()
        
        # Handle common aliases
        type_aliases = {
            'INT': 'INTEGER',
            'INT4': 'INTEGER',
            'INT8': 'BIGINT',
            'INT2': 'SMALLINT',
            'FLOAT4': 'REAL',
            'FLOAT8': 'DOUBLE PRECISION',
            'BOOL': 'BOOLEAN',
            'VARCHAR': 'CHARACTER VARYING',
            'CHAR': 'CHARACTER',
        }
        
        for alias, canonical in type_aliases.items():
            if type_str == alias:
                return canonical
        
        return type_str
    
    def normalize_body(self, body: str) -> str:
        """Normalize the function body"""
        # First extract declarations to build identifier mappings
        declarations = self.extract_declarations(body)
        
        # Register declared identifiers
        for decl_type, name, definition in declarations:
            if decl_type == 'cursor':
                self.mapper.map_cursor(name)
            elif decl_type in ('variable', 'record', 'alias'):
                self.mapper.map_variable(name)
        
        # Normalize whitespace
        body = re.sub(r'\s+', ' ', body).strip()
        
        # Normalize DECLARE section
        body = self._normalize_declare_section(body)
        
        # Normalize statements in BEGIN...END block
        body = self._normalize_statements(body)
        
        return body
    
    def _normalize_declare_section(self, body: str) -> str:
        """Normalize the DECLARE section"""
        declare_match = re.search(r'\bDECLARE\b(.+?)\bBEGIN\b', body, re.IGNORECASE | re.DOTALL)
        if not declare_match:
            return body
        
        declare_content = declare_match.group(1)
        normalized_decls = []
        
        # Process each declaration
        decl_parts = declare_content.split(';')
        for part in decl_parts:
            part = part.strip()
            if not part:
                continue
            
            normalized = self._normalize_single_declaration(part)
            if normalized:
                normalized_decls.append(normalized)
        
        # Rebuild DECLARE section
        new_declare = 'DECLARE ' + '; '.join(normalized_decls) + ';'
        
        # Replace in body
        body = re.sub(
            r'\bDECLARE\b.+?\bBEGIN\b',
            new_declare + ' BEGIN',
            body,
            flags=re.IGNORECASE | re.DOTALL
        )
        
        return body
    
    def _normalize_single_declaration(self, decl: str) -> Optional[str]:
        """Normalize a single declaration"""
        decl = decl.strip()
        if not decl:
            return None
        
        # Cursor declaration
        cursor_match = re.match(
            r'(\w+)\s+CURSOR\s+(.+)',
            decl, re.IGNORECASE | re.DOTALL
        )
        if cursor_match:
            cursor_name = cursor_match.group(1)
            cursor_rest = cursor_match.group(2)
            abstract_name = self.mapper.map_cursor(cursor_name)
            # Normalize the cursor query
            normalized_rest = self._normalize_expression(cursor_rest)
            return f"{abstract_name} CURSOR {normalized_rest}"
        
        # Record type
        record_match = re.match(r'(\w+)\s+(RECORD)', decl, re.IGNORECASE)
        if record_match:
            var_name = record_match.group(1)
            abstract_name = self.mapper.map_variable(var_name)
            return f"{abstract_name} RECORD"
        
        # %ROWTYPE
        rowtype_match = re.match(r'(\w+)\s+(\w+)(%ROWTYPE)', decl, re.IGNORECASE)
        if rowtype_match:
            var_name = rowtype_match.group(1)
            table_name = rowtype_match.group(2)
            abstract_name = self.mapper.map_variable(var_name)
            return f"{abstract_name} {table_name}%ROWTYPE"
        
        # %TYPE
        type_match = re.match(r'(\w+)\s+(.+)(%TYPE)', decl, re.IGNORECASE)
        if type_match:
            var_name = type_match.group(1)
            ref = type_match.group(2)
            abstract_name = self.mapper.map_variable(var_name)
            return f"{abstract_name} {ref}%TYPE"
        
        # Regular variable
        var_match = re.match(r'(\w+)\s+(.+)', decl, re.IGNORECASE)
        if var_match:
            var_name = var_match.group(1)
            var_type = var_match.group(2)
            if not self.mapper.is_system_identifier(var_name):
                abstract_name = self.mapper.map_variable(var_name)
                normalized_type = self._normalize_type(var_type.split(':=')[0].split('DEFAULT')[0].strip())
                # Handle default value
                default_match = re.search(r'(:=|DEFAULT)\s*(.+)', var_type, re.IGNORECASE)
                if default_match:
                    default_val = self._normalize_expression(default_match.group(2))
                    return f"{abstract_name} {normalized_type} := {default_val}"
                return f"{abstract_name} {normalized_type}"
        
        return decl
    
    def _normalize_statements(self, body: str) -> str:
        """Normalize statements in the body"""
        # Extract BEGIN...END block
        begin_match = re.search(r'\bBEGIN\b(.+)\bEND\b', body, re.IGNORECASE | re.DOTALL)
        if not begin_match:
            return body
        
        stmt_content = begin_match.group(1)
        normalized = self._normalize_expression(stmt_content)
        
        # Rebuild body
        prefix = body[:begin_match.start()] + 'BEGIN '
        suffix = ' END' + body[begin_match.end():]
        
        return prefix + normalized + suffix
    
    def _normalize_expression(self, expr: str) -> str:
        """Normalize an expression, abstracting identifiers"""
        if not expr:
            return expr
        
        # Normalize whitespace
        expr = re.sub(r'\s+', ' ', expr).strip()
        
        # Replace known variables, cursors, and parameters
        # Use word boundaries to avoid partial matches
        
        # First, protect quoted identifiers by replacing them with placeholders
        quoted_identifiers = {}
        quoted_counter = [0]
        
        def save_quoted(match):
            name = match.group(1)
            placeholder = f"__QUOTED_{quoted_counter[0]}__"
            quoted_identifiers[placeholder] = f'"{name}"'
            quoted_counter[0] += 1
            return placeholder
        
        expr = re.sub(r'"([^"]+)"', save_quoted, expr)
        
        # Replace identifiers with their abstract mappings
        def replace_identifier(match):
            name = match.group(0)
            
            # Don't replace our placeholders
            if name.startswith('__QUOTED_') and name.endswith('__'):
                return name
            
            # Check if it's a system identifier or keyword
            if self.mapper.is_system_identifier(name):
                return name.upper()
            
            # Check if we have a mapping
            mapped = self.mapper.get_mapped(name)
            if mapped:
                return mapped
            
            # Keep unmapped identifiers (likely table/column names)
            return name
        
        # Replace identifiers (word characters not starting with digit)
        expr = re.sub(r'\b(?![0-9])(\w+)\b', replace_identifier, expr)
        
        # Restore quoted identifiers
        for placeholder, original in quoted_identifiers.items():
            expr = expr.replace(placeholder, original)
        
        # Normalize operators
        expr = re.sub(r'\s*:=\s*', ' := ', expr)
        expr = re.sub(r'\s*=\s*', ' = ', expr)
        expr = re.sub(r'\s*<>\s*', ' <> ', expr)
        expr = re.sub(r'\s*!=\s*', ' <> ', expr)  # Normalize != to <>
        expr = re.sub(r'\s*>=\s*', ' >= ', expr)
        expr = re.sub(r'\s*<=\s*', ' <= ', expr)
        expr = re.sub(r'\s*>\s*', ' > ', expr)
        expr = re.sub(r'\s*<\s*', ' < ', expr)
        
        # Normalize punctuation
        expr = re.sub(r'\s*;\s*', '; ', expr)
        expr = re.sub(r'\s*,\s*', ', ', expr)
        expr = re.sub(r'\s*\(\s*', '(', expr)
        expr = re.sub(r'\s*\)\s*', ')', expr)
        
        # Normalize keywords
        keywords = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE',
            'VALUES', 'AND', 'OR', 'NOT', 'NULL', 'IF', 'THEN', 'ELSE', 'ELSIF',
            'END IF', 'LOOP', 'END LOOP', 'EXIT', 'WHEN', 'FOUND', 'OPEN', 'CLOSE',
            'FETCH', 'RETURN', 'RAISE', 'NOTICE', 'EXCEPTION', 'CURRENT', 'OF',
            'FOR', 'IN', 'PERFORM', 'EXECUTE', 'USING', 'IS', 'DISTINCT',
            'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'AS',
            'ORDER', 'BY', 'ASC', 'DESC', 'LIMIT', 'OFFSET', 'GROUP', 'HAVING',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CONCAT', 'LENGTH', 'TRIM',
            'MOD', 'REVERSE', 'CASE', 'COALESCE', 'NULLIF'
        ]
        
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            expr = re.sub(pattern, kw, expr, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        expr = re.sub(r'\s+', ' ', expr).strip()
        
        return expr
    
    def normalize_footer(self, footer: str) -> str:
        """Normalize the footer section"""
        footer = re.sub(r'\s+', ' ', footer).strip()
        
        # Normalize LANGUAGE clause
        footer = re.sub(r'LANGUAGE\s+plpgsql', 'LANGUAGE PLPGSQL', footer, flags=re.IGNORECASE)
        
        # Remove volatility markers (VOLATILE, STABLE, IMMUTABLE) as they're optional
        footer = re.sub(r'\b(VOLATILE|STABLE|IMMUTABLE)\b', '', footer, flags=re.IGNORECASE)
        
        # Remove SECURITY markers
        footer = re.sub(r'\bSECURITY\s+(DEFINER|INVOKER)\b', '', footer, flags=re.IGNORECASE)
        
        # Clean up
        footer = re.sub(r'\s+', ' ', footer).strip()
        
        return footer.upper()
    
    def normalize_trigger_statement(self, trigger_sql: str) -> str:
        """Normalize a CREATE TRIGGER statement"""
        trigger_sql = re.sub(r'\s+', ' ', trigger_sql).strip()
        
        # Abstract trigger name
        trigger_sql = re.sub(
            r'(CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER)\s+(\w+)',
            r'\1 <TRIGGER_NAME>',
            trigger_sql,
            flags=re.IGNORECASE
        )
        
        # Normalize timing (BEFORE/AFTER/INSTEAD OF)
        trigger_sql = re.sub(r'\bBEFORE\b', 'BEFORE', trigger_sql, flags=re.IGNORECASE)
        trigger_sql = re.sub(r'\bAFTER\b', 'AFTER', trigger_sql, flags=re.IGNORECASE)
        trigger_sql = re.sub(r'\bINSTEAD\s+OF\b', 'INSTEAD OF', trigger_sql, flags=re.IGNORECASE)
        
        # Normalize events (INSERT/UPDATE/DELETE)
        trigger_sql = re.sub(r'\bINSERT\b', 'INSERT', trigger_sql, flags=re.IGNORECASE)
        trigger_sql = re.sub(r'\bUPDATE\b', 'UPDATE', trigger_sql, flags=re.IGNORECASE)
        trigger_sql = re.sub(r'\bDELETE\b', 'DELETE', trigger_sql, flags=re.IGNORECASE)
        
        # Normalize FOR EACH ROW/STATEMENT
        trigger_sql = re.sub(r'FOR\s+EACH\s+ROW', 'FOR EACH ROW', trigger_sql, flags=re.IGNORECASE)
        trigger_sql = re.sub(r'FOR\s+EACH\s+STATEMENT', 'FOR EACH STATEMENT', trigger_sql, flags=re.IGNORECASE)
        
        # Normalize EXECUTE FUNCTION/PROCEDURE
        trigger_sql = re.sub(
            r'EXECUTE\s+(FUNCTION|PROCEDURE)\s+(\w+)',
            r'EXECUTE FUNCTION <FUNC_NAME>',
            trigger_sql,
            flags=re.IGNORECASE
        )
        
        return trigger_sql.upper()
    
    def normalize(self, code: str) -> str:
        """
        Main normalization method.
        Returns normalized code representation.
        """
        plsql_type = get_plsql_type(code)
        
        if plsql_type == PLpgSQLType.TRIGGER:
            return self._normalize_trigger(code)
        else:
            return self._normalize_function_or_procedure(code)
    
    def _normalize_function_or_procedure(self, code: str) -> str:
        """Normalize a function or procedure"""
        header, body, footer = self.extract_function_body(code)
        
        normalized_header = self.normalize_header(header)
        normalized_body = self.normalize_body(body) if body else ""
        normalized_footer = self.normalize_footer(footer)
        
        # Clean up the result
        result = f"{normalized_header} $$ {normalized_body} $$ {normalized_footer}"
        result = re.sub(r'\s+', ' ', result).strip()
        return result
    
    def _normalize_trigger(self, code: str) -> str:
        """Normalize a trigger (function + trigger statement)"""
        # Split into individual statements
        statements = self._split_statements(code)
        
        normalized_parts = []
        
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            
            if re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER', stmt, re.IGNORECASE):
                # Normalize trigger statement
                normalized_parts.append(self.normalize_trigger_statement(stmt))
            elif re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION', stmt, re.IGNORECASE):
                # Normalize trigger function
                normalized_parts.append(self._normalize_function_or_procedure(stmt))
        
        return ' ||| '.join(normalized_parts)
    
    def _split_statements(self, code: str) -> List[str]:
        """Split code into individual SQL statements"""
        statements = []
        current = ""
        in_dollar = False
        dollar_tag = ""
        
        i = 0
        while i < len(code):
            # Check for $$ or $tag$
            if code[i] == '$':
                # Look for tag
                j = i + 1
                while j < len(code) and (code[j].isalnum() or code[j] == '_'):
                    j += 1
                if j < len(code) and code[j] == '$':
                    tag = code[i:j+1]
                    if not in_dollar:
                        in_dollar = True
                        dollar_tag = tag
                    elif tag == dollar_tag:
                        in_dollar = False
                        dollar_tag = ""
                    current += tag
                    i = j + 1
                    continue
            
            current += code[i]
            
            # Check for statement end
            if code[i] == ';' and not in_dollar:
                statements.append(current.strip())
                current = ""
            
            i += 1
        
        if current.strip():
            statements.append(current.strip())
        
        return statements


class PLpgSQLASTBuilder:
    """
    Builds an Abstract Syntax Tree from normalized PL/pgSQL code.
    """
    
    def __init__(self):
        pass
    
    def build(self, normalized_code: str) -> ASTNode:
        """Build AST from normalized code"""
        # Handle trigger (multiple statements separated by |||)
        if ' ||| ' in normalized_code:
            parts = normalized_code.split(' ||| ')
            children = [self._build_statement(part.strip()) for part in parts if part.strip()]
            return ASTNode('TRIGGER_DEFINITION', children=children)
        else:
            return self._build_statement(normalized_code)
    
    def _build_statement(self, code: str) -> ASTNode:
        """Build AST for a single statement"""
        # Determine statement type
        if re.match(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE', code, re.IGNORECASE):
            return self._build_procedure(code)
        elif re.match(r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION', code, re.IGNORECASE):
            return self._build_function(code)
        elif re.match(r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER', code, re.IGNORECASE):
            return self._build_trigger_stmt(code)
        else:
            return ASTNode('UNKNOWN', value=code)
    
    def _build_procedure(self, code: str) -> ASTNode:
        """Build AST for procedure"""
        children = []
        
        # Extract parts using $$ delimiters
        header_match = re.search(r'^(.+?)\$\$\s*(.+?)\s*\$\$(.*)$', code, re.DOTALL)
        if header_match:
            header = header_match.group(1).strip()
            body = header_match.group(2).strip()
            footer = header_match.group(3).strip()
            
            children.append(self._build_header(header, 'PROCEDURE'))
            children.append(self._build_body(body))
            children.append(ASTNode('FOOTER', value=footer))
        else:
            children.append(ASTNode('RAW', value=code))
        
        return ASTNode('PROCEDURE', children=children)
    
    def _build_function(self, code: str) -> ASTNode:
        """Build AST for function"""
        children = []
        
        # Extract parts using $$ delimiters
        header_match = re.search(r'^(.+?)\$\$\s*(.+?)\s*\$\$(.*)$', code, re.DOTALL)
        if header_match:
            header = header_match.group(1).strip()
            body = header_match.group(2).strip()
            footer = header_match.group(3).strip()
            
            children.append(self._build_header(header, 'FUNCTION'))
            children.append(self._build_body(body))
            children.append(ASTNode('FOOTER', value=footer))
        else:
            children.append(ASTNode('RAW', value=code))
        
        return ASTNode('FUNCTION', children=children)
    
    def _build_trigger_stmt(self, code: str) -> ASTNode:
        """Build AST for trigger statement"""
        return ASTNode('TRIGGER_STATEMENT', value=code)
    
    def _build_header(self, header: str, obj_type: str) -> ASTNode:
        """Build AST for header"""
        children = []
        
        # Extract parameters
        param_match = re.search(r'\(([^)]*)\)', header)
        if param_match:
            params_str = param_match.group(1)
            if params_str.strip():
                params = [p.strip() for p in params_str.split(',')]
                param_nodes = [ASTNode('PARAMETER', value=p) for p in params]
                children.append(ASTNode('PARAMETERS', children=param_nodes))
        
        # Extract return type
        return_match = re.search(r'RETURNS\s+(\S+)', header, re.IGNORECASE)
        if return_match:
            children.append(ASTNode('RETURNS', value=return_match.group(1)))
        
        return ASTNode('HEADER', value=obj_type, children=children)
    
    def _build_body(self, body: str) -> ASTNode:
        """Build AST for body"""
        children = []
        
        # Extract DECLARE section
        declare_match = re.search(r'DECLARE\s+(.+?)\s+BEGIN', body, re.IGNORECASE | re.DOTALL)
        if declare_match:
            decl_content = declare_match.group(1)
            decl_nodes = self._parse_declarations(decl_content)
            if decl_nodes:
                children.append(ASTNode('DECLARE', children=decl_nodes))
        
        # Extract BEGIN...END section
        begin_match = re.search(r'BEGIN\s+(.+?)\s+END', body, re.IGNORECASE | re.DOTALL)
        if begin_match:
            stmt_content = begin_match.group(1)
            stmt_nodes = self._parse_statements(stmt_content)
            if stmt_nodes:
                children.append(ASTNode('BLOCK', children=stmt_nodes))
        
        return ASTNode('BODY', children=children)
    
    def _parse_declarations(self, decl_content: str) -> List[ASTNode]:
        """Parse declarations into AST nodes"""
        nodes = []
        decls = [d.strip() for d in decl_content.split(';') if d.strip()]
        
        for decl in decls:
            if 'CURSOR' in decl.upper():
                nodes.append(ASTNode('CURSOR_DECL', value=decl))
            elif 'RECORD' in decl.upper():
                nodes.append(ASTNode('RECORD_DECL', value=decl))
            elif '%ROWTYPE' in decl.upper():
                nodes.append(ASTNode('ROWTYPE_DECL', value=decl))
            elif '%TYPE' in decl.upper():
                nodes.append(ASTNode('TYPE_DECL', value=decl))
            else:
                nodes.append(ASTNode('VAR_DECL', value=decl))
        
        return nodes
    
    def _parse_statements(self, stmt_content: str) -> List[ASTNode]:
        """Parse statements into AST nodes"""
        nodes = []
        
        # Use sqlparse for tokenization
        parsed = sqlparse.parse(stmt_content)
        
        for stmt in parsed:
            stmt_str = str(stmt).strip()
            if not stmt_str:
                continue
            
            # Determine statement type and create appropriate node
            upper_stmt = stmt_str.upper()
            
            if upper_stmt.startswith('IF'):
                nodes.append(ASTNode('IF_STMT', value=stmt_str))
            elif upper_stmt.startswith('FOR') or upper_stmt.startswith('WHILE'):
                nodes.append(ASTNode('LOOP_STMT', value=stmt_str))
            elif upper_stmt.startswith('SELECT'):
                nodes.append(ASTNode('SELECT_STMT', value=stmt_str))
            elif upper_stmt.startswith('INSERT'):
                nodes.append(ASTNode('INSERT_STMT', value=stmt_str))
            elif upper_stmt.startswith('UPDATE'):
                nodes.append(ASTNode('UPDATE_STMT', value=stmt_str))
            elif upper_stmt.startswith('DELETE'):
                nodes.append(ASTNode('DELETE_STMT', value=stmt_str))
            elif upper_stmt.startswith('RETURN'):
                nodes.append(ASTNode('RETURN_STMT', value=stmt_str))
            elif upper_stmt.startswith('RAISE'):
                nodes.append(ASTNode('RAISE_STMT', value=stmt_str))
            elif upper_stmt.startswith('OPEN'):
                nodes.append(ASTNode('OPEN_STMT', value=stmt_str))
            elif upper_stmt.startswith('CLOSE'):
                nodes.append(ASTNode('CLOSE_STMT', value=stmt_str))
            elif upper_stmt.startswith('FETCH'):
                nodes.append(ASTNode('FETCH_STMT', value=stmt_str))
            elif upper_stmt.startswith('EXIT'):
                nodes.append(ASTNode('EXIT_STMT', value=stmt_str))
            elif upper_stmt.startswith('PERFORM'):
                nodes.append(ASTNode('PERFORM_STMT', value=stmt_str))
            else:
                nodes.append(ASTNode('STMT', value=stmt_str))
        
        return nodes


def compare_ast(node1: ASTNode, node2: ASTNode, debug: bool = False) -> bool:
    """
    Compare two AST nodes for semantic equivalence.
    """
    if debug:
        print(f"Comparing: {node1.node_type}({node1.value}) vs {node2.node_type}({node2.value})")
    
    # Compare node types
    if node1.node_type != node2.node_type:
        if debug:
            print(f"  Node type mismatch: {node1.node_type} != {node2.node_type}")
        return False
    
    # Compare values
    if node1.value != node2.value:
        if debug:
            print(f"  Value mismatch: {node1.value} != {node2.value}")
        return False
    
    # Compare children count
    if len(node1.children) != len(node2.children):
        if debug:
            print(f"  Children count mismatch: {len(node1.children)} != {len(node2.children)}")
        return False
    
    # Recursively compare children
    for i, (child1, child2) in enumerate(zip(node1.children, node2.children)):
        if not compare_ast(child1, child2, debug):
            if debug:
                print(f"  Child {i} mismatch")
            return False
    
    return True


def is_plpgsql_semantically_equivalent(code1: str, code2: str, debug: bool = False) -> bool:
    """
    Check if two PL/pgSQL code blocks are semantically equivalent.
    
    This function normalizes both code blocks and compares their AST structures,
    abstracting away differences in:
    - Whitespace and formatting
    - Variable names
    - Parameter names
    - Cursor names
    - Non-semantic code variations
    
    Args:
        code1: First PL/pgSQL code block
        code2: Second PL/pgSQL code block
        debug: If True, print debug information
        
    Returns:
        True if the code blocks are semantically equivalent, False otherwise
    """
    try:
        # Check types match
        type1 = get_plsql_type(code1)
        type2 = get_plsql_type(code2)
        
        if debug:
            print(f"=== PL/pgSQL Semantic Equivalence Check ===")
            print(f"Code1 type: {type1}")
            print(f"Code2 type: {type2}")
        
        if type1 != type2:
            if debug:
                print(f"Type mismatch: {type1} != {type2}")
            return False
        
        # Normalize both code blocks
        normalizer1 = PLpgSQLNormalizer()
        normalizer2 = PLpgSQLNormalizer()
        
        normalized1 = normalizer1.normalize(code1)
        normalized2 = normalizer2.normalize(code2)
        
        if debug:
            print(f"\nNormalized Code 1:\n{normalized1}")
            print(f"\nNormalized Code 2:\n{normalized2}")
        
        # Build ASTs
        builder1 = PLpgSQLASTBuilder()
        builder2 = PLpgSQLASTBuilder()
        
        ast1 = builder1.build(normalized1)
        ast2 = builder2.build(normalized2)
        
        if debug:
            print(f"\nAST 1: {ast1}")
            print(f"\nAST 2: {ast2}")
        
        # Compare ASTs
        result = compare_ast(ast1, ast2, debug)
        
        if debug:
            print(f"\nResult: {result}")
        
        return result
        
    except Exception as e:
        if debug:
            print(f"Error during comparison: {e}")
            import traceback
            traceback.print_exc()
        return False


# =============================================================================
# COMPATIBILITY ALIASES
# =============================================================================

def is_exact_match(plpgsql1: str, plpgsql2: str) -> bool:
    """
    Alias for is_plpgsql_semantically_equivalent.
    Check semantic equivalence of PL/pgSQL code without debug output.
    """
    return is_plpgsql_semantically_equivalent(plpgsql1, plpgsql2, debug=False)


def debug_semantic_equivalence(plpgsql1: str, plpgsql2: str) -> bool:
    """
    Alias for is_plpgsql_semantically_equivalent with debug enabled.
    Check semantic equivalence of PL/pgSQL code with verbose debug output.
    """
    return is_plpgsql_semantically_equivalent(plpgsql1, plpgsql2, debug=True)


def is_exact_match_hybrid_plpgsql(plpgsql1: str, plpgsql2: str, debug: bool = False) -> bool:
    """
    Legacy alias for backward compatibility.
    """
    return is_plpgsql_semantically_equivalent(plpgsql1, plpgsql2, debug=debug)


def debug_semantic_equivalence_ast(plpgsql1: str, plpgsql2: str) -> bool:
    """
    Legacy alias for backward compatibility.
    """
    return is_plpgsql_semantically_equivalent(plpgsql1, plpgsql2, debug=True)

# =============================================================================
# TEST CASES
# =============================================================================

def run_tests():
    """Run comprehensive test cases for the PL/pgSQL semantic equivalence checker"""
    
    print("=" * 80)
    print("PL/pgSQL SEMANTIC EQUIVALENCE CHECKER - TEST SUITE")
    print("=" * 80)
    
    test_results = []
    
    # =========================================================================
    # PROCEDURE TESTS
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("PROCEDURE TESTS")
    print("=" * 40)
    
    # Test 1: Different variable names (should be True)
    print("\n[Test 1] Different variable names in procedure")
    proc1 = """CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ 
               DECLARE v_count INTEGER; 
               BEGIN 
                   SELECT COUNT(*) INTO v_count FROM users WHERE id = para_id;
                   IF v_count > 0 THEN 
                       UPDATE users SET status = 'active' WHERE id = para_id;
                   END IF;
               END; $$;"""
    
    proc2 = """CREATE OR REPLACE PROCEDURE sp(p_id int8) LANGUAGE plpgsql AS $$ 
               DECLARE cnt INTEGER; 
               BEGIN 
                   SELECT COUNT(*) INTO cnt FROM users WHERE id = p_id;
                   IF cnt > 0 THEN 
                       UPDATE users SET status = 'active' WHERE id = p_id;
                   END IF;
               END; $$;"""
    
    result = is_plpgsql_semantically_equivalent(proc1, proc2)
    test_results.append(("Proc: Different variable names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 2: Different cursor names (should be True)
    print("\n[Test 2] Different cursor names")
    proc3 = """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
               LANGUAGE plpgsql AS $$ 
               DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
               rec RECORD; 
               BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; 
               UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cursor; END LOOP; 
               CLOSE ref_cursor; END; $$;"""
    
    proc4 = """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
               LANGUAGE plpgsql AS $$ 
               DECLARE my_cur CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
               r RECORD; 
               BEGIN OPEN my_cur; LOOP FETCH my_cur INTO r; EXIT WHEN NOT FOUND; 
               UPDATE "bank" SET "city" = para_city WHERE CURRENT OF my_cur; END LOOP; 
               CLOSE my_cur; END; $$;"""
    
    result = is_plpgsql_semantically_equivalent(proc3, proc4)
    test_results.append(("Proc: Different cursor names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 3: Different formatting/whitespace (should be True)
    print("\n[Test 3] Different formatting and whitespace")
    proc5 = """CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;"""
    
    proc6 = """CREATE OR REPLACE PROCEDURE sp(para_id int8) 
               LANGUAGE plpgsql 
               AS $$ 
               BEGIN 
                   insert into "access_logs" values(para_id, 1, 0); 
               END; 
               $$;"""
    
    result = is_plpgsql_semantically_equivalent(proc5, proc6)
    test_results.append(("Proc: Different formatting", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 4: Different logic (should be False)
    print("\n[Test 4] Different logic - should NOT match")
    proc7 = """CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ 
               BEGIN 
                   INSERT INTO logs VALUES(para_id, 1, 0); 
               END; $$;"""
    
    proc8 = """CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ 
               BEGIN 
                   DELETE FROM logs WHERE id = para_id; 
               END; $$;"""
    
    result = is_plpgsql_semantically_equivalent(proc7, proc8)
    test_results.append(("Proc: Different logic", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 5: Complex procedure with IF-ELSIF-ELSE
    print("\n[Test 5] Complex procedure with IF-ELSIF-ELSE")
    proc9 = """CREATE OR REPLACE PROCEDURE process_college_leader_analysis()
LANGUAGE plpgsql
AS $$
DECLARE
    v_college_id_val INTEGER;
    v_leader_name_text TEXT;
    v_member_count INTEGER;
BEGIN
    FOR v_college_id_val IN SELECT "College_ID" FROM college ORDER BY "College_ID"
    LOOP
        SELECT COUNT(*) INTO v_member_count FROM member WHERE "College_ID" = v_college_id_val;
        IF v_member_count = 0 THEN
            DELETE FROM college WHERE "College_ID" = v_college_id_val;
        ELSIF v_member_count > 2 THEN
            UPDATE college SET "Leader_Name" = CONCAT("Leader_Name", ' (Active)') WHERE "College_ID" = v_college_id_val;
        ELSE
            UPDATE college SET "College_Location" = REVERSE("College_Location") WHERE "College_ID" = v_college_id_val;
        END IF;
    END LOOP;
END;
$$;"""
    
    proc10 = """CREATE OR REPLACE PROCEDURE process_college_leader_analysis()
LANGUAGE plpgsql
AS $$
DECLARE
    college_id INTEGER;
    leader_name TEXT;
    member_cnt INTEGER;
BEGIN
    FOR college_id IN SELECT "College_ID" FROM college ORDER BY "College_ID"
    LOOP
        SELECT COUNT(*) INTO member_cnt FROM member WHERE "College_ID" = college_id;
        IF member_cnt = 0 THEN
            DELETE FROM college WHERE "College_ID" = college_id;
        ELSIF member_cnt > 2 THEN
            UPDATE college SET "Leader_Name" = CONCAT("Leader_Name", ' (Active)') WHERE "College_ID" = college_id;
        ELSE
            UPDATE college SET "College_Location" = REVERSE("College_Location") WHERE "College_ID" = college_id;
        END IF;
    END LOOP;
END;
$$;"""
    
    result = is_plpgsql_semantically_equivalent(proc9, proc10)
    test_results.append(("Proc: Complex IF-ELSIF-ELSE", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # =========================================================================
    # FUNCTION TESTS
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("FUNCTION TESTS")
    print("=" * 40)
    
    # Test 6: Simple function with different parameter names
    print("\n[Test 6] Function with different parameter names")
    func1 = """CREATE OR REPLACE FUNCTION get_employee_salary(p_employee_id numeric)
RETURNS numeric
AS $$
BEGIN
    IF p_employee_id > 0 THEN
        RETURN (SELECT "SALARY" FROM employees WHERE "EMPLOYEE_ID" = p_employee_id);
    ELSE
        RETURN NULL;
    END IF;
END;
$$ LANGUAGE plpgsql;"""
    
    func2 = """CREATE OR REPLACE FUNCTION get_employee_salary(emp_id numeric)
RETURNS numeric
AS $$
BEGIN
    IF emp_id > 0 THEN
        RETURN (SELECT "SALARY" FROM employees WHERE "EMPLOYEE_ID" = emp_id);
    ELSE
        RETURN NULL;
    END IF;
END;
$$ LANGUAGE plpgsql;"""
    
    result = is_plpgsql_semantically_equivalent(func1, func2)
    test_results.append(("Func: Different parameter names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 7: Function returning different types (should be False)
    print("\n[Test 7] Function with different return types")
    func3 = """CREATE OR REPLACE FUNCTION test_func(p_id integer)
RETURNS integer
AS $$
BEGIN
    RETURN p_id * 2;
END;
$$ LANGUAGE plpgsql;"""
    
    func4 = """CREATE OR REPLACE FUNCTION test_func(p_id integer)
RETURNS text
AS $$
BEGIN
    RETURN p_id * 2;
END;
$$ LANGUAGE plpgsql;"""
    
    result = is_plpgsql_semantically_equivalent(func3, func4)
    test_results.append(("Func: Different return types", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 8: Function with different local variable names
    print("\n[Test 8] Function with different local variable names")
    func5 = """CREATE OR REPLACE FUNCTION calculate_total(p_order_id integer)
RETURNS numeric
AS $$
DECLARE
    total_amount numeric := 0;
    v_price numeric;
    v_qty integer;
BEGIN
    FOR v_price, v_qty IN SELECT "price", "quantity" FROM order_items WHERE "order_id" = p_order_id
    LOOP
        total_amount := total_amount + (v_price * v_qty);
    END LOOP;
    RETURN total_amount;
END;
$$ LANGUAGE plpgsql;"""
    
    func6 = """CREATE OR REPLACE FUNCTION calculate_total(order_id integer)
RETURNS numeric
AS $$
DECLARE
    sum_total numeric := 0;
    item_price numeric;
    qty integer;
BEGIN
    FOR item_price, qty IN SELECT "price", "quantity" FROM order_items WHERE "order_id" = order_id
    LOOP
        sum_total := sum_total + (item_price * qty);
    END LOOP;
    RETURN sum_total;
END;
$$ LANGUAGE plpgsql;"""
    
    result = is_plpgsql_semantically_equivalent(func5, func6)
    test_results.append(("Func: Different local variable names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # =========================================================================
    # TRIGGER TESTS
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("TRIGGER TESTS")
    print("=" * 40)
    
    # Test 9: Trigger with different function/trigger names
    print("\n[Test 9] Trigger with different names")
    trigger1 = """CREATE OR REPLACE FUNCTION update_college_location_on_college_update() RETURNS TRIGGER AS $$
BEGIN
  IF NEW."College_Location" IS DISTINCT FROM 'Updated Location' THEN
    UPDATE college
    SET "College_Location" = 'Updated Location'
    WHERE "College_ID" = NEW."College_ID";
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_college_location_on_college_update
AFTER UPDATE ON college
FOR EACH ROW EXECUTE FUNCTION update_college_location_on_college_update();"""
    
    trigger2 = """CREATE OR REPLACE FUNCTION college_loc_update_trigger_func() RETURNS TRIGGER AS $$
BEGIN
  IF NEW."College_Location" IS DISTINCT FROM 'Updated Location' THEN
    UPDATE college
    SET "College_Location" = 'Updated Location'
    WHERE "College_ID" = NEW."College_ID";
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER college_location_trigger
AFTER UPDATE ON college
FOR EACH ROW EXECUTE FUNCTION college_loc_update_trigger_func();"""
    
    result = is_plpgsql_semantically_equivalent(trigger1, trigger2)
    test_results.append(("Trigger: Different names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 10: Trigger with different timing (BEFORE vs AFTER - should be False)
    print("\n[Test 10] Trigger with different timing")
    trigger3 = """CREATE OR REPLACE FUNCTION audit_func() RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO audit_log VALUES(NEW.id, NOW());
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_trigger
AFTER INSERT ON users
FOR EACH ROW EXECUTE FUNCTION audit_func();"""
    
    trigger4 = """CREATE OR REPLACE FUNCTION audit_func() RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO audit_log VALUES(NEW.id, NOW());
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_trigger
BEFORE INSERT ON users
FOR EACH ROW EXECUTE FUNCTION audit_func();"""
    
    result = is_plpgsql_semantically_equivalent(trigger3, trigger4)
    test_results.append(("Trigger: Different timing (BEFORE/AFTER)", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 11: Trigger with different variable names in function
    print("\n[Test 11] Trigger with different variable names in function")
    trigger5 = """CREATE OR REPLACE FUNCTION validate_data() RETURNS TRIGGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_count FROM related_table WHERE id = NEW.related_id;
    IF v_count = 0 THEN
        RAISE EXCEPTION 'Related record not found';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_trigger
BEFORE INSERT ON main_table
FOR EACH ROW EXECUTE FUNCTION validate_data();"""
    
    trigger6 = """CREATE OR REPLACE FUNCTION validate_data() RETURNS TRIGGER AS $$
DECLARE
    cnt INTEGER;
BEGIN
    SELECT COUNT(*) INTO cnt FROM related_table WHERE id = NEW.related_id;
    IF cnt = 0 THEN
        RAISE EXCEPTION 'Related record not found';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_trigger
BEFORE INSERT ON main_table
FOR EACH ROW EXECUTE FUNCTION validate_data();"""
    
    result = is_plpgsql_semantically_equivalent(trigger5, trigger6)
    test_results.append(("Trigger: Different variable names in function", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # =========================================================================
    # EDGE CASES
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("EDGE CASE TESTS")
    print("=" * 40)
    
    # Test 12: Procedure vs Function (should be False - different types)
    print("\n[Test 12] Procedure vs Function")
    proc_code = """CREATE OR REPLACE PROCEDURE do_something(p_id integer) LANGUAGE plpgsql AS $$
BEGIN
    UPDATE table1 SET col = 1 WHERE id = p_id;
END;
$$;"""
    
    func_code = """CREATE OR REPLACE FUNCTION do_something(p_id integer) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    UPDATE table1 SET col = 1 WHERE id = p_id;
END;
$$;"""
    
    result = is_plpgsql_semantically_equivalent(proc_code, func_code)
    test_results.append(("Edge: Procedure vs Function", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 13: Empty procedure bodies
    print("\n[Test 13] Empty procedure bodies")
    empty1 = """CREATE OR REPLACE PROCEDURE empty_proc() LANGUAGE plpgsql AS $$ BEGIN NULL; END; $$;"""
    empty2 = """CREATE OR REPLACE PROCEDURE empty_proc() LANGUAGE plpgsql AS $$ BEGIN NULL; END; $$;"""
    
    result = is_plpgsql_semantically_equivalent(empty1, empty2)
    test_results.append(("Edge: Empty procedure bodies", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 14: Different number of parameters (should be False)
    print("\n[Test 14] Different number of parameters")
    params1 = """CREATE OR REPLACE PROCEDURE test_params(a integer, b integer) LANGUAGE plpgsql AS $$
BEGIN
    RAISE NOTICE 'a=%', a;
END;
$$;"""
    
    params2 = """CREATE OR REPLACE PROCEDURE test_params(x integer) LANGUAGE plpgsql AS $$
BEGIN
    RAISE NOTICE 'x=%', x;
END;
$$;"""
    
    result = is_plpgsql_semantically_equivalent(params1, params2)
    test_results.append(("Edge: Different number of parameters", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 15: RECORD type declaration
    print("\n[Test 15] RECORD type declarations")
    record1 = """CREATE OR REPLACE PROCEDURE test_record() LANGUAGE plpgsql AS $$
DECLARE
    row_data RECORD;
BEGIN
    FOR row_data IN SELECT * FROM users LOOP
        RAISE NOTICE '%', row_data.name;
    END LOOP;
END;
$$;"""
    
    record2 = """CREATE OR REPLACE PROCEDURE test_record() LANGUAGE plpgsql AS $$
DECLARE
    rec RECORD;
BEGIN
    FOR rec IN SELECT * FROM users LOOP
        RAISE NOTICE '%', rec.name;
    END LOOP;
END;
$$;"""
    
    result = is_plpgsql_semantically_equivalent(record1, record2)
    test_results.append(("Edge: RECORD type declarations", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result, expected in test_results if result == expected)
    total = len(test_results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("\nDetailed Results:")
    for name, result, expected in test_results:
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"  {status}: {name}")
    
    return passed == total


if __name__ == "__main__":

    print("=== Get Database Schema Graph Tests ===\n")
    print(get_database_schema_graph()["3d_coordinate_system_for_spatial_data_management"], "\n")

    print("=== restore_databases Tests ===\n")
    restore_databases(conn_info, host, port, user, password, ["3d_coordinate_system_for_spatial_data_management"])

    print("=== check_plsql_executability Tests ===\n")
    print(check_plsql_executability(f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;""",
                                    ["call sp(5);", "call sp(6);"],
                                    "3d_coordinate_system_for_spatial_data_management"))

    print("=== compare_plsql Tests ===\n")
    print(compare_plsql("3d_coordinate_system_for_spatial_data_management",
                        plsql1=f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;""",
                        plsql2=f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0,'2025-08-08'); END; $$;""",
                        call_plsqls=["call sp(5);", "call sp(6);"],
                        include_system_tables=True), "\n")
    print(compare_plsql("3d_coordinate_system_for_spatial_data_management",
                        plsql1=f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;""",
                        plsql2=f"""CREATE OR REPLACE PROCEDURE sp(para_id int8) LANGUAGE plpgsql AS $$ BEGIN insert into "access_logs" values(para_id,1,0); END; $$;""",
                        call_plsqls=["call sp(5);", "call sp(6);"],
                        include_system_tables=True), "\n")
    
    print("=== restore_databases Tests ===\n")
    restore_databases(conn_info, host, port, user, password, ["3d_coordinate_system_for_spatial_data_management"])


    print("=== Get Table Info Tests ===\n")
    print(get_tables_info("3d_coordinate_system_for_spatial_data_management"), "\n")

    print("=== Get Database Schema Tests ===\n")
    print(get_database_schema("3d_coordinate_system_for_spatial_data_management"), "\n")

    print("=== Get All User Tables Tests ===\n")
    print(get_all_user_tables("3d_coordinate_system_for_spatial_data_management"), "\n")

    print("=== PL/pgSQL Semantic Equivalence Tests ===\n")
    # Test 1: Different cursor names - should return True
    print("Test 1: Different cursor names (should be True)")
    result1 = is_exact_match(
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cursor; END LOOP; 
           CLOSE ref_cursor; END; $$;""",
        
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cur; END LOOP; 
           CLOSE ref_cur; END; $$;"""
    )
    print(f"Result: {result1}\n")
    
    # Test 2: Different formatting and spacing - should return True
    print("Test 2: Different formatting and spacing (should be True)")
    result2 = is_exact_match(
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cursor; END LOOP; 
           CLOSE ref_cursor; END; $$;""",
        
        """CREATE OR REPLACE PROCEDURE sp (para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql   AS $$  
           DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD;   
           BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cur; END LOOP; 
           CLOSE ref_cur; END; $$;"""
    )
    print(f"Result: {result2}\n")

    # print("expect True")
    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cursor; END LOOP; CLOSE ref_cursor; END; $$;""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cur; END LOOP; CLOSE ref_cur; END; $$;"""))
    
    # print("expect True")
    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cursor; END LOOP; CLOSE ref_cursor; END; $$;""",
                         f"""CREATE OR REPLACE PROCEDURE sp (para_state text, para_city text, para_bname text) LANGUAGE plpgsql   AS $$  \nDECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD;   BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cur; END LOOP; CLOSE ref_cur; END; $$;"""))