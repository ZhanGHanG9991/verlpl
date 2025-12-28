import os
import re
import oracledb
import pandas as pd
from pathlib import Path
import sqlparse
from sqlparse import sql, tokens
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import json

from pl_settings import oc_config, get_dataset_config

host = oc_config['host']
port = oc_config['port']
user = oc_config['user']
password = oc_config['password']
service_name = oc_config['service_name']

# 这些变量需要通过 initialize_dataset_paths() 函数来设置
input_path = None
db_schema_graph_path = None
db_schema_dict_path = None

def initialize_dataset_paths():
    """
    根据数据集名称初始化路径配置
    
    Args:
        dataset_name: 数据集名称
    """
    global input_path, db_schema_graph_path, db_schema_dict_path
    
    # dataset_config = get_dataset_config(dataset_name)
    input_path = "/workspace/opt/projects/verlpl/examples/datasets/train/database/oracle"
    db_schema_graph_path = "/workspace/opt/projects/verlpl/examples/datasets/train/schema/oracle_db_schema_graph.json"
    db_schema_dict_path = "/workspace/opt/projects/verlpl/examples/datasets/train/schema/oracle_db_schema_dict.json"

class OracleConnectionManager:
    """
    Oracle连接管理器，支持上下文管理和连接复用
    """
    def __init__(self):
        self.username = user
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self.importer = None
    
    def __enter__(self):
        """进入上下文时建立连接"""
        self.importer = OracleSchemaImporter()
        if not self.importer.connect():
            raise ConnectionError("无法连接到Oracle数据库")
        return self.importer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时断开连接"""
        if self.importer:
            self.importer.disconnect()
            self.importer = None

class OracleSchemaImporter:
    def __init__(self):
        """
        初始化Oracle连接参数
        """
        self.username = user
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self.connection = None
        
    def connect(self):
        """
        连接到Oracle数据库（需要有创建用户的权限）
        """
        try:
            # 构建连接字符串
            dsn = f"{self.host}:{self.port}/{self.service_name}"
            self.connection = oracledb.connect(
                user=self.username, 
                password=self.password, 
                dsn=dsn
            )
            return True
        except oracledb.Error as e:
            print(f"连接Oracle数据库失败: {e}")
            return False
    
    def disconnect(self):
        """
        断开数据库连接
        """
        if self.connection:
            self.connection.close()
    
    def extract_schema_name_from_filename(self, filename):
        """
        从文件名提取Schema名
        例: allergy_1.sqlite.sql -> ALLERGY_1
        """
        # 移除所有后缀
        schema_name = filename
        for suffix in ['.sqlite.sql', '.sql']:
            if schema_name.endswith(suffix):
                schema_name = schema_name[:-len(suffix)]
                break
        
        # 确保名称符合Oracle标识符规范
        schema_name = re.sub(r'[^a-zA-Z0-9_]', '_', schema_name)
        
        # Oracle用户名限制
        if len(schema_name) > 30:
            schema_name = schema_name[:30]
        
        return schema_name.lower()
    
    def create_schema(self, schema_name):
        """
        创建Oracle Schema（用户）
        """
        try:
            cursor = self.connection.cursor()
            
            # 生成密码
            password = f"{schema_name.lower()}_pwd"
            
            # 检查用户是否已存在
            cursor.execute("""
                SELECT COUNT(*) FROM all_users WHERE username = :username
            """, [schema_name])
            
            user_exists = cursor.fetchone()[0] > 0
            
            if user_exists:
                # 删除现有用户
                try:
                    cursor.execute(f"DROP USER {schema_name} CASCADE")
                except oracledb.Error as e:
                    if "does not exist" not in str(e).lower():
                        print(f"删除用户失败: {e}")
            
            # 创建用户
            cursor.execute(f"""
                CREATE USER {schema_name} IDENTIFIED BY {password}
                DEFAULT TABLESPACE USERS
                TEMPORARY TABLESPACE TEMP
            """)
            
            # 授权
            cursor.execute(f"GRANT CONNECT, RESOURCE TO {schema_name}")
            cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {schema_name}")
            
            # 提交
            self.connection.commit()
            cursor.close()
            
            return schema_name, password
            
        except oracledb.Error as e:
            print(f"创建Schema失败: {e}")
            return None, None
    
    def connect_to_schema(self, schema_name, password):
        """
        连接到指定Schema
        """
        try:
            dsn = f"{self.host}:{self.port}/{self.service_name}"
            schema_connection = oracledb.connect(
                user=schema_name, 
                password=password, 
                dsn=dsn
            )
            return schema_connection
        except oracledb.Error as e:
            print(f"连接Schema失败: {e}")
            return None
    
    def execute_sql_file_in_schema(self, sql_file, schema_name, password):
        """
        在指定Schema中执行SQL文件
        """
        # 连接到指定Schema
        schema_connection = self.connect_to_schema(schema_name, password)
        if not schema_connection:
            return False
        
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句（以分号为分隔符）
            sql_statements = []
            statements = sql_content.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    sql_statements.append(statement)
            
            if not sql_statements:
                print(f"SQL文件无效: 没有找到有效的SQL语句")
                schema_connection.close()
                return False
            
            cursor = schema_connection.cursor()
            success_count = 0
            
            for i, sql in enumerate(sql_statements):
                try:
                    cursor.execute(sql)
                    success_count += 1
                except oracledb.Error as e:
                    error_msg = str(e)
                    # 忽略常见的无害错误
                    if "table or view does not exist" in error_msg.lower() and "drop" in sql.lower():
                        continue
                    elif "name is already used by an existing object" in error_msg.lower():
                        # 提取表名
                        table_match = re.search(r'CREATE TABLE\s+"?(\w+)"?', sql, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1)
                            try:
                                cursor.execute(f"DROP TABLE {table_name} CASCADE CONSTRAINTS")
                                cursor.execute(sql)
                                success_count += 1
                            except Exception as drop_error:
                                print(f"重新创建表失败 {table_name}: {drop_error}")
                        continue
                    else:
                        print(f"SQL执行失败 [{i+1}]: {e}")
                        continue
            
            # 提交事务
            schema_connection.commit()
            cursor.close()
            schema_connection.close()
            
            return success_count > 0
            
        except Exception as e:
            print(f"执行SQL文件失败: {e}")
            if schema_connection:
                schema_connection.close()
            return False
    
    def recreate_schema_from_sql(self, schema_name, sql_file_path):
        """
        重新创建Oracle Schema - 先删除后创建
        
        Args:
            schema_name (str): Schema名称（用户名）
            sql_file_path (str): SQL文件路径
        
        Returns:
            tuple: (success, schema_name, password) 成功标志、schema名、密码
        """
        try:
            cursor = self.connection.cursor()
            
            # 生成密码
            password = f"{schema_name.lower()}_pwd"
            
            # 1. 检查并删除现有Schema
            cursor.execute("""
                SELECT COUNT(*) FROM all_users WHERE username = :username
            """, [schema_name])
            
            user_exists = cursor.fetchone()[0] > 0
            
            if user_exists:
                try:
                    cursor.execute(f"DROP USER {schema_name} CASCADE")
                except oracledb.Error as e:
                    if "does not exist" not in str(e).lower():
                        print(f"删除Schema失败: {e}")
                        return False, None, None
            
            # 2. 创建新的Schema
            cursor.execute(f"""
                CREATE USER {schema_name} IDENTIFIED BY {password}
                DEFAULT TABLESPACE USERS
                TEMPORARY TABLESPACE TEMP
            """)
            
            # 3. 授权
            cursor.execute(f"GRANT CONNECT, RESOURCE TO {schema_name}")
            cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {schema_name}")
            
            # 提交创建用户的操作
            self.connection.commit()
            cursor.close()
            
            # 4. 执行SQL文件导入数据
            if self._execute_sql_file_in_schema(sql_file_path, schema_name, password):
                return True, schema_name, password
            else:
                print(f"Schema数据导入失败")
                return False, schema_name, password
                
        except oracledb.Error as e:
            print(f"重新创建Schema失败: {e}")
            return False, None, None

    def _execute_sql_file_in_schema(self, sql_file_path, schema_name, password):
        """
        在指定Schema中执行SQL文件（内部方法）
        
        Args:
            sql_file_path (str): SQL文件路径
            schema_name (str): Schema名称
            password (str): Schema密码
        
        Returns:
            bool: 执行成功标志
        """
        # 连接到指定Schema
        schema_connection = self.connect_to_schema(schema_name, password)
        if not schema_connection:
            return False
        
        try:
            # 读取SQL文件
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句
            sql_statements = []
            statements = sql_content.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    sql_statements.append(statement)
            
            if not sql_statements:
                print(f"SQL文件无效: 没有找到有效的SQL语句")
                schema_connection.close()
                return False
            
            cursor = schema_connection.cursor()
            success_count = 0
            
            # 执行每条SQL语句
            for i, sql in enumerate(sql_statements):
                try:
                    cursor.execute(sql)
                    success_count += 1
                except oracledb.Error as e:
                    error_msg = str(e)
                    # 忽略常见的无害错误
                    if "table or view does not exist" in error_msg.lower() and "drop" in sql.lower():
                        continue
                    elif "name is already used by an existing object" in error_msg.lower():
                        # 提取表名并重建
                        table_match = re.search(r'CREATE TABLE\s+"?(\w+)"?', sql, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1)
                            try:
                                cursor.execute(f"DROP TABLE {table_name} CASCADE CONSTRAINTS")
                                cursor.execute(sql)
                                success_count += 1
                            except Exception as drop_error:
                                print(f"重新创建表失败 {table_name}: {drop_error}")
                        continue
                    else:
                        print(f"SQL执行失败 [{i+1}]: {e}")
                        continue
            
            # 提交事务
            schema_connection.commit()
            cursor.close()
            schema_connection.close()
            
            return success_count > 0
            
        except Exception as e:
            print(f"执行SQL文件失败: {e}")
            if schema_connection:
                schema_connection.close()
            return False

    def import_file_as_schema(self, sql_file):
        """
        为单个SQL文件创建Schema并导入数据
        """
        sql_path = Path(sql_file)
        
        if not sql_path.exists():
            print(f"文件不存在: {sql_file}")
            return False, None, None
        
        # 从文件名提取Schema名
        schema_name = self.extract_schema_name_from_filename(sql_path.name)
        
        # 1. 创建Schema
        schema_name, password = self.create_schema(schema_name)
        if not schema_name:
            return False, None, None
        
        # 2. 在Schema中导入数据
        success = self.execute_sql_file_in_schema(sql_path, schema_name, password)
        
        if success:
            print(f"Schema创建成功: {schema_name}")
            return True, schema_name, password
        else:
            print(f"Schema导入失败: {schema_name}")
            return False, schema_name, password
    
    def import_directory_as_schemas(self, sql_directory):
        """
        为目录中的每个SQL文件创建独立的Schema
        """
        sql_path = Path(sql_directory)
        
        if not sql_path.exists():
            print(f"目录不存在: {sql_directory}")
            return False
        
        sql_files = list(sql_path.glob("*.sql"))
        
        if not sql_files:
            print(f"目录中没有找到SQL文件: {sql_directory}")
            return False
        
        success_count = 0
        created_schemas = []
        
        for sql_file in sorted(sql_files):
            schema_name = self.extract_schema_name_from_filename(sql_file.name)
            
            # 1. 创建Schema
            created_schema, password = self.create_schema(schema_name)
            if not created_schema:
                continue
            
            # 2. 在Schema中导入数据
            if self.execute_sql_file_in_schema(sql_file, created_schema, password):
                success_count += 1
                created_schemas.append({
                    'file': sql_file.name,
                    'schema': created_schema,
                    'password': password,
                    'connection_string': f"{created_schema}/{password}@{self.host}:{self.port}/{self.service_name}"
                })
        
        print(f"批量导入完成: {success_count}/{len(sql_files)} 成功")
        
        if created_schemas:
            for schema_info in created_schemas:
                print(f"Schema: {schema_info['schema']} | 连接: {schema_info['connection_string']}")
        
        return success_count == len(sql_files)


def get_tables_info(database_name):
    """
    获取Oracle指定schema下的数据表信息
    
    Args:
        schema_name: Oracle schema名称
        host: Oracle数据库主机地址
        port: Oracle数据库端口号
        
    Returns:
        dict: 表名为key，列信息列表为value的字典
        格式: {'table_name': [('column_name', 'data_type'), ...]}
    """

    schema_name=database_name.upper()
    tables_info = {}
    
    try:
        # 建立连接 - oracledb支持直接传入参数
        with oracledb.connect(user=user, password=password, 
                             host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 查询指定schema下的所有表名
                print(conn)
                cur.execute("""
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = :schema_name
                    ORDER BY table_name
                """, schema_name=schema_name.upper())
                
                result = cur.fetchall()
                table_names = [table_name[0] for table_name in result]
                
                # 获取每个表的列信息
                for table_name in table_names:
                    cur.execute("""
                        SELECT column_name, data_type
                        FROM all_tab_columns 
                        WHERE owner = :schema_name AND table_name = :table_name
                        ORDER BY column_id
                    """, schema_name=schema_name.upper(), table_name=table_name)
                    
                    result = cur.fetchall()
                    # 直接使用Oracle原始数据类型，不做任何转换
                    tables_info[table_name] = [(col[0], col[1]) for col in result]
                    
    except oracledb.Error as e:
        print(f"Oracle数据库连接或查询出错: {e}")
        return {}
        
    return tables_info

def get_database_schema(database_name):
    """获取数据库schema信息，返回符合DatabaseSchema类型的字典
    
    Args:
        database_name: Oracle schema名称
        
    Returns:
        Dict包含:
        - table_names: List[str] - 所有表名列表
        - tables: Dict[str, List[str]] - 表名到列名列表的映射
    """
    schema_name = database_name.upper()
    table_names = []
    tables = {}
    
    try:
        # 建立连接
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 获取所有表名
                cur.execute("""
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = :schema_name
                    ORDER BY table_name
                """, schema_name=schema_name)
                
                result = cur.fetchall()
                table_names = [table_name[0] for table_name in result]
                
                # 获取每个表的列名
                for table_name in table_names:
                    cur.execute("""
                        SELECT column_name
                        FROM all_tab_columns 
                        WHERE owner = :schema_name AND table_name = :table_name
                        ORDER BY column_id
                    """, schema_name=schema_name, table_name=table_name)
                    
                    result = cur.fetchall()
                    tables[table_name] = [column_name[0] for column_name in result]
                    
    except oracledb.Error as e:
        print(f"Oracle数据库连接或查询出错: {e}")
        return {
            'table_names': [],
            'tables': {}
        }
    
    return {
        'table_names': table_names,
        'tables': tables
    }

def _get_foreign_key_relations(database_name):
    """获取数据库中所有表的外键关系
    
    Args:
        database_name: Oracle schema名称
        
    Returns:
        Dict[str, List[str]]: 字典，键为表名，值为与该表有外键关联的其他表名列表
    """
    schema_name = database_name.upper()
    foreign_key_relations = {}
    
    try:
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 获取所有用户表
                cur.execute("""
                    SELECT table_name
                    FROM all_tables
                    WHERE owner = :schema_name
                    AND table_name NOT LIKE 'BIN$%'
                    ORDER BY table_name
                """, {'schema_name': schema_name})
                tables = [row[0] for row in cur.fetchall()]
                
                # 初始化所有表的外键关系列表
                for table in tables:
                    foreign_key_relations[table] = []
                
                # 查询所有外键关系
                # 在Oracle中，外键约束类型为'R' (Referential)
                cur.execute("""
                    SELECT
                        a.table_name AS from_table,
                        c_pk.table_name AS to_table
                    FROM all_constraints a
                    JOIN all_constraints c_pk 
                        ON a.r_constraint_name = c_pk.constraint_name
                        AND a.r_owner = c_pk.owner
                    WHERE a.constraint_type = 'R'
                        AND a.owner = :schema_name
                        AND c_pk.owner = :schema_name
                    ORDER BY a.table_name
                """, {'schema_name': schema_name})
                
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


def get_detailed_database_schema_oracle(database_name, sample_limit=3):
    """获取Oracle数据库的详细schema信息，适合text2sql任务
    
    Args:
        database_name: Oracle schema名称
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
                                'full_data_type': str,
                                'data_length': int,
                                'data_precision': int,
                                'data_scale': int,
                                'nullable': str,
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
    schema_name = database_name.upper()
    
    try:
        # 建立连接
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 获取所有表名
                cur.execute("""
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = :schema_name
                    AND table_name NOT LIKE 'BIN$%'
                    ORDER BY table_name
                """, schema_name=schema_name)
                
                table_names = [row[0] for row in cur.fetchall()]
                
                schema_dict = {
                    'database_name': database_name,
                    'tables': {},
                    'relationships': []
                }
                
                formatted_parts = []
                
                for table_name in table_names:
                    # 获取列详细信息
                    cur.execute("""
                        SELECT 
                            atc.column_name,
                            atc.data_type,
                            atc.data_length,
                            atc.data_precision,
                            atc.data_scale,
                            atc.nullable,
                            CASE 
                                WHEN acc.constraint_type = 'P' THEN 'PRIMARY KEY'
                                ELSE NULL 
                            END as constraint_type,
                            acc.comments as column_comment
                        FROM all_tab_columns atc
                        LEFT JOIN all_col_comments acc 
                            ON atc.owner = acc.owner 
                            AND atc.table_name = acc.table_name 
                            AND atc.column_name = acc.column_name
                        LEFT JOIN (
                            SELECT 
                                acc.column_name,
                                ac.constraint_type
                            FROM all_cons_columns acc
                            JOIN all_constraints ac 
                                ON acc.owner = ac.owner 
                                AND acc.constraint_name = ac.constraint_name
                            WHERE ac.owner = :schema_name 
                                AND acc.table_name = :table_name
                                AND ac.constraint_type IN ('P', 'U')
                        ) acc ON atc.column_name = acc.column_name
                        WHERE atc.owner = :schema_name 
                            AND atc.table_name = :table_name
                        ORDER BY atc.column_id
                    """, schema_name=schema_name, table_name=table_name)
                    
                    columns_info = cur.fetchall()
                    
                    # 数据采样
                    try:
                        cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {database_name}")
                        cur.execute(f"""
                            SELECT * FROM {table_name} 
                            WHERE ROWNUM <= {sample_limit}
                        """)
                        sample_data = cur.fetchall()
                        column_names = [desc[0] for desc in cur.description] if cur.description else []
                    except Exception as e:
                        print(f"无法采样表 {table_name} 的数据: {e}")
                        sample_data = []
                        column_names = []
                    
                    # 为每个列收集样例数据
                    column_examples = {}
                    if sample_data:
                        for i, row in enumerate(sample_data):
                            for j, col_name in enumerate(column_names):
                                if j < len(row):
                                    if col_name not in column_examples:
                                        column_examples[col_name] = []
                                    value_example = str(row[j]) if row[j] is not None else "NULL"
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
                    
                    # 构建格式化的表描述（保持原有功能）
                    table_schema = f"Table: {table_name}\n"
                    table_schema += "Columns:\n"
                    
                    for col in columns_info:
                        (column_name, data_type, data_length, data_precision, 
                         data_scale, nullable, constraint_type, col_comment) = col
                        
                        # 构建完整的数据类型
                        full_data_type = data_type
                        if data_type in ['VARCHAR2', 'CHAR', 'RAW']:
                            full_data_type += f"({data_length})"
                        elif data_type in ['NUMBER']:
                            if data_precision and data_scale:
                                full_data_type += f"({data_precision},{data_scale})"
                            elif data_precision:
                                full_data_type += f"({data_precision})"
                        
                        # 添加到字典
                        column_dict = {
                            'name': column_name,
                            'data_type': data_type,
                            'full_data_type': full_data_type,
                            'data_length': data_length,
                            'data_precision': data_precision,
                            'data_scale': data_scale,
                            'nullable': nullable,
                            'constraint_type': constraint_type,
                            'comment': col_comment,
                            'examples': column_examples.get(column_name, [])[:5]  # 最多5个样例
                        }
                        table_info['columns'].append(column_dict)
                        
                        # 构建格式化字符串
                        col_info = f"  - {column_name} ({full_data_type})"
                        
                        if constraint_type == 'PRIMARY KEY':
                            col_info += " PRIMARY KEY"
                        
                        if nullable == 'N':
                            col_info += " NOT NULL"
                        
                        if column_name in column_examples and column_examples[column_name]:
                            examples = column_examples[column_name][:5]
                            col_info += f" examples: {examples}"
                        
                        if col_comment:
                            col_info += f" - {col_comment}"
                        
                        table_schema += col_info + "\n"
                    
                    # 保存表信息到字典
                    schema_dict['tables'][table_name] = table_info
                    formatted_parts.append(table_schema)
                
                # 外键关系部分
                relationship_summary = "\nTable Relationships:\n"
                all_relationships = []
                
                for table_name in table_names:
                    cur.execute("""
                        SELECT
                            acc.column_name,
                            acc_child.table_name AS foreign_table_name,
                            acc_child.column_name AS foreign_column_name
                        FROM all_constraints ac
                        JOIN all_cons_columns acc 
                            ON ac.owner = acc.owner 
                            AND ac.constraint_name = acc.constraint_name
                        JOIN all_constraints ac_parent 
                            ON ac.r_constraint_name = ac_parent.constraint_name
                            AND ac.r_owner = ac_parent.owner
                        JOIN all_cons_columns acc_child 
                            ON ac_parent.owner = acc_child.owner 
                            AND ac_parent.constraint_name = acc_child.constraint_name
                        WHERE ac.owner = :schema_name 
                            AND ac.constraint_type = 'R'
                            AND ac.table_name = :table_name
                    """, schema_name=schema_name, table_name=table_name)
                    
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
                
                # 去重并排序
                all_relationships = sorted(list(set(all_relationships)))
                
                if all_relationships:
                    relationship_summary += "\n".join(all_relationships)
                else:
                    relationship_summary += "No foreign key relationships found."
                
                formatted_parts.append(relationship_summary)
                
                # 添加格式化字符串到字典
                schema_dict['formatted_string'] = "\n" + "="*80 + "\n" + "\n".join(formatted_parts) + "\n" + "="*80 + "\n"
                
    except oracledb.Error as e:
        error_msg = f"Oracle数据库连接或查询出错: {e}"
        print(error_msg)
        return {'error': error_msg}
    
    return schema_dict

# 辅助函数：从字典重新生成特定表的prompt（Oracle专用）
def generate_schema_prompt_from_dict(schema_dict, table_names=None):
    """从Oracle schema字典生成特定表的prompt字符串
    
    Args:
        schema_dict: get_detailed_database_schema_oracle返回的字典
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
            col_info = f"  - {col['name']} ({col['full_data_type']})"
            
            if col['constraint_type'] == 'PRIMARY KEY':
                col_info += " PRIMARY KEY"
            
            if col['nullable'] == 'N':
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


def get_database_schema_graph():
    """获取所有Oracle数据库的schema图谱
    
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
    """获取所有 Oracle SQL 数据库的schema JSON

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
    
    for sql_file in tqdm(sql_files[:10]):
        # 从文件名中提取数据库名（去除.sql后缀）
        database_name = sql_file[:-4]
        
        try:
            print(f"Processing database: {database_name}")
            db_schema_dict[database_name] = get_detailed_database_schema_oracle(database_name)
                
        except Exception as e:
            print(f"Error processing database {database_name}: {e}")
            continue
    
    # 保存至db_schema_dict_path
    try:
        os.makedirs(os.path.dirname(db_schema_dict_path), exist_ok=True)
        with open(db_schema_dict_path, 'w', encoding='utf-8') as f:
            json.dump(db_schema_dict, f, ensure_ascii=False, indent=2)
        print(f"Database schema dict saved to {db_schema_dict_path}")
    except Exception as e:
        print(f"Error saving database schema dict: {e}")
    
    return db_schema_dict

def get_all_user_tables(database_name):
    schema_name = database_name.upper()
    
    with oracledb.connect(user=user, password=password, 
                          host=host, port=port, service_name=service_name) as conn:
        with conn.cursor() as cur:
            # 获取指定schema中的所有用户表（排除系统表和回收站表）
            cur.execute("""
                SELECT table_name
                FROM all_tables
                WHERE owner = :schema_name
                AND table_name NOT LIKE 'BIN$%'
                ORDER BY table_name
            """, {'schema_name': schema_name})
            
            result = cur.fetchall()
            return [table_name[0] for table_name in result]

def get_important_system_tables():
    """返回需要监控的重要系统表列表"""
    return [
        'all_constraints',          # 约束信息  
        'all_triggers',             # 触发器信息
        'all_sequences',            # 序列信息
        'all_views'                 # 视图信息
    ]

def fetch_system_table_data(system_table):
    """获取系统表数据，处理可能的权限或存在性问题"""
    try:
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 检查表/视图是否存在（检查all_tables和all_views）
                cur.execute("""
                    SELECT CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM all_tables 
                            WHERE table_name = :table_name
                        ) THEN 1
                        WHEN EXISTS (
                            SELECT 1 FROM all_views 
                            WHERE view_name = :table_name
                        ) THEN 1
                        ELSE 0
                    END as table_exists
                    FROM dual
                """, {'table_name': system_table.upper()})
                
                if not cur.fetchone()[0]:
                    return None
                
                # 尝试查询系统表数据
                # Oracle需要显式指定要排序的列，这里使用ROWNUM作为默认排序
                cur.execute(f"""
                    SELECT * FROM {system_table} 
                    WHERE ROWNUM <= 100
                    ORDER BY 1
                """)
                result = cur.fetchall()
                return result
                
    except Exception as e:
        print(f"Warning: Could not fetch data from system table {system_table}: {e}")
        return None

def cleanup_plsql_objects(plsql_code, database_name):
    """清理PL/SQL代码中可能创建的函数、存储过程和触发器（Oracle版本）"""
    database_name = database_name.upper()
    try:
        recreate_database_with_context(database_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {database_name}")
                
                # 分析PL/SQL代码，识别创建的对象
                objects_to_drop = analyze_plsql_objects(plsql_code)
                
                # 清理识别到的对象
                for obj_type, obj_name in objects_to_drop:
                    if obj_type == 'function':
                        cur.execute(f"DROP FUNCTION {obj_name}")
                    elif obj_type == 'procedure':
                        cur.execute(f"DROP PROCEDURE {obj_name}")
                    elif obj_type == 'trigger':
                        cur.execute(f"DROP TRIGGER {obj_name}")
                
                conn.commit()
                
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
        ('function', r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', re.IGNORECASE),
        ('procedure', r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', re.IGNORECASE),
        ('trigger', r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', re.IGNORECASE)
    ]
    
    for obj_type, pattern, flags in patterns:
        matches = re.finditer(pattern, plsql_code, flags)
        objects.extend((obj_type, match.group(1)) for match in matches)
    
    return objects

def recreate_database_with_context(database_name):
    """
    使用上下文管理器的数据库重建函数（推荐使用）
    
    Args:
        database_name (str): 数据库名称
    
    Returns:
        tuple: (success, schema_name, password)
    """
    schema_name = database_name.upper()
    
    try:
        with OracleConnectionManager() as importer:
            success, schema_name, password = importer.recreate_schema_from_sql(
                schema_name=schema_name,
                sql_file_path=os.path.join(input_path, database_name.lower() + ".sql")
            )
            
            return success, schema_name, password
    except ConnectionError as e:
        print(f"连接错误: {e}")
        return False, None, None
    except Exception as e:
        print(f"操作失败: {e}")
        return False, None, None

def recreate_databases_with_context(database_names):
    """
    使用上下文管理器的批量数据库重建函数（推荐使用）
    
    Args:
        database_names (list): 数据库名称列表
    
    Returns:
        dict: {database_name: (success, schema_name, password)}
    """
    results = {}
    
    try:
        with OracleConnectionManager() as importer:
            for database_name in database_names:
                database_name = database_name.upper()
                
                success, schema_name, password = importer.recreate_schema_from_sql(
                    schema_name=database_name,
                    sql_file_path=os.path.join(input_path, database_name.lower() + ".sql")
                )
                
                results[database_name] = (success, schema_name, password)
        
        return results
    except ConnectionError as e:
        print(f"连接错误: {e}")
        return {name: (False, None, None) for name in database_names}
    except Exception as e:
        print(f"操作失败: {e}")
        return {name: (False, None, None) for name in database_names}

def check_plsql_executability(generated_plsql, call_plsqls, database_name):
    database_name = database_name.upper()
    execution_error = None
    try:
        recreate_database_with_context(database_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {database_name}")
                cur.call_timeout = 2 * 1000  # timeout单位为毫秒
                cur.execute(generated_plsql)
                for call in call_plsqls:
                    cur.execute(call)
                conn.commit()
    except Exception as e:
        execution_error = str(e)
    
    return execution_error

def fetch_query_results(query):
    with oracledb.connect(user=user, password=password, 
                          host=host, port=port, service_name=service_name) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            return result
        
def compare_plsql(schema_name, plsql1, plsql2, call_plsqls, include_system_tables):
    """
    比较两个PL/SQL代码的执行结果
    
    Args:
        schema_name: Oracle schema名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否包含系统表比较
    
    Returns:
        True or False
    """
    try:
        schema_name = schema_name.upper()

        # 获取所有用户表
        all_user_tables = get_all_user_tables(schema_name)
        print(f"Found {len(all_user_tables)} user tables to compare: {all_user_tables}")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Will compare {len(important_system_tables)} system tables")
        
        # 第一次执行
        recreate_database_with_context(schema_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql1)
                for call in call_plsqls:
                    cur.execute(call)
                conn.commit()

        # 收集第一次执行后的数据
        user_tables_results1 = {}
        system_tables_results1 = {}
        
        # 获取所有用户表数据
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        select_query = f'SELECT * FROM {table} ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        user_tables_results1[table] = pd.DataFrame(result)
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        user_tables_results1[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results1[sys_table] = fetch_system_table_data(sys_table)
        
        # 第二次执行
        recreate_database_with_context(schema_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql2)
                for call in call_plsqls:
                    cur.execute(call)
                conn.commit()

        # 收集第二次执行后的数据
        user_tables_results2 = {}
        system_tables_results2 = {}
        
        # 获取所有用户表数据
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        select_query = f'SELECT * FROM {table} ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        user_tables_results2[table] = pd.DataFrame(result)
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        user_tables_results2[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results2[sys_table] = fetch_system_table_data(sys_table)

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


def will_change_data(schema_name, plsql_code, call_plsqls, include_system_tables=False):
    """
    检测执行PL/SQL代码和调用语句是否会改变数据库数据
    
    Args:
        schema_name: Oracle schema名称
        plsql_code: 要检测的PL/SQL代码
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否包含系统表检测
    
    Returns:
        dict: 包含检测结果的字典
    """
    try:
        schema_name = schema_name.upper()

        # 获取所有用户表
        all_user_tables = get_all_user_tables(schema_name)
        print(f"Monitoring {len(all_user_tables)} user tables: {all_user_tables}")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Monitoring {len(important_system_tables)} system tables")
        
        # 记录执行前的数据状态
        before_user_tables_data = {}
        before_system_tables_data = {}
        
        # 获取执行前的用户表数据
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        # 获取表结构和数据
                        select_query = f'SELECT * FROM {table} ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        before_user_tables_data[table] = {
                            'data': pd.DataFrame(result),
                            'row_count': len(result)
                        }
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        before_user_tables_data[table] = None
        
        # 获取执行前的系统表数据
        for sys_table in important_system_tables:
            before_system_tables_data[sys_table] = fetch_system_table_data(sys_table)
        
        # 执行PL/SQL代码
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                try:
                    # 执行主PL/SQL代码
                    cur.execute(plsql_code)
                    
                    # 执行调用语句
                    for call in call_plsqls:
                        cur.execute(call)
                    
                    conn.commit()
                    
                except Exception as e:
                    print(f"Error during PL/SQL execution: {e}")
                    # 发生错误时回滚
                    conn.rollback()
                    return {
                        'will_change_data': False,
                        'error': str(e),
                        'execution_successful': False
                    }
        
        # 记录执行后的数据状态
        after_user_tables_data = {}
        after_system_tables_data = {}
        
        # 获取执行后的用户表数据
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        select_query = f'SELECT * FROM {table} ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        after_user_tables_data[table] = {
                            'data': pd.DataFrame(result),
                            'row_count': len(result)
                        }
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        after_user_tables_data[table] = None
        
        # 获取执行后的系统表数据
        for sys_table in important_system_tables:
            after_system_tables_data[sys_table] = fetch_system_table_data(sys_table)
        
        # 比较数据变化
        user_tables_changed = []
        user_tables_changes_detail = {}
        
        for table in all_user_tables:
            before_data = before_user_tables_data.get(table)
            after_data = after_user_tables_data.get(table)
            
            if before_data is None or after_data is None:
                # 如果表访问有问题，标记为可能变化
                user_tables_changed.append(table)
                user_tables_changes_detail[table] = "Table access error"
                continue
            
            # 比较行数变化
            if before_data['row_count'] != after_data['row_count']:
                user_tables_changed.append(table)
                user_tables_changes_detail[table] = f"Row count changed: {before_data['row_count']} -> {after_data['row_count']}"
                continue
            
            # 比较数据内容变化
            if not before_data['data'].equals(after_data['data']):
                user_tables_changed.append(table)
                user_tables_changes_detail[table] = "Data content changed"
        
        # 比较系统表变化
        system_tables_changed = []
        system_tables_changes_detail = {}
        
        if include_system_tables:
            for sys_table in important_system_tables:
                before_data = before_system_tables_data.get(sys_table)
                after_data = after_system_tables_data.get(sys_table)
                
                if before_data is None or after_data is None:
                    system_tables_changed.append(sys_table)
                    system_tables_changes_detail[sys_table] = "Table access error"
                elif before_data != after_data:
                    system_tables_changed.append(sys_table)
                    system_tables_changes_detail[sys_table] = "System table data changed"
        
        # 判断是否改变了数据
        will_change = len(user_tables_changed) > 0 or len(system_tables_changed) > 0
        
        result = {
            'will_change_data': will_change,
            'execution_successful': True,
            'user_tables_changed': user_tables_changed,
            'system_tables_changed': system_tables_changed,
            'user_tables_changes_detail': user_tables_changes_detail,
            'system_tables_changes_detail': system_tables_changes_detail,
            'total_user_tables_monitored': len(all_user_tables),
            'total_system_tables_monitored': len(important_system_tables),
            'changed_user_tables_count': len(user_tables_changed),
            'changed_system_tables_count': len(system_tables_changed)
        }
        
        print(f"Data change detection result: {result['will_change_data']}")
        if will_change:
            print(f"Changed user tables: {user_tables_changed}")
            if system_tables_changed:
                print(f"Changed system tables: {system_tables_changed}")
        
        return result
    
    except Exception as e:
        print(f"Error in will_change_data: {e}")
        return {
            'will_change_data': False,
            'error': str(e),
            'execution_successful': False
        }


def will_change_data_simple(schema_name, plsql_code, call_plsqls):
    """
    简化版本：只返回布尔值表示是否会改变数据
    """
    result = will_change_data(schema_name, plsql_code, call_plsqls, False)
    return result.get('will_change_data', False)

def compare_plsql_function(schema_name, plsql1, plsql2, call_plsqls):
    """
    比较两个PL/SQL函数代码在Oracle中的执行结果
    
    Args:
        schema_name: Oracle schema名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
    
    Returns:
        True or False
    """
    try:
        schema_name = schema_name.upper()
        
        # 第一次执行
        recreate_database_with_context(schema_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql1)
                conn.commit()

        # 收集第一次执行的结果
        function_results1 = {}
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for i, call_plsql in enumerate(call_plsqls):
                    try:
                        cur.execute(call_plsql)
                        result = cur.fetchall()
                        
                        # 获取列名
                        col_names = [desc[0] for desc in cur.description] if cur.description else []
                        
                        if col_names:
                            function_results1[i] = {
                                'sql': call_plsql,
                                'result': pd.DataFrame(result, columns=col_names)
                            }
                        else:
                            # 如果没有列名（例如执行存储过程），创建默认DataFrame
                            function_results1[i] = {
                                'sql': call_plsql,
                                'result': pd.DataFrame(result) if result else pd.DataFrame()
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
        recreate_database_with_context(schema_name)
        
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql2)
                conn.commit()

        # 收集第二次执行的结果
        function_results2 = {}
        with oracledb.connect(user=user, password=password, 
                              host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for i, call_plsql in enumerate(call_plsqls):
                    try:
                        cur.execute(call_plsql)
                        result = cur.fetchall()
                        
                        # 获取列名
                        col_names = [desc[0] for desc in cur.description] if cur.description else []
                        
                        if col_names:
                            function_results2[i] = {
                                'sql': call_plsql,
                                'result': pd.DataFrame(result, columns=col_names)
                            }
                        else:
                            # 如果没有列名，创建默认DataFrame
                            function_results2[i] = {
                                'sql': call_plsql,
                                'result': pd.DataFrame(result) if result else pd.DataFrame()
                            }
                    except Exception as e:
                        print(f"Warning: Could not execute call statement {i}: {call_plsql}")
                        print(f"Error: {e}")
                        function_results2[i] = {
                            'sql': call_plsql,
                            'result': None,
                            'error': str(e)
                        }

        # 比较两次执行结果
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
        print(f"Error in compare_plsql_function: {e}")
        import traceback
        traceback.print_exc()
        return False

"""
Oracle PL/SQL Semantic Equivalence Checker

This module provides tools for comparing two Oracle PL/SQL code blocks to determine if they are
semantically equivalent, even when they differ in:
- Whitespace and formatting
- Variable names (identifiers)
- Parameter names
- Cursor names
- Exception handler names
- Code structure spacing

The tool uses a hybrid approach:
1. Text preprocessing to normalize syntax and formatting
2. Abstract Syntax Tree (AST) parsing using sqlparse
3. Semantic comparison of normalized AST structures

Supports:
- Procedures (CREATE OR REPLACE PROCEDURE)
- Functions (CREATE OR REPLACE FUNCTION)
- Triggers (CREATE OR REPLACE TRIGGER)

Usage:
    from oracle_plsql_checker import is_exact_match
    
    result = is_exact_match(code1, code2)
    
    # With debug output
    result = debug_semantic_equivalence(code1, code2)
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple


class PLSQLType(Enum):
    """Enumeration of PL/SQL object types"""
    PROCEDURE = "procedure"
    FUNCTION = "function"
    TRIGGER = "trigger"
    PACKAGE = "package"
    UNKNOWN = "unknown"


@dataclass
class ASTNode:
    """Abstract Syntax Tree Node for PL/SQL"""
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


def get_plsql_type(plsql_code: str) -> PLSQLType:
    """
    Determine the type of PL/SQL code (procedure, function, trigger, or package)
    
    Args:
        plsql_code: The PL/SQL code to analyze
        
    Returns:
        PLSQLType enum value indicating the type
    """
    code_upper = plsql_code.upper()
    
    # Check for trigger first
    has_trigger = bool(re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+', code_upper))
    has_function = bool(re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+', code_upper))
    has_procedure = bool(re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+', code_upper))
    has_package = bool(re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?PACKAGE\s+', code_upper))
    
    if has_trigger:
        return PLSQLType.TRIGGER
    elif has_package:
        return PLSQLType.PACKAGE
    elif has_procedure:
        return PLSQLType.PROCEDURE
    elif has_function:
        return PLSQLType.FUNCTION
    else:
        return PLSQLType.UNKNOWN


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
        self.exception_map: Dict[str, str] = {}
        self.type_map: Dict[str, str] = {}  # For user-defined types
        self.loop_var_map: Dict[str, str] = {}  # For loop variables
        self.counter = {'param': 0, 'var': 0, 'cursor': 0, 'label': 0, 'exception': 0, 'type': 0, 'loop_var': 0}
        
        # Oracle system objects that should NOT be abstracted
        self.system_objects = {
            # Trigger pseudo-records
            'NEW', 'OLD', ':NEW', ':OLD',
            # System variables
            'SQL', 'SQLCODE', 'SQLERRM', 'SQL%FOUND', 'SQL%NOTFOUND', 
            'SQL%ROWCOUNT', 'SQL%ISOPEN', 'SQL%BULK_ROWCOUNT', 'SQL%BULK_EXCEPTIONS',
            # Boolean values
            'TRUE', 'FALSE', 'NULL',
            # Cursor attributes
            '%FOUND', '%NOTFOUND', '%ROWCOUNT', '%ISOPEN', '%TYPE', '%ROWTYPE',
            # System functions
            'SYSDATE', 'SYSTIMESTAMP', 'USER', 'UID', 'CURRENT_DATE', 
            'CURRENT_TIMESTAMP', 'SESSIONTIMEZONE', 'DBTIMEZONE',
            'ROWNUM', 'ROWID', 'LEVEL', 'CONNECT_BY_ISLEAF', 'CONNECT_BY_ISCYCLE',
            # Standard exceptions
            'NO_DATA_FOUND', 'TOO_MANY_ROWS', 'INVALID_CURSOR', 'ZERO_DIVIDE',
            'CURSOR_ALREADY_OPEN', 'DUP_VAL_ON_INDEX', 'VALUE_ERROR', 'INVALID_NUMBER',
            'STORAGE_ERROR', 'PROGRAM_ERROR', 'NOT_LOGGED_ON', 'LOGIN_DENIED',
            'TIMEOUT_ON_RESOURCE', 'OTHERS',
            # DBMS packages
            'DBMS_OUTPUT', 'DBMS_LOB', 'UTL_FILE', 'DBMS_SQL', 'DBMS_LOCK',
        }
        
        # PL/SQL keywords that should not be abstracted
        self.keywords = {
            # DDL
            'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'TRIGGER', 'PACKAGE', 'BODY',
            'ALTER', 'DROP', 'COMPILE', 'GRANT', 'REVOKE',
            # Structure
            'IS', 'AS', 'BEGIN', 'END', 'DECLARE', 'EXCEPTION', 'WHEN', 'THEN', 'PRAGMA',
            # Control flow
            'IF', 'ELSIF', 'ELSE', 'CASE', 'LOOP', 'WHILE', 'FOR', 'EXIT', 'CONTINUE',
            'GOTO', 'RETURN', 'RAISE', 'NULL',
            # DML
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'MERGE',
            'VALUES', 'DEFAULT', 'RETURNING', 'BULK', 'COLLECT', 'FORALL', 'SAVE', 'EXCEPTIONS',
            # Clauses
            'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'ESCAPE', 'IS',
            'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'NATURAL', 'FULL', 'ON', 'USING',
            'GROUP', 'BY', 'ORDER', 'ASC', 'DESC', 'NULLS', 'FIRST', 'LAST',
            'HAVING', 'UNION', 'INTERSECT', 'MINUS', 'ALL', 'DISTINCT', 'UNIQUE',
            'CONNECT', 'START', 'WITH', 'PRIOR', 'NOCYCLE',
            # Cursor
            'CURSOR', 'OPEN', 'CLOSE', 'FETCH', 'NEXT', 'PRIOR', 'CURRENT', 'OF',
            'REF', 'SYS_REFCURSOR', 'SCROLL', 'HOLD',
            # Transaction
            'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'WORK',
            # Parameters
            'IN', 'OUT', 'NOCOPY',
            # Trigger
            'BEFORE', 'AFTER', 'INSTEAD', 'EACH', 'ROW', 'STATEMENT', 'FOLLOWS', 'PRECEDES',
            'REFERENCING', 'ENABLE', 'DISABLE', 'COMPOUND', 'CALL',
            # Data types
            'NUMBER', 'VARCHAR2', 'VARCHAR', 'CHAR', 'NCHAR', 'NVARCHAR2', 'CLOB', 'NCLOB', 'BLOB',
            'DATE', 'TIMESTAMP', 'INTERVAL', 'DAY', 'SECOND', 'YEAR', 'MONTH', 'ZONE', 'LOCAL',
            'INTEGER', 'INT', 'SMALLINT', 'REAL', 'FLOAT', 'DOUBLE', 'PRECISION',
            'BINARY_INTEGER', 'PLS_INTEGER', 'SIMPLE_INTEGER', 'BINARY_FLOAT', 'BINARY_DOUBLE',
            'BOOLEAN', 'LONG', 'RAW', 'ROWID', 'UROWID', 'XMLTYPE', 'ANYDATA', 'ANYTYPE',
            'RECORD', 'TABLE', 'VARRAY', 'ARRAY', 'TYPE', 'SUBTYPE', 'OBJECT', 'REF',
            # Aggregate functions
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'LISTAGG', 'STRAGG',
            'STDDEV', 'VARIANCE', 'FIRST_VALUE', 'LAST_VALUE', 'RANK', 'DENSE_RANK', 'ROW_NUMBER',
            'LEAD', 'LAG', 'NTILE', 'CUME_DIST', 'PERCENT_RANK', 'PERCENTILE_CONT', 'PERCENTILE_DISC',
            # String functions
            'LENGTH', 'LENGTHB', 'LENGTHC', 'LENGTH2', 'LENGTH4',
            'SUBSTR', 'SUBSTRB', 'SUBSTRC', 'SUBSTR2', 'SUBSTR4',
            'INSTR', 'INSTRB', 'INSTRC', 'INSTR2', 'INSTR4',
            'UPPER', 'LOWER', 'INITCAP', 'TRIM', 'LTRIM', 'RTRIM', 'LPAD', 'RPAD',
            'REPLACE', 'TRANSLATE', 'CONCAT', 'REVERSE', 'ASCII', 'CHR', 'NCHR',
            'SOUNDEX', 'REGEXP_LIKE', 'REGEXP_INSTR', 'REGEXP_SUBSTR', 'REGEXP_REPLACE', 'REGEXP_COUNT',
            # Numeric functions
            'ABS', 'SIGN', 'CEIL', 'FLOOR', 'ROUND', 'TRUNC', 'MOD', 'POWER', 'SQRT',
            'EXP', 'LN', 'LOG', 'SIN', 'COS', 'TAN', 'ASIN', 'ACOS', 'ATAN', 'ATAN2',
            'SINH', 'COSH', 'TANH', 'GREATEST', 'LEAST', 'NANVL', 'WIDTH_BUCKET',
            # Date functions
            'ADD_MONTHS', 'MONTHS_BETWEEN', 'NEXT_DAY', 'LAST_DAY', 'EXTRACT',
            'TO_DATE', 'TO_CHAR', 'TO_NUMBER', 'TO_TIMESTAMP', 'TO_TIMESTAMP_TZ',
            'TO_DSINTERVAL', 'TO_YMINTERVAL', 'NUMTODSINTERVAL', 'NUMTOYMINTERVAL',
            'FROM_TZ', 'SYS_EXTRACT_UTC', 'SESSIONTIMEZONE', 'DBTIMEZONE',
            # Conversion
            'CAST', 'CONVERT', 'NVL', 'NVL2', 'NULLIF', 'COALESCE', 'DECODE', 'DUMP',
            'EMPTY_CLOB', 'EMPTY_BLOB', 'HEXTORAW', 'RAWTOHEX', 'ROWIDTOCHAR', 'CHARTOROWID',
            # Misc
            'OVER', 'PARTITION', 'ROWS', 'RANGE', 'UNBOUNDED', 'PRECEDING', 'FOLLOWING',
            'WITHIN', 'RESPECT', 'IGNORE', 'KEEP', 'DENSE_RANK',
            'AUTONOMOUS_TRANSACTION', 'EXCEPTION_INIT', 'RESTRICT_REFERENCES', 'SERIALLY_REUSABLE',
            # Object-oriented
            'MEMBER', 'STATIC', 'CONSTRUCTOR', 'MAP', 'ORDER', 'SELF', 'FINAL', 'INSTANTIABLE',
            'OVERRIDING', 'UNDER', 'NOT', 'DETERMINISTIC', 'PARALLEL_ENABLE', 'PIPELINED', 'RESULT_CACHE',
            # Additional
            'PUT_LINE', 'GET_LINE', 'PUT', 'NEW_LINE'
        }
    
    def is_system_identifier(self, name: str) -> bool:
        """Check if identifier is a system object or keyword"""
        upper_name = name.upper().strip('"').lstrip(':')
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
    
    def map_exception(self, name: str) -> str:
        """Map a user-defined exception name to abstract placeholder"""
        if self.is_system_identifier(name):
            return name.upper()
        lower_name = name.lower().strip('"')
        if lower_name not in self.exception_map:
            self.exception_map[lower_name] = f"<EXCEPTION_{self.counter['exception']}>"
            self.counter['exception'] += 1
        return self.exception_map[lower_name]
    
    def map_type(self, name: str) -> str:
        """Map a user-defined type name to abstract placeholder"""
        if self.is_system_identifier(name):
            return name.upper()
        lower_name = name.lower().strip('"')
        if lower_name not in self.type_map:
            self.type_map[lower_name] = f"<TYPE_{self.counter['type']}>"
            self.counter['type'] += 1
        return self.type_map[lower_name]
    
    def map_loop_var(self, name: str) -> str:
        """Map a loop variable name to abstract placeholder"""
        if self.is_system_identifier(name):
            return name.upper()
        lower_name = name.lower().strip('"')
        if lower_name not in self.loop_var_map:
            self.loop_var_map[lower_name] = f"<LOOP_VAR_{self.counter['loop_var']}>"
            self.counter['loop_var'] += 1
        return self.loop_var_map[lower_name]
    
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
        if lower_name in self.exception_map:
            return self.exception_map[lower_name]
        if lower_name in self.type_map:
            return self.type_map[lower_name]
        if lower_name in self.loop_var_map:
            return self.loop_var_map[lower_name]
        return None


class OraclePLSQLNormalizer:
    """
    Normalizes Oracle PL/SQL code by:
    1. Extracting and parsing the structure
    2. Abstracting user-defined identifiers
    3. Normalizing whitespace and formatting
    """
    
    def __init__(self):
        self.mapper = IdentifierMapper()
        self.declared_vars: Set[str] = set()
        self.declared_cursors: Set[str] = set()
        self.declared_params: Set[str] = set()
        self.declared_exceptions: Set[str] = set()
        self.declared_types: Set[str] = set()  # Track user-defined types
        self.declared_loop_vars: Set[str] = set()  # Track loop variables
    
    def extract_body_parts(self, code: str) -> Tuple[str, str, str, str]:
        """
        Extract header, declaration section, body, and exception handler from PL/SQL code.
        
        For Oracle PL/SQL:
        CREATE OR REPLACE PROCEDURE/FUNCTION name(params) IS/AS
            [DECLARE section - for triggers only]
            [declaration section]
        BEGIN
            body
        [EXCEPTION
            exception handlers]
        END [name];
        
        Returns:
            (header, declarations, body, exception_section)
        """
        # Normalize line breaks
        code = code.replace('\\n', '\n')
        
        # Find IS/AS keyword (marks end of header)
        is_as_match = re.search(r'\b(IS|AS)\b', code, re.IGNORECASE)
        if not is_as_match:
            return code, "", "", ""
        
        header = code[:is_as_match.end()].strip()
        rest = code[is_as_match.end():].strip()
        
        # Find BEGIN keyword
        begin_match = re.search(r'\bBEGIN\b', rest, re.IGNORECASE)
        if not begin_match:
            return header, rest, "", ""
        
        declarations = rest[:begin_match.start()].strip()
        rest_after_begin = rest[begin_match.end():].strip()
        
        # Find EXCEPTION section (if present)
        exception_match = re.search(r'\bEXCEPTION\b', rest_after_begin, re.IGNORECASE)
        
        # Find END keyword - need to handle nested BEGIN...END blocks
        end_match = self._find_matching_end(rest_after_begin)
        
        if exception_match and (end_match is None or exception_match.start() < end_match):
            body = rest_after_begin[:exception_match.start()].strip()
            exception_and_end = rest_after_begin[exception_match.end():].strip()
            # Extract exception section (up to END)
            end_in_exc = self._find_matching_end(exception_and_end)
            if end_in_exc:
                exception_section = exception_and_end[:end_in_exc].strip()
            else:
                exception_section = exception_and_end.strip()
        else:
            if end_match:
                body = rest_after_begin[:end_match].strip()
            else:
                body = rest_after_begin.strip()
            exception_section = ""
        
        return header, declarations, body, exception_section
    
    def _find_matching_end(self, code: str) -> Optional[int]:
        """Find the position of matching END keyword, handling nested blocks"""
        depth = 0
        i = 0
        code_upper = code.upper()
        
        while i < len(code):
            # Check for BEGIN
            if code_upper[i:i+5] == 'BEGIN':
                if i == 0 or not code[i-1].isalnum():
                    if i + 5 >= len(code) or not code[i+5].isalnum():
                        depth += 1
                        i += 5
                        continue
            
            # Check for END
            if code_upper[i:i+3] == 'END':
                if i == 0 or not code[i-1].isalnum():
                    # Check if it's END IF, END LOOP, END CASE, etc. vs END (block end)
                    rest = code[i+3:].strip()
                    if not rest or rest[0] in (';', ' '):
                        # Check what follows
                        next_word_match = re.match(r'\s*(\w+)', rest)
                        if next_word_match:
                            next_word = next_word_match.group(1).upper()
                            if next_word in ('IF', 'LOOP', 'CASE'):
                                i += 3
                                continue
                        if depth == 0:
                            return i
                        depth -= 1
            i += 1
        
        # Fallback: find last END
        match = re.search(r'\bEND\b\s*[^;]*;?\s*$', code, re.IGNORECASE)
        if match:
            return match.start()
        return None
    
    def extract_trigger_parts(self, code: str) -> Tuple[str, str, str, str, str]:
        """
        Extract trigger-specific parts from Oracle trigger code.
        
        Returns:
            (header, timing_event, declarations, body, exception_section)
        """
        code = code.replace('\\n', '\n')
        
        # Find the DECLARE or BEGIN keyword to separate trigger definition from body
        declare_match = re.search(r'\bDECLARE\b', code, re.IGNORECASE)
        begin_match = re.search(r'\bBEGIN\b', code, re.IGNORECASE)
        
        if declare_match and (not begin_match or declare_match.start() < begin_match.start()):
            # Has DECLARE section
            trigger_def = code[:declare_match.start()].strip()
            rest = code[declare_match.start():]
        elif begin_match:
            # No DECLARE, starts with BEGIN
            trigger_def = code[:begin_match.start()].strip()
            rest = code[begin_match.start():]
        else:
            return code, "", "", "", ""
        
        # Parse trigger definition for timing and event
        # Format: CREATE OR REPLACE TRIGGER name BEFORE/AFTER INSERT/UPDATE/DELETE ON table FOR EACH ROW
        timing_event = ""
        timing_match = re.search(
            r'((?:BEFORE|AFTER|INSTEAD\s+OF)\s+(?:INSERT|UPDATE|DELETE)(?:\s+OR\s+(?:INSERT|UPDATE|DELETE))*\s+ON\s+\w+(?:\s+FOR\s+EACH\s+(?:ROW|STATEMENT))?)',
            trigger_def, re.IGNORECASE
        )
        if timing_match:
            timing_event = timing_match.group(1)
        
        # Extract header (CREATE OR REPLACE TRIGGER name)
        header_match = re.match(r'(CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+\w+)', trigger_def, re.IGNORECASE)
        header = header_match.group(1) if header_match else trigger_def
        
        # Parse rest as body
        _, declarations, body, exception_section = self.extract_body_parts(f"DUMMY IS {rest}")
        
        return header, timing_event, declarations, body, exception_section
    
    def extract_parameters(self, header: str) -> List[Tuple[str, str, str]]:
        """
        Extract parameters from function/procedure header.
        
        Returns:
            List of (name, mode, type) tuples
        """
        params = []
        
        # Match parameter list in parentheses
        paren_match = re.search(r'\(\s*([^)]*)\s*\)', header, re.DOTALL)
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
            
            # Parse parameter: name [IN|OUT|IN OUT] [NOCOPY] type [:= default]
            param_match = re.match(
                r'(\w+)\s+(?:(IN\s+OUT|IN|OUT)\s+)?(?:NOCOPY\s+)?(.+?)(?:\s*:=\s*.+)?$',
                part, re.IGNORECASE | re.DOTALL
            )
            if param_match:
                param_name = param_match.group(1)
                param_mode = (param_match.group(2) or 'IN').upper().replace('IN OUT', 'IN_OUT')
                param_type = param_match.group(3).strip()
                params.append((param_name, param_mode, param_type))
                self.declared_params.add(param_name.lower())
        
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
    
    def extract_declarations(self, decl_section: str) -> List[Tuple[str, str, str]]:
        """
        Extract declarations from DECLARE section.
        
        Returns:
            List of (type, name, definition) tuples
            type can be: 'variable', 'cursor', 'exception', 'type', 'subtype', 'constant'
        """
        declarations = []
        
        if not decl_section.strip():
            return declarations
        
        # Remove DECLARE keyword if present
        decl_section = re.sub(r'^\s*DECLARE\s*', '', decl_section, flags=re.IGNORECASE)
        
        # Split by semicolon
        decl_parts = self._split_by_semicolon(decl_section)
        
        for part in decl_parts:
            part = part.strip()
            if not part:
                continue
            
            # Check for cursor declaration: CURSOR cursor_name IS SELECT...
            cursor_match = re.match(
                r'CURSOR\s+(\w+)\s+IS\s+(.+)',
                part, re.IGNORECASE | re.DOTALL
            )
            if cursor_match:
                cursor_name = cursor_match.group(1)
                cursor_query = cursor_match.group(2)
                declarations.append(('cursor', cursor_name, cursor_query))
                self.declared_cursors.add(cursor_name.lower())
                continue
            
            # Check for cursor (parameterized): CURSOR cursor_name(params) IS SELECT...
            cursor_param_match = re.match(
                r'CURSOR\s+(\w+)\s*\([^)]*\)\s+IS\s+(.+)',
                part, re.IGNORECASE | re.DOTALL
            )
            if cursor_param_match:
                cursor_name = cursor_param_match.group(1)
                cursor_query = cursor_param_match.group(2)
                declarations.append(('cursor', cursor_name, cursor_query))
                self.declared_cursors.add(cursor_name.lower())
                continue
            
            # Alternative cursor syntax: cursor_name CURSOR FOR SELECT...
            cursor_for_match = re.match(
                r'(\w+)\s+CURSOR\s+(?:FOR\s+)?(.+)',
                part, re.IGNORECASE | re.DOTALL
            )
            if cursor_for_match:
                cursor_name = cursor_for_match.group(1)
                cursor_query = cursor_for_match.group(2)
                declarations.append(('cursor', cursor_name, cursor_query))
                self.declared_cursors.add(cursor_name.lower())
                continue
            
            # Check for exception declaration
            exception_match = re.match(r'(\w+)\s+EXCEPTION\s*$', part, re.IGNORECASE)
            if exception_match:
                exc_name = exception_match.group(1)
                declarations.append(('exception', exc_name, 'EXCEPTION'))
                self.declared_exceptions.add(exc_name.lower())
                continue
            
            # Check for TYPE declaration
            type_match = re.match(r'TYPE\s+(\w+)\s+IS\s+(.+)', part, re.IGNORECASE | re.DOTALL)
            if type_match:
                type_name = type_match.group(1)
                type_def = type_match.group(2)
                declarations.append(('type', type_name, type_def))
                self.declared_types.add(type_name.lower())
                continue
            
            # Check for SUBTYPE declaration
            subtype_match = re.match(r'SUBTYPE\s+(\w+)\s+IS\s+(.+)', part, re.IGNORECASE | re.DOTALL)
            if subtype_match:
                subtype_name = subtype_match.group(1)
                subtype_def = subtype_match.group(2)
                declarations.append(('subtype', subtype_name, subtype_def))
                continue
            
            # Check for CONSTANT declaration
            const_match = re.match(
                r'(\w+)\s+CONSTANT\s+(.+?)\s*:=\s*(.+)',
                part, re.IGNORECASE | re.DOTALL
            )
            if const_match:
                const_name = const_match.group(1)
                const_type = const_match.group(2)
                const_val = const_match.group(3)
                declarations.append(('constant', const_name, f"{const_type} := {const_val}"))
                self.declared_vars.add(const_name.lower())
                continue
            
            # Check for %TYPE reference
            type_ref_match = re.match(r'(\w+)\s+(.+%TYPE)\s*(?::=\s*(.+))?', part, re.IGNORECASE)
            if type_ref_match:
                var_name = type_ref_match.group(1)
                var_type = type_ref_match.group(2)
                default_val = type_ref_match.group(3)
                def_str = f"{var_type}" + (f" := {default_val}" if default_val else "")
                declarations.append(('variable', var_name, def_str))
                self.declared_vars.add(var_name.lower())
                continue
            
            # Check for %ROWTYPE reference
            rowtype_match = re.match(r'(\w+)\s+(.+%ROWTYPE)\s*', part, re.IGNORECASE)
            if rowtype_match:
                var_name = rowtype_match.group(1)
                var_type = rowtype_match.group(2)
                declarations.append(('record', var_name, var_type))
                self.declared_vars.add(var_name.lower())
                continue
            
            # Check for RECORD type variable
            record_match = re.match(r'(\w+)\s+(\w+)\s*$', part, re.IGNORECASE)
            if record_match:
                var_name = record_match.group(1)
                var_type = record_match.group(2)
                if not self.mapper.is_system_identifier(var_name):
                    declarations.append(('variable', var_name, var_type))
                    self.declared_vars.add(var_name.lower())
                continue
            
            # Regular variable declaration
            var_match = re.match(r'(\w+)\s+(.+)', part, re.IGNORECASE | re.DOTALL)
            if var_match:
                var_name = var_match.group(1)
                var_def = var_match.group(2)
                if not self.mapper.is_system_identifier(var_name):
                    declarations.append(('variable', var_name, var_def))
                    self.declared_vars.add(var_name.lower())
        
        return declarations
    
    def _split_by_semicolon(self, text: str) -> List[str]:
        """Split text by semicolon, handling nested structures"""
        parts = []
        current = ""
        depth = 0
        in_string = False
        string_char = None
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # Handle strings
            if char in ("'", '"') and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                # Check for escaped quote
                if i + 1 < len(text) and text[i + 1] == string_char:
                    current += char
                    i += 1
                else:
                    in_string = False
                    string_char = None
            
            if not in_string:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == ';' and depth == 0:
                    parts.append(current.strip())
                    current = ""
                    i += 1
                    continue
            
            current += char
            i += 1
        
        if current.strip():
            parts.append(current.strip())
        
        return parts
    
    def normalize_header(self, header: str, obj_type: PLSQLType) -> str:
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
            for name, mode, ptype in params:
                abstract_name = self.mapper.map_parameter(name)
                ptype_normalized = self._normalize_type(ptype)
                if mode == 'IN':
                    param_strs.append(f"{abstract_name} IN {ptype_normalized}")
                elif mode == 'OUT':
                    param_strs.append(f"{abstract_name} OUT {ptype_normalized}")
                elif mode == 'IN_OUT':
                    param_strs.append(f"{abstract_name} IN OUT {ptype_normalized}")
                else:
                    param_strs.append(f"{abstract_name} {ptype_normalized}")
            
            new_param_list = ', '.join(param_strs)
            header = re.sub(r'\([^)]*\)', f'({new_param_list})', header, count=1)
        
        # Abstract procedure/function name
        if obj_type == PLSQLType.PROCEDURE:
            header = re.sub(
                r'(PROCEDURE)\s+(\w+)',
                r'\1 <PROC_NAME>',
                header,
                flags=re.IGNORECASE
            )
        elif obj_type == PLSQLType.FUNCTION:
            header = re.sub(
                r'(FUNCTION)\s+(\w+)',
                r'\1 <FUNC_NAME>',
                header,
                flags=re.IGNORECASE
            )
            # Normalize RETURN type
            header = re.sub(
                r'RETURN\s+(\w+(?:\([^)]*\))?)',
                lambda m: f"RETURN {self._normalize_type(m.group(1))}",
                header,
                flags=re.IGNORECASE
            )
        elif obj_type == PLSQLType.TRIGGER:
            header = re.sub(
                r'(TRIGGER)\s+(\w+)',
                r'\1 <TRIGGER_NAME>',
                header,
                flags=re.IGNORECASE
            )
        
        return header.upper()
    
    def _normalize_type(self, type_str: str) -> str:
        """Normalize an Oracle data type"""
        type_str = type_str.strip().upper()
        
        # Handle size specifications - normalize VARCHAR2(n) to VARCHAR2(<SIZE>)
        type_str = re.sub(r'(VARCHAR2|CHAR|NUMBER|RAW)\s*\([^)]+\)', r'\1(<SIZE>)', type_str)
        
        # Handle common aliases
        type_aliases = {
            'INT': 'INTEGER',
            'VARCHAR': 'VARCHAR2',
        }
        
        for alias, canonical in type_aliases.items():
            if type_str == alias:
                return canonical
        
        return type_str
    
    def normalize_declarations(self, declarations: List[Tuple[str, str, str]]) -> str:
        """Normalize declarations and return as string"""
        normalized_decls = []
        
        for decl_type, name, definition in declarations:
            if decl_type == 'cursor':
                abstract_name = self.mapper.map_cursor(name)
                normalized_query = self._normalize_expression(definition)
                normalized_decls.append(f"{abstract_name} CURSOR IS {normalized_query}")
            elif decl_type == 'exception':
                abstract_name = self.mapper.map_exception(name)
                normalized_decls.append(f"{abstract_name} EXCEPTION")
            elif decl_type == 'variable':
                abstract_name = self.mapper.map_variable(name)
                normalized_def = self._normalize_variable_definition(definition)
                normalized_decls.append(f"{abstract_name} {normalized_def}")
            elif decl_type == 'constant':
                abstract_name = self.mapper.map_variable(name)
                normalized_def = self._normalize_variable_definition(definition)
                normalized_decls.append(f"{abstract_name} CONSTANT {normalized_def}")
            elif decl_type == 'record':
                abstract_name = self.mapper.map_variable(name)
                normalized_decls.append(f"{abstract_name} {definition.upper()}")
            elif decl_type == 'type':
                # Abstract the type name
                abstract_name = self.mapper.map_type(name)
                normalized_def = self._normalize_expression(definition)
                normalized_decls.append(f"TYPE {abstract_name} IS {normalized_def}")
            elif decl_type == 'subtype':
                # Abstract the subtype name
                abstract_name = self.mapper.map_type(name)
                normalized_def = self._normalize_expression(definition)
                normalized_decls.append(f"SUBTYPE {abstract_name} IS {normalized_def}")
        
        return '; '.join(normalized_decls)
    
    def _normalize_variable_definition(self, definition: str) -> str:
        """Normalize a variable definition (type and optional default value)"""
        # Split on := for default value
        if ':=' in definition:
            type_part, default_part = definition.split(':=', 1)
            type_normalized = self._normalize_type_reference(type_part.strip())
            default_normalized = self._normalize_expression(default_part.strip())
            return f"{type_normalized} := {default_normalized}"
        else:
            return self._normalize_type_reference(definition)
    
    def _normalize_type_reference(self, type_str: str) -> str:
        """Normalize a type reference, replacing user-defined types with abstractions"""
        type_str = type_str.strip()
        
        # Check if this is a user-defined type (exists in declared_types)
        if type_str.lower() in self.declared_types:
            return self.mapper.map_type(type_str)
        
        # Otherwise use standard type normalization
        return self._normalize_type(type_str)
    
    def normalize_body(self, body: str) -> str:
        """Normalize the body section"""
        if not body.strip():
            return ""
        
        # First, extract and map loop variables
        self._extract_loop_variables(body)
        
        # Extract and map variables from nested DECLARE blocks
        self._extract_nested_declarations(body)
        
        return self._normalize_expression(body)
    
    def _extract_nested_declarations(self, body: str) -> None:
        """Extract and map variables declared in nested DECLARE blocks"""
        # Find all nested DECLARE...BEGIN blocks
        declare_pattern = re.compile(
            r'\bDECLARE\s+(.*?)\s*\bBEGIN\b',
            re.IGNORECASE | re.DOTALL
        )
        
        for match in declare_pattern.finditer(body):
            decl_section = match.group(1)
            # Extract declarations from this nested block
            nested_decls = self.extract_declarations(decl_section)
            for decl_type, name, _ in nested_decls:
                if decl_type == 'cursor':
                    self.mapper.map_cursor(name)
                elif decl_type == 'exception':
                    self.mapper.map_exception(name)
                elif decl_type in ('variable', 'constant', 'record'):
                    self.mapper.map_variable(name)
                elif decl_type == 'type':
                    self.mapper.map_type(name)
    
    def _extract_loop_variables(self, body: str) -> None:
        """Extract loop variables from FOR loops and add them to the mapper"""
        # Match FOR loop_var IN ... LOOP patterns
        for_matches = re.finditer(r'\bFOR\s+(\w+)\s+IN\b', body, re.IGNORECASE)
        for match in for_matches:
            loop_var = match.group(1)
            if not self.mapper.is_system_identifier(loop_var):
                self.declared_loop_vars.add(loop_var.lower())
                self.mapper.map_loop_var(loop_var)
    
    def normalize_exception_section(self, exception_section: str) -> str:
        """Normalize the exception handling section"""
        if not exception_section.strip():
            return ""
        
        return self._normalize_expression(exception_section)
    
    def _normalize_expression(self, expr: str) -> str:
        """Normalize an expression, abstracting identifiers"""
        if not expr:
            return expr
        
        # Normalize whitespace
        expr = re.sub(r'\s+', ' ', expr).strip()
        
        # Protect string literals
        string_literals = {}
        string_counter = [0]
        
        def save_string(match):
            placeholder = f"__STRING_{string_counter[0]}__"
            string_literals[placeholder] = match.group(0)
            string_counter[0] += 1
            return placeholder
        
        # Protect single-quoted strings
        expr = re.sub(r"'(?:[^']|'')*'", save_string, expr)
        
        # Protect quoted identifiers
        quoted_identifiers = {}
        quoted_counter = [0]
        
        def save_quoted(match):
            name = match.group(1)
            placeholder = f"__QUOTED_{quoted_counter[0]}__"
            quoted_identifiers[placeholder] = f'"{name}"'
            quoted_counter[0] += 1
            return placeholder
        
        expr = re.sub(r'"([^"]+)"', save_quoted, expr)
        
        # Replace :NEW and :OLD references (keep them as-is, just normalize case)
        expr = re.sub(r':NEW\.(\w+)', lambda m: f':NEW.{m.group(1)}', expr, flags=re.IGNORECASE)
        expr = re.sub(r':OLD\.(\w+)', lambda m: f':OLD.{m.group(1)}', expr, flags=re.IGNORECASE)
        
        # Replace identifiers with their abstract mappings
        def replace_identifier(match):
            name = match.group(0)
            
            # Don't replace our placeholders
            if name.startswith('__') and name.endswith('__'):
                return name
            
            # Check if it's a system identifier or keyword
            if self.mapper.is_system_identifier(name):
                return name.upper()
            
            # Check if we have a mapping
            mapped = self.mapper.get_mapped(name)
            if mapped:
                return mapped
            
            # Keep unmapped identifiers (likely table/column names)
            return name.upper()
        
        # Replace identifiers (word characters not starting with digit)
        expr = re.sub(r'\b(?![0-9])(\w+)\b', replace_identifier, expr)
        
        # Normalize operators
        expr = re.sub(r'\s*:=\s*', ' := ', expr)
        expr = re.sub(r'\s*=\s*', ' = ', expr)
        expr = re.sub(r'\s*<>\s*', ' <> ', expr)
        expr = re.sub(r'\s*!=\s*', ' <> ', expr)  # Normalize != to <>
        expr = re.sub(r'\s*>=\s*', ' >= ', expr)
        expr = re.sub(r'\s*<=\s*', ' <= ', expr)
        expr = re.sub(r'\s*>\s*(?!>)', ' > ', expr)  # Avoid >> 
        expr = re.sub(r'(?<!<)\s*<\s*(?!<|>)', ' < ', expr)  # Avoid << and <>
        expr = re.sub(r'\|\|', ' || ', expr)
        
        # Normalize punctuation
        expr = re.sub(r'\s*;\s*', '; ', expr)
        expr = re.sub(r'\s*,\s*', ', ', expr)
        expr = re.sub(r'\s*\(\s*', '(', expr)
        expr = re.sub(r'\s*\)\s*', ')', expr)
        
        # Restore quoted identifiers
        for placeholder, original in quoted_identifiers.items():
            expr = expr.replace(placeholder, original)
        
        # Restore string literals
        for placeholder, original in string_literals.items():
            expr = expr.replace(placeholder, original)
        
        # Clean up multiple spaces
        expr = re.sub(r'\s+', ' ', expr).strip()
        
        return expr
    
    def normalize_trigger_timing(self, timing_event: str) -> str:
        """Normalize trigger timing and event clause"""
        timing_event = re.sub(r'\s+', ' ', timing_event).strip().upper()
        
        # Normalize BEFORE/AFTER/INSTEAD OF
        timing_event = re.sub(r'\bBEFORE\b', 'BEFORE', timing_event)
        timing_event = re.sub(r'\bAFTER\b', 'AFTER', timing_event)
        timing_event = re.sub(r'\bINSTEAD\s+OF\b', 'INSTEAD OF', timing_event)
        
        # Normalize events
        timing_event = re.sub(r'\bINSERT\b', 'INSERT', timing_event)
        timing_event = re.sub(r'\bUPDATE\b', 'UPDATE', timing_event)
        timing_event = re.sub(r'\bDELETE\b', 'DELETE', timing_event)
        
        # Normalize FOR EACH ROW/STATEMENT
        timing_event = re.sub(r'FOR\s+EACH\s+ROW', 'FOR EACH ROW', timing_event)
        timing_event = re.sub(r'FOR\s+EACH\s+STATEMENT', 'FOR EACH STATEMENT', timing_event)
        
        return timing_event
    
    def normalize(self, code: str) -> str:
        """
        Main normalization method.
        Returns normalized code representation.
        """
        plsql_type = get_plsql_type(code)
        
        if plsql_type == PLSQLType.TRIGGER:
            return self._normalize_trigger(code)
        elif plsql_type == PLSQLType.PROCEDURE:
            return self._normalize_procedure(code)
        elif plsql_type == PLSQLType.FUNCTION:
            return self._normalize_function(code)
        else:
            return self._normalize_generic(code)
    
    def _normalize_procedure(self, code: str) -> str:
        """Normalize a procedure"""
        header, declarations, body, exception_section = self.extract_body_parts(code)
        
        # First extract declarations to build identifier mappings
        decls = self.extract_declarations(declarations)
        for decl_type, name, _ in decls:
            if decl_type == 'cursor':
                self.mapper.map_cursor(name)
            elif decl_type == 'exception':
                self.mapper.map_exception(name)
            elif decl_type in ('variable', 'constant', 'record'):
                self.mapper.map_variable(name)
        
        normalized_header = self.normalize_header(header, PLSQLType.PROCEDURE)
        normalized_decls = self.normalize_declarations(decls)
        normalized_body = self.normalize_body(body)
        normalized_exc = self.normalize_exception_section(exception_section)
        
        result_parts = [normalized_header]
        if normalized_decls:
            result_parts.append(f"DECLARE {normalized_decls}")
        result_parts.append(f"BEGIN {normalized_body}")
        if normalized_exc:
            result_parts.append(f"EXCEPTION {normalized_exc}")
        result_parts.append("END")
        
        return ' '.join(result_parts)
    
    def _normalize_function(self, code: str) -> str:
        """Normalize a function"""
        header, declarations, body, exception_section = self.extract_body_parts(code)
        
        # First extract declarations to build identifier mappings
        decls = self.extract_declarations(declarations)
        for decl_type, name, _ in decls:
            if decl_type == 'cursor':
                self.mapper.map_cursor(name)
            elif decl_type == 'exception':
                self.mapper.map_exception(name)
            elif decl_type in ('variable', 'constant', 'record'):
                self.mapper.map_variable(name)
        
        normalized_header = self.normalize_header(header, PLSQLType.FUNCTION)
        normalized_decls = self.normalize_declarations(decls)
        normalized_body = self.normalize_body(body)
        normalized_exc = self.normalize_exception_section(exception_section)
        
        result_parts = [normalized_header]
        if normalized_decls:
            result_parts.append(f"DECLARE {normalized_decls}")
        result_parts.append(f"BEGIN {normalized_body}")
        if normalized_exc:
            result_parts.append(f"EXCEPTION {normalized_exc}")
        result_parts.append("END")
        
        return ' '.join(result_parts)
    
    def _normalize_trigger(self, code: str) -> str:
        """Normalize a trigger"""
        header, timing_event, declarations, body, exception_section = self.extract_trigger_parts(code)
        
        # First extract declarations to build identifier mappings
        decls = self.extract_declarations(declarations)
        for decl_type, name, _ in decls:
            if decl_type == 'cursor':
                self.mapper.map_cursor(name)
            elif decl_type == 'exception':
                self.mapper.map_exception(name)
            elif decl_type in ('variable', 'constant', 'record'):
                self.mapper.map_variable(name)
        
        normalized_header = self.normalize_header(header, PLSQLType.TRIGGER)
        normalized_timing = self.normalize_trigger_timing(timing_event)
        normalized_decls = self.normalize_declarations(decls)
        normalized_body = self.normalize_body(body)
        normalized_exc = self.normalize_exception_section(exception_section)
        
        result_parts = [normalized_header, normalized_timing]
        if normalized_decls:
            result_parts.append(f"DECLARE {normalized_decls}")
        result_parts.append(f"BEGIN {normalized_body}")
        if normalized_exc:
            result_parts.append(f"EXCEPTION {normalized_exc}")
        result_parts.append("END")
        
        return ' '.join(result_parts)
    
    def _normalize_generic(self, code: str) -> str:
        """Normalize generic PL/SQL code"""
        return self._normalize_expression(code)


class OraclePLSQLASTBuilder:
    """
    Builds an Abstract Syntax Tree from normalized Oracle PL/SQL code.
    """
    
    def __init__(self):
        pass
    
    def build(self, normalized_code: str) -> ASTNode:
        """Build AST from normalized code"""
        # Determine statement type
        if re.match(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE', normalized_code, re.IGNORECASE):
            return self._build_procedure(normalized_code)
        elif re.match(r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION', normalized_code, re.IGNORECASE):
            return self._build_function(normalized_code)
        elif re.match(r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER', normalized_code, re.IGNORECASE):
            return self._build_trigger(normalized_code)
        else:
            return ASTNode('UNKNOWN', value=normalized_code)
    
    def _build_procedure(self, code: str) -> ASTNode:
        """Build AST for procedure"""
        children = []
        
        # Extract header
        header_match = re.match(r'(CREATE\s+OR\s+REPLACE\s+PROCEDURE\s+<PROC_NAME>(?:\([^)]*\))?(?:\s+IS|\s+AS)?)', code, re.IGNORECASE)
        if header_match:
            children.append(self._build_header(header_match.group(1), 'PROCEDURE'))
            rest = code[header_match.end():].strip()
        else:
            rest = code
        
        # Extract DECLARE section
        declare_match = re.search(r'DECLARE\s+(.+?)(?=\s+BEGIN)', rest, re.IGNORECASE)
        if declare_match:
            children.append(self._build_declarations(declare_match.group(1)))
        
        # Extract body
        begin_match = re.search(r'BEGIN\s+(.+?)(?=\s+EXCEPTION|\s+END)', rest, re.IGNORECASE | re.DOTALL)
        if begin_match:
            children.append(self._build_body(begin_match.group(1)))
        
        # Extract exception section
        exc_match = re.search(r'EXCEPTION\s+(.+?)(?=\s+END)', rest, re.IGNORECASE | re.DOTALL)
        if exc_match:
            children.append(self._build_exception(exc_match.group(1)))
        
        return ASTNode('PROCEDURE', children=children)
    
    def _build_function(self, code: str) -> ASTNode:
        """Build AST for function"""
        children = []
        
        # Extract header
        header_match = re.match(
            r'(CREATE\s+OR\s+REPLACE\s+FUNCTION\s+<FUNC_NAME>(?:\([^)]*\))?\s+RETURN\s+\S+(?:\s+IS|\s+AS)?)',
            code, re.IGNORECASE
        )
        if header_match:
            children.append(self._build_header(header_match.group(1), 'FUNCTION'))
            rest = code[header_match.end():].strip()
        else:
            rest = code
        
        # Extract DECLARE section
        declare_match = re.search(r'DECLARE\s+(.+?)(?=\s+BEGIN)', rest, re.IGNORECASE)
        if declare_match:
            children.append(self._build_declarations(declare_match.group(1)))
        
        # Extract body
        begin_match = re.search(r'BEGIN\s+(.+?)(?=\s+EXCEPTION|\s+END)', rest, re.IGNORECASE | re.DOTALL)
        if begin_match:
            children.append(self._build_body(begin_match.group(1)))
        
        # Extract exception section
        exc_match = re.search(r'EXCEPTION\s+(.+?)(?=\s+END)', rest, re.IGNORECASE | re.DOTALL)
        if exc_match:
            children.append(self._build_exception(exc_match.group(1)))
        
        return ASTNode('FUNCTION', children=children)
    
    def _build_trigger(self, code: str) -> ASTNode:
        """Build AST for trigger"""
        children = []
        
        # Extract header
        header_match = re.match(r'(CREATE\s+OR\s+REPLACE\s+TRIGGER\s+<TRIGGER_NAME>)', code, re.IGNORECASE)
        if header_match:
            children.append(ASTNode('HEADER', value=header_match.group(1)))
        
        # Extract timing/event
        timing_match = re.search(
            r'((?:BEFORE|AFTER|INSTEAD\s+OF)\s+(?:INSERT|UPDATE|DELETE)(?:\s+OR\s+(?:INSERT|UPDATE|DELETE))*\s+ON\s+\w+(?:\s+FOR\s+EACH\s+(?:ROW|STATEMENT))?)',
            code, re.IGNORECASE
        )
        if timing_match:
            children.append(ASTNode('TIMING_EVENT', value=timing_match.group(1)))
        
        # Extract DECLARE section
        declare_match = re.search(r'DECLARE\s+(.+?)(?=\s+BEGIN)', code, re.IGNORECASE)
        if declare_match:
            children.append(self._build_declarations(declare_match.group(1)))
        
        # Extract body
        begin_match = re.search(r'BEGIN\s+(.+?)(?=\s+EXCEPTION|\s+END)', code, re.IGNORECASE | re.DOTALL)
        if begin_match:
            children.append(self._build_body(begin_match.group(1)))
        
        # Extract exception section
        exc_match = re.search(r'EXCEPTION\s+(.+?)(?=\s+END)', code, re.IGNORECASE | re.DOTALL)
        if exc_match:
            children.append(self._build_exception(exc_match.group(1)))
        
        return ASTNode('TRIGGER', children=children)
    
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
        
        # Extract return type (for functions)
        return_match = re.search(r'RETURN\s+(\S+)', header, re.IGNORECASE)
        if return_match:
            children.append(ASTNode('RETURNS', value=return_match.group(1)))
        
        return ASTNode('HEADER', value=obj_type, children=children)
    
    def _build_declarations(self, decl_content: str) -> ASTNode:
        """Build AST for declarations"""
        nodes = []
        decls = [d.strip() for d in decl_content.split(';') if d.strip()]
        
        for decl in decls:
            upper_decl = decl.upper()
            if 'CURSOR' in upper_decl:
                nodes.append(ASTNode('CURSOR_DECL', value=decl))
            elif 'EXCEPTION' in upper_decl:
                nodes.append(ASTNode('EXCEPTION_DECL', value=decl))
            elif '%ROWTYPE' in upper_decl:
                nodes.append(ASTNode('ROWTYPE_DECL', value=decl))
            elif '%TYPE' in upper_decl:
                nodes.append(ASTNode('TYPE_DECL', value=decl))
            elif 'CONSTANT' in upper_decl:
                nodes.append(ASTNode('CONSTANT_DECL', value=decl))
            else:
                nodes.append(ASTNode('VAR_DECL', value=decl))
        
        return ASTNode('DECLARE', children=nodes)
    
    def _build_body(self, body_content: str) -> ASTNode:
        """Build AST for body"""
        nodes = self._parse_statements(body_content)
        return ASTNode('BODY', children=nodes)
    
    def _build_exception(self, exc_content: str) -> ASTNode:
        """Build AST for exception section"""
        nodes = []
        
        # Split by WHEN keyword
        handlers = re.split(r'\bWHEN\b', exc_content, flags=re.IGNORECASE)
        
        for handler in handlers:
            handler = handler.strip()
            if not handler:
                continue
            nodes.append(ASTNode('EXCEPTION_HANDLER', value=handler))
        
        return ASTNode('EXCEPTION', children=nodes)
    
    def _parse_statements(self, stmt_content: str) -> List[ASTNode]:
        """Parse statements into AST nodes using regex-based parsing"""
        nodes = []
        
        # Split content into statements by semicolon, handling nested structures
        statements = self._split_statements(stmt_content)
        
        for stmt_str in statements:
            stmt_str = stmt_str.strip()
            if not stmt_str:
                continue
            
            upper_stmt = stmt_str.upper()
            
            if upper_stmt.startswith('IF'):
                nodes.append(ASTNode('IF_STMT', value=stmt_str))
            elif upper_stmt.startswith('FOR') or upper_stmt.startswith('WHILE') or 'LOOP' in upper_stmt:
                nodes.append(ASTNode('LOOP_STMT', value=stmt_str))
            elif upper_stmt.startswith('CASE'):
                nodes.append(ASTNode('CASE_STMT', value=stmt_str))
            elif upper_stmt.startswith('SELECT'):
                nodes.append(ASTNode('SELECT_STMT', value=stmt_str))
            elif upper_stmt.startswith('INSERT'):
                nodes.append(ASTNode('INSERT_STMT', value=stmt_str))
            elif upper_stmt.startswith('UPDATE'):
                nodes.append(ASTNode('UPDATE_STMT', value=stmt_str))
            elif upper_stmt.startswith('DELETE'):
                nodes.append(ASTNode('DELETE_STMT', value=stmt_str))
            elif upper_stmt.startswith('MERGE'):
                nodes.append(ASTNode('MERGE_STMT', value=stmt_str))
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
            elif upper_stmt.startswith('CONTINUE'):
                nodes.append(ASTNode('CONTINUE_STMT', value=stmt_str))
            elif upper_stmt.startswith('COMMIT'):
                nodes.append(ASTNode('COMMIT_STMT', value=stmt_str))
            elif upper_stmt.startswith('ROLLBACK'):
                nodes.append(ASTNode('ROLLBACK_STMT', value=stmt_str))
            elif 'DBMS_OUTPUT' in upper_stmt or 'PUT_LINE' in upper_stmt:
                nodes.append(ASTNode('OUTPUT_STMT', value=stmt_str))
            elif ':=' in stmt_str:
                nodes.append(ASTNode('ASSIGN_STMT', value=stmt_str))
            else:
                nodes.append(ASTNode('STMT', value=stmt_str))
        
        return nodes
    
    def _split_statements(self, content: str) -> List[str]:
        """Split content into individual statements, handling nested blocks"""
        statements = []
        current = ""
        depth = 0
        in_string = False
        string_char = None
        
        i = 0
        content_upper = content.upper()
        
        while i < len(content):
            char = content[i]
            
            # Handle string literals
            if char == "'" and not in_string:
                in_string = True
                string_char = "'"
            elif char == "'" and in_string and string_char == "'":
                # Check for escaped quote
                if i + 1 < len(content) and content[i + 1] == "'":
                    current += char
                    i += 1
                else:
                    in_string = False
                    string_char = None
            
            if not in_string:
                # Track BEGIN/END, IF/END IF, LOOP/END LOOP, CASE/END CASE depth
                remaining = content_upper[i:]
                
                # Check for block openers
                if remaining.startswith('BEGIN') and (i == 0 or not content[i-1].isalnum()):
                    if i + 5 >= len(content) or not content[i+5].isalnum():
                        depth += 1
                elif remaining.startswith('CASE') and (i == 0 or not content[i-1].isalnum()):
                    if i + 4 >= len(content) or not content[i+4].isalnum():
                        depth += 1
                elif remaining.startswith('IF') and (i == 0 or not content[i-1].isalnum()):
                    if i + 2 >= len(content) or not content[i+2].isalnum():
                        # Check if followed by THEN to confirm it's control flow IF
                        if re.search(r'^IF\s+.+?\s+THEN\b', remaining, re.IGNORECASE):
                            depth += 1
                elif remaining.startswith('LOOP') and (i == 0 or not content[i-1].isalnum()):
                    if i + 4 >= len(content) or not content[i+4].isalnum():
                        depth += 1
                
                # Check for block closers
                if remaining.startswith('END IF') and (i == 0 or not content[i-1].isalnum()):
                    depth = max(0, depth - 1)
                    current += 'END IF'
                    i += 6
                    continue
                elif remaining.startswith('END LOOP') and (i == 0 or not content[i-1].isalnum()):
                    depth = max(0, depth - 1)
                    current += 'END LOOP'
                    i += 8
                    continue
                elif remaining.startswith('END CASE') and (i == 0 or not content[i-1].isalnum()):
                    depth = max(0, depth - 1)
                    current += 'END CASE'
                    i += 8
                    continue
                elif remaining.startswith('END') and (i == 0 or not content[i-1].isalnum()):
                    if i + 3 >= len(content) or not content[i+3].isalnum():
                        depth = max(0, depth - 1)
                
                # Statement separator
                if char == ';' and depth == 0:
                    if current.strip():
                        statements.append(current.strip())
                    current = ""
                    i += 1
                    continue
            
            current += char
            i += 1
        
        if current.strip():
            statements.append(current.strip())
        
        return statements


def compare_ast(node1: ASTNode, node2: ASTNode, debug: bool = False) -> bool:
    """
    Compare two AST nodes for semantic equivalence.
    """
    if debug:
        print(f"Comparing: {node1.node_type}({node1.value[:50] if node1.value else ''}) vs "
              f"{node2.node_type}({node2.value[:50] if node2.value else ''})")
    
    # Compare node types
    if node1.node_type != node2.node_type:
        if debug:
            print(f"  Node type mismatch: {node1.node_type} != {node2.node_type}")
        return False
    
    # Compare values
    if node1.value != node2.value:
        if debug:
            print(f"  Value mismatch")
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


def is_plsql_semantically_equivalent(code1: str, code2: str, debug: bool = False) -> bool:
    """
    Check if two Oracle PL/SQL code blocks are semantically equivalent.
    
    This function normalizes both code blocks and compares their AST structures,
    abstracting away differences in:
    - Whitespace and formatting
    - Variable names
    - Parameter names
    - Cursor names
    - Exception names
    - Non-semantic code variations
    
    Args:
        code1: First PL/SQL code block
        code2: Second PL/SQL code block
        debug: If True, print debug information
        
    Returns:
        True if the code blocks are semantically equivalent, False otherwise
    """
    try:
        # Check types match
        type1 = get_plsql_type(code1)
        type2 = get_plsql_type(code2)
        
        if debug:
            print(f"=== Oracle PL/SQL Semantic Equivalence Check ===")
            print(f"Code1 type: {type1}")
            print(f"Code2 type: {type2}")
        
        if type1 != type2:
            if debug:
                print(f"Type mismatch: {type1} != {type2}")
            return False
        
        # Normalize both code blocks
        normalizer1 = OraclePLSQLNormalizer()
        normalizer2 = OraclePLSQLNormalizer()
        
        normalized1 = normalizer1.normalize(code1)
        normalized2 = normalizer2.normalize(code2)
        
        if debug:
            print(f"\nNormalized Code 1:\n{normalized1}")
            print(f"\nNormalized Code 2:\n{normalized2}")
        
        # Build ASTs
        builder1 = OraclePLSQLASTBuilder()
        builder2 = OraclePLSQLASTBuilder()
        
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

def is_exact_match(plsql1: str, plsql2: str) -> bool:
    """
    Check semantic equivalence of Oracle PL/SQL code without debug output.
    """
    return is_plsql_semantically_equivalent(plsql1, plsql2, debug=False)


def debug_semantic_equivalence(plsql1: str, plsql2: str) -> bool:
    """
    Check semantic equivalence of Oracle PL/SQL code with verbose debug output.
    """
    return is_plsql_semantically_equivalent(plsql1, plsql2, debug=True)


def is_exact_match_hybrid(plsql1: str, plsql2: str, debug: bool = False) -> bool:
    """
    Legacy alias for backward compatibility.
    """
    return is_plsql_semantically_equivalent(plsql1, plsql2, debug=debug)


def debug_semantic_equivalence_ast(plsql1: str, plsql2: str) -> bool:
    """
    Legacy alias for backward compatibility.
    """
    return is_plsql_semantically_equivalent(plsql1, plsql2, debug=True)

# =============================================================================
# TEST CASES
# =============================================================================

def run_tests():
    """Run comprehensive test cases for the Oracle PL/SQL semantic equivalence checker"""
    
    print("=" * 80)
    print("ORACLE PL/SQL SEMANTIC EQUIVALENCE CHECKER - TEST SUITE")
    print("=" * 80)
    
    test_results = []
    
    # =========================================================================
    # PROCEDURE TESTS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PROCEDURE TESTS")
    print("=" * 60)
    
    # Test 1: Different variable names (should be True)
    print("\n[Test 1] Different variable names in procedure")
    proc1 = """CREATE OR REPLACE PROCEDURE update_customer(p_id IN NUMBER, p_name IN VARCHAR2)
IS
    v_count NUMBER;
    v_old_name VARCHAR2(255);
BEGIN
    SELECT COUNT(*) INTO v_count FROM customers WHERE customer_id = p_id;
    IF v_count > 0 THEN
        SELECT customer_name INTO v_old_name FROM customers WHERE customer_id = p_id;
        UPDATE customers SET customer_name = p_name WHERE customer_id = p_id;
    END IF;
END;"""
    
    proc2 = """CREATE OR REPLACE PROCEDURE update_customer(cust_id IN NUMBER, new_name IN VARCHAR2)
IS
    cnt NUMBER;
    old_name VARCHAR2(255);
BEGIN
    SELECT COUNT(*) INTO cnt FROM customers WHERE customer_id = cust_id;
    IF cnt > 0 THEN
        SELECT customer_name INTO old_name FROM customers WHERE customer_id = cust_id;
        UPDATE customers SET customer_name = new_name WHERE customer_id = cust_id;
    END IF;
END;"""
    
    result = is_plsql_semantically_equivalent(proc1, proc2)
    test_results.append(("Proc: Different variable names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 2: Different cursor names (should be True)
    print("\n[Test 2] Different cursor names")
    proc3 = """CREATE OR REPLACE PROCEDURE process_orders(p_status IN VARCHAR2)
IS
    CURSOR order_cursor IS SELECT * FROM orders WHERE status = p_status;
    v_order orders%ROWTYPE;
BEGIN
    OPEN order_cursor;
    LOOP
        FETCH order_cursor INTO v_order;
        EXIT WHEN order_cursor%NOTFOUND;
        UPDATE orders SET processed = 'Y' WHERE order_id = v_order.order_id;
    END LOOP;
    CLOSE order_cursor;
END;"""
    
    proc4 = """CREATE OR REPLACE PROCEDURE process_orders(order_status IN VARCHAR2)
IS
    CURSOR c_orders IS SELECT * FROM orders WHERE status = order_status;
    rec_order orders%ROWTYPE;
BEGIN
    OPEN c_orders;
    LOOP
        FETCH c_orders INTO rec_order;
        EXIT WHEN c_orders%NOTFOUND;
        UPDATE orders SET processed = 'Y' WHERE order_id = rec_order.order_id;
    END LOOP;
    CLOSE c_orders;
END;"""
    
    result = is_plsql_semantically_equivalent(proc3, proc4)
    test_results.append(("Proc: Different cursor names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 3: Different whitespace and formatting (should be True)
    print("\n[Test 3] Different whitespace and formatting")
    proc5 = """CREATE OR REPLACE PROCEDURE simple_proc(p_id NUMBER) IS v_name VARCHAR2(100); BEGIN SELECT name INTO v_name FROM users WHERE id=p_id; DBMS_OUTPUT.PUT_LINE(v_name); END;"""
    
    proc6 = """CREATE   OR   REPLACE   PROCEDURE   simple_proc(
        p_id   NUMBER
    )
    IS
        v_name   VARCHAR2(100);
    BEGIN
        SELECT   name   INTO   v_name
        FROM     users
        WHERE    id = p_id;
        
        DBMS_OUTPUT.PUT_LINE(v_name);
    END;"""
    
    result = is_plsql_semantically_equivalent(proc5, proc6)
    test_results.append(("Proc: Different whitespace/formatting", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 4: Different logic (should be False)
    print("\n[Test 4] Different logic")
    proc7 = """CREATE OR REPLACE PROCEDURE calc_proc(p_val IN NUMBER)
IS
    v_result NUMBER;
BEGIN
    v_result := p_val * 2;
    INSERT INTO results VALUES(v_result);
END;"""
    
    proc8 = """CREATE OR REPLACE PROCEDURE calc_proc(p_val IN NUMBER)
IS
    v_result NUMBER;
BEGIN
    v_result := p_val * 3;
    INSERT INTO results VALUES(v_result);
END;"""
    
    result = is_plsql_semantically_equivalent(proc7, proc8)
    test_results.append(("Proc: Different logic", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 5: Procedure with exception handling
    print("\n[Test 5] Procedure with exception handling")
    proc9 = """CREATE OR REPLACE PROCEDURE proc_with_exception(p_customer_id IN NUMBER)
IS
    v_name VARCHAR2(255);
BEGIN
    SELECT customer_name INTO v_name FROM customers WHERE customer_id = p_customer_id;
    DBMS_OUTPUT.PUT_LINE(v_name);
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('Customer not found');
    WHEN OTHERS THEN
        RAISE;
END;"""
    
    proc10 = """CREATE OR REPLACE PROCEDURE proc_with_exception(cust_id IN NUMBER)
IS
    cust_name VARCHAR2(255);
BEGIN
    SELECT customer_name INTO cust_name FROM customers WHERE customer_id = cust_id;
    DBMS_OUTPUT.PUT_LINE(cust_name);
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('Customer not found');
    WHEN OTHERS THEN
        RAISE;
END;"""
    
    result = is_plsql_semantically_equivalent(proc9, proc10)
    test_results.append(("Proc: With exception handling", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # =========================================================================
    # FUNCTION TESTS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("FUNCTION TESTS")
    print("=" * 60)
    
    # Test 6: Function with different variable names
    print("\n[Test 6] Function with different variable names")
    func1 = """CREATE OR REPLACE FUNCTION get_director_gender_by_series(series_id NUMBER) RETURN VARCHAR2 IS
  director_gender VARCHAR2(255);
BEGIN
  SELECT d.GENDER INTO director_gender FROM DIRECTOR d JOIN DIRECTED_BY db ON d.DID = db.DID WHERE db.MSID = series_id;
  IF director_gender IS NULL THEN
    director_gender := 'Unknown';
  END IF;
  RETURN director_gender;
END;"""
    
    func2 = """CREATE OR REPLACE FUNCTION get_director_gender_by_series(p_series_id NUMBER) RETURN VARCHAR2 IS
  v_gender VARCHAR2(255);
BEGIN
  SELECT d.GENDER INTO v_gender FROM DIRECTOR d JOIN DIRECTED_BY db ON d.DID = db.DID WHERE db.MSID = p_series_id;
  IF v_gender IS NULL THEN
    v_gender := 'Unknown';
  END IF;
  RETURN v_gender;
END;"""
    
    result = is_plsql_semantically_equivalent(func1, func2)
    test_results.append(("Func: Different variable names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 7: Function with different return type (should be False)
    print("\n[Test 7] Function with different return type")
    func3 = """CREATE OR REPLACE FUNCTION get_count(p_id NUMBER) RETURN NUMBER IS
    v_count NUMBER;
BEGIN
    SELECT COUNT(*) INTO v_count FROM items WHERE id = p_id;
    RETURN v_count;
END;"""
    
    func4 = """CREATE OR REPLACE FUNCTION get_count(p_id NUMBER) RETURN VARCHAR2 IS
    v_count VARCHAR2(20);
BEGIN
    SELECT COUNT(*) INTO v_count FROM items WHERE id = p_id;
    RETURN v_count;
END;"""
    
    result = is_plsql_semantically_equivalent(func3, func4)
    test_results.append(("Func: Different return type", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 8: Function with complex logic
    print("\n[Test 8] Function with complex CASE expression")
    func5 = """CREATE OR REPLACE FUNCTION get_grade(p_score NUMBER) RETURN VARCHAR2 IS
    v_grade VARCHAR2(2);
BEGIN
    v_grade := CASE
        WHEN p_score >= 90 THEN 'A'
        WHEN p_score >= 80 THEN 'B'
        WHEN p_score >= 70 THEN 'C'
        ELSE 'F'
    END;
    RETURN v_grade;
END;"""
    
    func6 = """CREATE OR REPLACE FUNCTION get_grade(score_val NUMBER) RETURN VARCHAR2 IS
    grade_result VARCHAR2(2);
BEGIN
    grade_result := CASE
        WHEN score_val >= 90 THEN 'A'
        WHEN score_val >= 80 THEN 'B'
        WHEN score_val >= 70 THEN 'C'
        ELSE 'F'
    END;
    RETURN grade_result;
END;"""
    
    result = is_plsql_semantically_equivalent(func5, func6)
    test_results.append(("Func: Complex CASE expression", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # =========================================================================
    # TRIGGER TESTS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("TRIGGER TESTS")
    print("=" * 60)
    
    # Test 9: Trigger with different variable names
    print("\n[Test 9] Trigger with different variable names")
    trigger1 = """CREATE OR REPLACE TRIGGER trg_leader_year_validation
BEFORE INSERT OR UPDATE ON CLUB_LEADER
FOR EACH ROW
DECLARE
  v_max_ranking NUMBER;
BEGIN
  SELECT MAX(OVERALL_RANKING) INTO v_max_ranking FROM CLUB;
  IF :NEW.YEAR_JOIN > '2019' THEN
    UPDATE MEMBER SET AGE = AGE + 1 WHERE MEMBER_ID = :NEW.MEMBER_ID;
  ELSE
    UPDATE MEMBER SET AGE = AGE - 1 WHERE MEMBER_ID = :NEW.MEMBER_ID;
  END IF;
END;"""
    
    trigger2 = """CREATE OR REPLACE TRIGGER trg_validate_leader_year
BEFORE INSERT OR UPDATE ON CLUB_LEADER
FOR EACH ROW
DECLARE
  max_rank NUMBER;
BEGIN
  SELECT MAX(OVERALL_RANKING) INTO max_rank FROM CLUB;
  IF :NEW.YEAR_JOIN > '2019' THEN
    UPDATE MEMBER SET AGE = AGE + 1 WHERE MEMBER_ID = :NEW.MEMBER_ID;
  ELSE
    UPDATE MEMBER SET AGE = AGE - 1 WHERE MEMBER_ID = :NEW.MEMBER_ID;
  END IF;
END;"""
    
    result = is_plsql_semantically_equivalent(trigger1, trigger2)
    test_results.append(("Trigger: Different variable names", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 10: Trigger with different timing (BEFORE vs AFTER - should be False)
    print("\n[Test 10] Trigger with different timing")
    trigger3 = """CREATE OR REPLACE TRIGGER audit_trigger
BEFORE INSERT ON users
FOR EACH ROW
BEGIN
    INSERT INTO audit_log VALUES(:NEW.id, SYSDATE);
END;"""
    
    trigger4 = """CREATE OR REPLACE TRIGGER audit_trigger
AFTER INSERT ON users
FOR EACH ROW
BEGIN
    INSERT INTO audit_log VALUES(:NEW.id, SYSDATE);
END;"""
    
    result = is_plsql_semantically_equivalent(trigger3, trigger4)
    test_results.append(("Trigger: Different timing (BEFORE/AFTER)", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 11: Trigger with different events (INSERT vs UPDATE - should be False)
    print("\n[Test 11] Trigger with different events")
    trigger5 = """CREATE OR REPLACE TRIGGER data_trigger
BEFORE INSERT ON data_table
FOR EACH ROW
BEGIN
    :NEW.created_date := SYSDATE;
END;"""
    
    trigger6 = """CREATE OR REPLACE TRIGGER data_trigger
BEFORE UPDATE ON data_table
FOR EACH ROW
BEGIN
    :NEW.created_date := SYSDATE;
END;"""
    
    result = is_plsql_semantically_equivalent(trigger5, trigger6)
    test_results.append(("Trigger: Different events (INSERT/UPDATE)", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # =========================================================================
    # EDGE CASES
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("EDGE CASE TESTS")
    print("=" * 60)
    
    # Test 12: Procedure vs Function (should be False)
    print("\n[Test 12] Procedure vs Function")
    proc_code = """CREATE OR REPLACE PROCEDURE do_something(p_id NUMBER)
IS
BEGIN
    UPDATE table1 SET col = 1 WHERE id = p_id;
END;"""
    
    func_code = """CREATE OR REPLACE FUNCTION do_something(p_id NUMBER) RETURN NUMBER
IS
BEGIN
    UPDATE table1 SET col = 1 WHERE id = p_id;
    RETURN 1;
END;"""
    
    result = is_plsql_semantically_equivalent(proc_code, func_code)
    test_results.append(("Edge: Procedure vs Function", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 13: Different number of parameters (should be False)
    print("\n[Test 13] Different number of parameters")
    params1 = """CREATE OR REPLACE PROCEDURE test_params(a NUMBER, b NUMBER)
IS
BEGIN
    DBMS_OUTPUT.PUT_LINE(a);
END;"""
    
    params2 = """CREATE OR REPLACE PROCEDURE test_params(x NUMBER)
IS
BEGIN
    DBMS_OUTPUT.PUT_LINE(x);
END;"""
    
    result = is_plsql_semantically_equivalent(params1, params2)
    test_results.append(("Edge: Different number of parameters", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 14: Same code with newline escapes
    print("\n[Test 14] Code with newline escapes")
    escaped1 = "CREATE OR REPLACE PROCEDURE test_proc(p_id NUMBER)\\nIS\\n    v_name VARCHAR2(100);\\nBEGIN\\n    SELECT name INTO v_name FROM users WHERE id = p_id;\\nEND;"
    escaped2 = """CREATE OR REPLACE PROCEDURE test_proc(param_id NUMBER)
IS
    user_name VARCHAR2(100);
BEGIN
    SELECT name INTO user_name FROM users WHERE id = param_id;
END;"""
    
    result = is_plsql_semantically_equivalent(escaped1, escaped2)
    test_results.append(("Edge: Code with newline escapes", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 15: Complex procedure from user example
    print("\n[Test 15] Complex procedure with multiple conditions")
    complex1 = """CREATE OR REPLACE PROCEDURE proc_process_customer_data(p_customer_id IN NUMBER, p_new_name IN VARCHAR2, p_new_phone IN VARCHAR2)
IS
    v_current_name VARCHAR2(255);
    v_current_phone VARCHAR2(255);
    v_truncated_phone VARCHAR2(255);
BEGIN
    v_truncated_phone := SUBSTRB(p_new_phone, 1, 255);
    
    SELECT CUSTOMER_NAME, CUSTOMER_PHONE INTO v_current_name, v_current_phone
    FROM CUSTOMERS WHERE CUSTOMER_ID = p_customer_id;
    
    IF (v_current_name != p_new_name OR (v_current_name IS NULL AND p_new_name IS NOT NULL) OR (v_current_name IS NOT NULL AND p_new_name IS NULL))
       AND (v_current_phone != v_truncated_phone OR (v_current_phone IS NULL AND v_truncated_phone IS NOT NULL) OR (v_current_phone IS NOT NULL AND v_truncated_phone IS NULL)) THEN
        UPDATE CUSTOMERS
        SET CUSTOMER_NAME = p_new_name,
            CUSTOMER_PHONE = v_truncated_phone
        WHERE CUSTOMER_ID = p_customer_id;
    ELSIF v_current_name != p_new_name OR (v_current_name IS NULL AND p_new_name IS NOT NULL) OR (v_current_name IS NOT NULL AND p_new_name IS NULL) THEN
        UPDATE CUSTOMERS
        SET CUSTOMER_NAME = p_new_name
        WHERE CUSTOMER_ID = p_customer_id;
    ELSIF v_current_phone != v_truncated_phone OR (v_current_phone IS NULL AND v_truncated_phone IS NOT NULL) OR (v_current_phone IS NOT NULL AND v_truncated_phone IS NULL) THEN
        UPDATE CUSTOMERS
        SET CUSTOMER_PHONE = v_truncated_phone
        WHERE CUSTOMER_ID = p_customer_id;
    END IF;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('Customer ID ' || p_customer_id || ' not found.');
    WHEN OTHERS THEN
        RAISE;
END;"""
    
    complex2 = """CREATE OR REPLACE PROCEDURE proc_process_customer_data(cust_id IN NUMBER, new_cust_name IN VARCHAR2, new_phone_num IN VARCHAR2)
IS
    curr_name VARCHAR2(255);
    curr_phone VARCHAR2(255);
    trunc_phone VARCHAR2(255);
BEGIN
    trunc_phone := SUBSTRB(new_phone_num, 1, 255);
    
    SELECT CUSTOMER_NAME, CUSTOMER_PHONE INTO curr_name, curr_phone
    FROM CUSTOMERS WHERE CUSTOMER_ID = cust_id;
    
    IF (curr_name != new_cust_name OR (curr_name IS NULL AND new_cust_name IS NOT NULL) OR (curr_name IS NOT NULL AND new_cust_name IS NULL))
       AND (curr_phone != trunc_phone OR (curr_phone IS NULL AND trunc_phone IS NOT NULL) OR (curr_phone IS NOT NULL AND trunc_phone IS NULL)) THEN
        UPDATE CUSTOMERS
        SET CUSTOMER_NAME = new_cust_name,
            CUSTOMER_PHONE = trunc_phone
        WHERE CUSTOMER_ID = cust_id;
    ELSIF curr_name != new_cust_name OR (curr_name IS NULL AND new_cust_name IS NOT NULL) OR (curr_name IS NOT NULL AND new_cust_name IS NULL) THEN
        UPDATE CUSTOMERS
        SET CUSTOMER_NAME = new_cust_name
        WHERE CUSTOMER_ID = cust_id;
    ELSIF curr_phone != trunc_phone OR (curr_phone IS NULL AND trunc_phone IS NOT NULL) OR (curr_phone IS NOT NULL AND trunc_phone IS NULL) THEN
        UPDATE CUSTOMERS
        SET CUSTOMER_PHONE = trunc_phone
        WHERE CUSTOMER_ID = cust_id;
    END IF;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('Customer ID ' || cust_id || ' not found.');
    WHEN OTHERS THEN
        RAISE;
END;"""
    
    result = is_plsql_semantically_equivalent(complex1, complex2)
    test_results.append(("Edge: Complex procedure with conditions", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 16: IN OUT parameter mode
    print("\n[Test 16] IN OUT parameter mode")
    inout1 = """CREATE OR REPLACE PROCEDURE swap_values(p_a IN OUT NUMBER, p_b IN OUT NUMBER)
IS
    v_temp NUMBER;
BEGIN
    v_temp := p_a;
    p_a := p_b;
    p_b := v_temp;
END;"""
    
    inout2 = """CREATE OR REPLACE PROCEDURE swap_values(x IN OUT NUMBER, y IN OUT NUMBER)
IS
    tmp NUMBER;
BEGIN
    tmp := x;
    x := y;
    y := tmp;
END;"""
    
    result = is_plsql_semantically_equivalent(inout1, inout2)
    test_results.append(("Edge: IN OUT parameter mode", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 17: BULK COLLECT usage
    print("\n[Test 17] BULK COLLECT usage")
    bulk1 = """CREATE OR REPLACE PROCEDURE get_all_names(p_dept_id IN NUMBER)
IS
    TYPE name_table IS TABLE OF VARCHAR2(100);
    v_names name_table;
BEGIN
    SELECT employee_name BULK COLLECT INTO v_names FROM employees WHERE department_id = p_dept_id;
    FOR i IN 1..v_names.COUNT LOOP
        DBMS_OUTPUT.PUT_LINE(v_names(i));
    END LOOP;
END;"""
    
    bulk2 = """CREATE OR REPLACE PROCEDURE get_all_names(dept_id IN NUMBER)
IS
    TYPE t_names IS TABLE OF VARCHAR2(100);
    l_names t_names;
BEGIN
    SELECT employee_name BULK COLLECT INTO l_names FROM employees WHERE department_id = dept_id;
    FOR idx IN 1..l_names.COUNT LOOP
        DBMS_OUTPUT.PUT_LINE(l_names(idx));
    END LOOP;
END;"""
    
    result = is_plsql_semantically_equivalent(bulk1, bulk2)
    test_results.append(("Edge: BULK COLLECT usage", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 18: Cursor FOR loop
    print("\n[Test 18] Cursor FOR loop")
    cursor_for1 = """CREATE OR REPLACE PROCEDURE process_employees(p_dept NUMBER)
IS
    CURSOR emp_cursor IS SELECT employee_id, employee_name FROM employees WHERE dept_id = p_dept;
BEGIN
    FOR emp_rec IN emp_cursor LOOP
        DBMS_OUTPUT.PUT_LINE(emp_rec.employee_name);
    END LOOP;
END;"""
    
    cursor_for2 = """CREATE OR REPLACE PROCEDURE process_employees(department_id NUMBER)
IS
    CURSOR c_emps IS SELECT employee_id, employee_name FROM employees WHERE dept_id = department_id;
BEGIN
    FOR rec IN c_emps LOOP
        DBMS_OUTPUT.PUT_LINE(rec.employee_name);
    END LOOP;
END;"""
    
    result = is_plsql_semantically_equivalent(cursor_for1, cursor_for2)
    test_results.append(("Edge: Cursor FOR loop", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 19: WHILE loop
    print("\n[Test 19] WHILE loop")
    while1 = """CREATE OR REPLACE PROCEDURE countdown(p_start NUMBER)
IS
    v_counter NUMBER;
BEGIN
    v_counter := p_start;
    WHILE v_counter > 0 LOOP
        DBMS_OUTPUT.PUT_LINE(v_counter);
        v_counter := v_counter - 1;
    END LOOP;
END;"""
    
    while2 = """CREATE OR REPLACE PROCEDURE countdown(start_val NUMBER)
IS
    cnt NUMBER;
BEGIN
    cnt := start_val;
    WHILE cnt > 0 LOOP
        DBMS_OUTPUT.PUT_LINE(cnt);
        cnt := cnt - 1;
    END LOOP;
END;"""
    
    result = is_plsql_semantically_equivalent(while1, while2)
    test_results.append(("Edge: WHILE loop", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 20: EXECUTE IMMEDIATE (Dynamic SQL)
    print("\n[Test 20] EXECUTE IMMEDIATE")
    exec1 = """CREATE OR REPLACE PROCEDURE run_dynamic(p_table VARCHAR2, p_value NUMBER)
IS
    v_sql VARCHAR2(1000);
BEGIN
    v_sql := 'UPDATE ' || p_table || ' SET status = 1 WHERE id = :1';
    EXECUTE IMMEDIATE v_sql USING p_value;
END;"""
    
    exec2 = """CREATE OR REPLACE PROCEDURE run_dynamic(tbl_name VARCHAR2, val NUMBER)
IS
    sql_stmt VARCHAR2(1000);
BEGIN
    sql_stmt := 'UPDATE ' || tbl_name || ' SET status = 1 WHERE id = :1';
    EXECUTE IMMEDIATE sql_stmt USING val;
END;"""
    
    result = is_plsql_semantically_equivalent(exec1, exec2)
    test_results.append(("Edge: EXECUTE IMMEDIATE", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 21: %TYPE and %ROWTYPE
    print("\n[Test 21] %TYPE and %ROWTYPE")
    type1 = """CREATE OR REPLACE PROCEDURE get_emp_info(p_id employees.employee_id%TYPE)
IS
    v_emp employees%ROWTYPE;
BEGIN
    SELECT * INTO v_emp FROM employees WHERE employee_id = p_id;
    DBMS_OUTPUT.PUT_LINE(v_emp.employee_name);
END;"""
    
    type2 = """CREATE OR REPLACE PROCEDURE get_emp_info(emp_id employees.employee_id%TYPE)
IS
    emp_rec employees%ROWTYPE;
BEGIN
    SELECT * INTO emp_rec FROM employees WHERE employee_id = emp_id;
    DBMS_OUTPUT.PUT_LINE(emp_rec.employee_name);
END;"""
    
    result = is_plsql_semantically_equivalent(type1, type2)
    test_results.append(("Edge: %TYPE and %ROWTYPE", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 22: Nested blocks
    print("\n[Test 22] Nested blocks")
    nested1 = """CREATE OR REPLACE PROCEDURE nested_proc(p_val NUMBER)
IS
    v_outer NUMBER;
BEGIN
    v_outer := p_val;
    BEGIN
        DECLARE
            v_inner NUMBER;
        BEGIN
            v_inner := v_outer * 2;
            DBMS_OUTPUT.PUT_LINE(v_inner);
        END;
    END;
END;"""
    
    nested2 = """CREATE OR REPLACE PROCEDURE nested_proc(val NUMBER)
IS
    outer_var NUMBER;
BEGIN
    outer_var := val;
    BEGIN
        DECLARE
            inner_var NUMBER;
        BEGIN
            inner_var := outer_var * 2;
            DBMS_OUTPUT.PUT_LINE(inner_var);
        END;
    END;
END;"""
    
    result = is_plsql_semantically_equivalent(nested1, nested2)
    test_results.append(("Edge: Nested blocks", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 23: MERGE statement
    print("\n[Test 23] MERGE statement")
    merge1 = """CREATE OR REPLACE PROCEDURE sync_data(p_source_id NUMBER)
IS
BEGIN
    MERGE INTO target_table t
    USING (SELECT * FROM source_table WHERE id = p_source_id) s
    ON (t.id = s.id)
    WHEN MATCHED THEN UPDATE SET t.name = s.name
    WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name);
END;"""
    
    merge2 = """CREATE OR REPLACE PROCEDURE sync_data(src_id NUMBER)
IS
BEGIN
    MERGE INTO target_table t
    USING (SELECT * FROM source_table WHERE id = src_id) s
    ON (t.id = s.id)
    WHEN MATCHED THEN UPDATE SET t.name = s.name
    WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name);
END;"""
    
    result = is_plsql_semantically_equivalent(merge1, merge2)
    test_results.append(("Edge: MERGE statement", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 24: User-defined exception
    print("\n[Test 24] User-defined exception")
    exc1 = """CREATE OR REPLACE PROCEDURE check_balance(p_amount NUMBER)
IS
    insufficient_funds EXCEPTION;
    v_balance NUMBER;
BEGIN
    SELECT balance INTO v_balance FROM accounts WHERE id = 1;
    IF v_balance < p_amount THEN
        RAISE insufficient_funds;
    END IF;
EXCEPTION
    WHEN insufficient_funds THEN
        DBMS_OUTPUT.PUT_LINE('Not enough funds');
END;"""
    
    exc2 = """CREATE OR REPLACE PROCEDURE check_balance(amount NUMBER)
IS
    not_enough_money EXCEPTION;
    bal NUMBER;
BEGIN
    SELECT balance INTO bal FROM accounts WHERE id = 1;
    IF bal < amount THEN
        RAISE not_enough_money;
    END IF;
EXCEPTION
    WHEN not_enough_money THEN
        DBMS_OUTPUT.PUT_LINE('Not enough funds');
END;"""
    
    result = is_plsql_semantically_equivalent(exc1, exc2)
    test_results.append(("Edge: User-defined exception", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 25: Different parameter order (should be False)
    print("\n[Test 25] Different parameter order")
    order1 = """CREATE OR REPLACE PROCEDURE test_order(p_a NUMBER, p_b VARCHAR2)
IS
BEGIN
    DBMS_OUTPUT.PUT_LINE(p_a || p_b);
END;"""
    
    order2 = """CREATE OR REPLACE PROCEDURE test_order(p_b VARCHAR2, p_a NUMBER)
IS
BEGIN
    DBMS_OUTPUT.PUT_LINE(p_a || p_b);
END;"""
    
    result = is_plsql_semantically_equivalent(order1, order2)
    test_results.append(("Edge: Different parameter order", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 26: FORALL statement
    print("\n[Test 26] FORALL statement")
    forall1 = """CREATE OR REPLACE PROCEDURE bulk_insert(p_count NUMBER)
IS
    TYPE num_array IS TABLE OF NUMBER;
    v_ids num_array := num_array();
BEGIN
    FOR i IN 1..p_count LOOP
        v_ids.EXTEND;
        v_ids(i) := i;
    END LOOP;
    FORALL idx IN 1..v_ids.COUNT
        INSERT INTO test_table (id) VALUES (v_ids(idx));
END;"""
    
    forall2 = """CREATE OR REPLACE PROCEDURE bulk_insert(cnt NUMBER)
IS
    TYPE t_nums IS TABLE OF NUMBER;
    l_ids t_nums := t_nums();
BEGIN
    FOR j IN 1..cnt LOOP
        l_ids.EXTEND;
        l_ids(j) := j;
    END LOOP;
    FORALL k IN 1..l_ids.COUNT
        INSERT INTO test_table (id) VALUES (l_ids(k));
END;"""
    
    result = is_plsql_semantically_equivalent(forall1, forall2)
    test_results.append(("Edge: FORALL statement", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 27: REF CURSOR
    print("\n[Test 27] REF CURSOR")
    ref1 = """CREATE OR REPLACE PROCEDURE get_data(p_cursor OUT SYS_REFCURSOR)
IS
BEGIN
    OPEN p_cursor FOR SELECT * FROM employees;
END;"""
    
    ref2 = """CREATE OR REPLACE PROCEDURE get_data(result_cursor OUT SYS_REFCURSOR)
IS
BEGIN
    OPEN result_cursor FOR SELECT * FROM employees;
END;"""
    
    result = is_plsql_semantically_equivalent(ref1, ref2)
    test_results.append(("Edge: REF CURSOR", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 28: REVERSE FOR loop
    print("\n[Test 28] REVERSE FOR loop")
    rev1 = """CREATE OR REPLACE PROCEDURE reverse_loop(p_max NUMBER)
IS
BEGIN
    FOR i IN REVERSE 1..p_max LOOP
        DBMS_OUTPUT.PUT_LINE(i);
    END LOOP;
END;"""
    
    rev2 = """CREATE OR REPLACE PROCEDURE reverse_loop(max_val NUMBER)
IS
BEGIN
    FOR j IN REVERSE 1..max_val LOOP
        DBMS_OUTPUT.PUT_LINE(j);
    END LOOP;
END;"""
    
    result = is_plsql_semantically_equivalent(rev1, rev2)
    test_results.append(("Edge: REVERSE FOR loop", result, True))
    print(f"Expected: True, Got: {result} {'✓' if result == True else '✗'}")
    
    # Test 29: Different DBMS_OUTPUT message (should be False)
    print("\n[Test 29] Different string literal")
    str1 = """CREATE OR REPLACE PROCEDURE test_str(p_id NUMBER)
IS
BEGIN
    DBMS_OUTPUT.PUT_LINE('Hello World');
END;"""
    
    str2 = """CREATE OR REPLACE PROCEDURE test_str(p_id NUMBER)
IS
BEGIN
    DBMS_OUTPUT.PUT_LINE('Goodbye World');
END;"""
    
    result = is_plsql_semantically_equivalent(str1, str2)
    test_results.append(("Edge: Different string literal", result, False))
    print(f"Expected: False, Got: {result} {'✓' if result == False else '✗'}")
    
    # Test 30: RECORD type
    print("\n[Test 30] RECORD type declaration")
    rec1 = """CREATE OR REPLACE PROCEDURE process_emp(p_id NUMBER)
IS
    TYPE emp_rec_type IS RECORD (
        id NUMBER,
        name VARCHAR2(100)
    );
    v_emp emp_rec_type;
BEGIN
    SELECT employee_id, employee_name INTO v_emp.id, v_emp.name FROM employees WHERE employee_id = p_id;
    DBMS_OUTPUT.PUT_LINE(v_emp.name);
END;"""
    
    rec2 = """CREATE OR REPLACE PROCEDURE process_emp(emp_id NUMBER)
IS
    TYPE t_employee IS RECORD (
        id NUMBER,
        name VARCHAR2(100)
    );
    l_emp t_employee;
BEGIN
    SELECT employee_id, employee_name INTO l_emp.id, l_emp.name FROM employees WHERE employee_id = emp_id;
    DBMS_OUTPUT.PUT_LINE(l_emp.name);
END;"""
    
    result = is_plsql_semantically_equivalent(rec1, rec2)
    test_results.append(("Edge: RECORD type declaration", result, True))
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

'''
if __name__ == "__main__":

    short_name = oracle_db_long_to_short["netcdf_file_metadata_management"]
    print(short_name)
    long_name = oracle_db_short_to_long[short_name]
    print(long_name)

    print("=== Get Database Schema Graph Tests ===\n")
    print(get_database_schema_graph()["natural_lpm_management"], "\n")

    print("=== Get Table Info Tests ===\n")
    print(get_tables_info(short_name), "\n")
    
    print("=== Get Database Schema Tests ===\n")
    print(get_database_schema(short_name), "\n")

    print("=== Get All User Tables Tests ===\n")
    print(get_all_user_tables(short_name), "\n")

    print("=== Get Important System Tables Tests ===\n")
    print(get_important_system_tables(), "\n")

    print("=== Fetch System Table Data Tests ===\n")
    print(fetch_system_table_data("all_constraints"), "\n")

    print("=== Recreate Database with Context Tests ===\n")
    print(recreate_database_with_context(short_name), "\n")

    print("=== Recreate Databases with Context Tests ===\n")
    print(recreate_databases_with_context([short_name]), "\n")

    print("=== Check PL/SQL Executability Tests ===\n")
    print(check_plsql_executability("""CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = para_name WHERE \"attribute_id\" = para_attribute_id; END;""",
                              ["BEGIN sp(0, '666'); COMMIT; END;",
                               "BEGIN sp(1, '790'); COMMIT; END;",
                               "BEGIN sp(0, '785'); COMMIT; END;"],
                                short_name))

    print("=== Compare PL/SQL Tests ===\n")
    print(compare_plsql(short_name,
                        """CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = para_name WHERE \"attribute_id\" = para_attribute_id; END;""",
                        """CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = '111122' WHERE \"attribute_id\" = para_attribute_id; END;""",
                        ["BEGIN sp(0, '666'); COMMIT; END;",
                         "BEGIN sp(1, '790'); COMMIT; END;",
                         "BEGIN sp(0, '785'); COMMIT; END;"],
                         True))

    print(compare_plsql(short_name,
                        """CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = para_name WHERE \"attribute_id\" = para_attribute_id; END;""",
                        """CREATE OR REPLACE PROCEDURE sp(para_attribute_id NUMBER, para_name VARCHAR2) IS BEGIN UPDATE \"attributes\" SET \"name\" = para_name WHERE \"attribute_id\" = para_attribute_id; END;""",
                        ["BEGIN sp(0, '666'); COMMIT; END;",
                         "BEGIN sp(1, '790'); COMMIT; END;",
                         "BEGIN sp(0, '785'); COMMIT; END;"],
                         True))

    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_count NUMBER; BEGIN SELECT COUNT(*) INTO record_count FROM "coach" WHERE "Gender" = para_Gender; IF record_count = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_coun NUMBER; BEGIN SELECT COUNT(*) INTO record_coun FROM "coach" WHERE "Gender" = para_Gender; IF record_coun = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END"""))

    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_count NUMBER; BEGIN SELECT COUNT(*) INTO record_count FROM "coach" WHERE "Gender" = para_Gender; IF record_count = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_Gender IN VARCHAR2) IS record_coun NUMBER; BEGIN SELECT COUNT(*) INTO record_coun FROM "coach" WHERE "Gender" = para_Gender; IF record_coun = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END"""))
'''