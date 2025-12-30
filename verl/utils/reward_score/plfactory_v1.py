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
from psycopg_pool import ConnectionPool
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

# PostgreSQL 连接池配置
# 注意：max_connections=300 时，最大并发 worker 数 ≈ (300 - 维护池) / _PG_POOL_MAX_SIZE
# 例如：(300 - 5) / 5 = 59 个并发 worker
_PG_POOL_MIN_SIZE = 1
_PG_POOL_MAX_SIZE = 5  # 每个 worker database 最多 5 个连接，节省总连接数
_PG_POOL_TIMEOUT = 30.0
_PG_POOL_IDLE_TIMEOUT = 60.0  # 1分钟未使用则清理，更积极地回收资源

# 模板数据库缓存 (key: original_db_name, value: template_db_name)
# 使用模板数据库可以大幅加速 CREATE DATABASE（利用 PostgreSQL 的 COW 机制）
_template_db_cache: Dict[str, str] = {}
_template_db_lock = threading.Lock()

# 连接池缓存 (key: conninfo string, value: ConnectionPool)
_pg_pools: Dict[str, ConnectionPool] = {}
_pg_pool_last_used: Dict[str, float] = {}  # 记录最后使用时间
_pg_pool_lock = threading.Lock()

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

# Oracle 连接重试配置
_OC_MAX_RETRIES = 3
_OC_RETRY_DELAY = 1.0
# 可重试的 Oracle 错误关键字（小写匹配）
_OC_RETRYABLE_ERRORS = [
    'closed the connection',  # DPY-4011
    'not connected',
    'connection lost',
    'dpy-4011',
    'dpy-1001',
    'timeout',
    'ora-00600',  # Oracle 内部错误
    'ora-27090',  # I/O 资源问题
    'ora-03113',  # end-of-file on communication channel
    'ora-03114',  # not connected to ORACLE
    'ora-12170',  # TNS:Connect timeout
    'ora-12541',  # TNS:no listener
    'ora-12543',  # TNS:destination host unreachable
]

pg_input_path = "/workspace/opt/projects/verlpl/examples/datasets/train/database/postgresql"
oc_input_path = "/workspace/opt/projects/verlpl/examples/datasets/train/database/oracle"


# ============================================================
# PostgreSQL 连接池管理
# ============================================================

def _make_worker_db_conninfo(dbname: str) -> str:
    """统一构建 worker database 的 conninfo 字符串，确保格式一致"""
    return psycopg.conninfo.make_conninfo(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        dbname=dbname
    )


def _check_connection(conn) -> None:
    """连接健康检查函数"""
    try:
        conn.execute("SELECT 1")
    except Exception:
        raise


def _get_pg_pool(conninfo: str, open_now: bool = True) -> ConnectionPool:
    """
    获取或创建 PostgreSQL 连接池
    
    参数:
        conninfo: 连接信息字符串
        open_now: 是否立即打开连接池（默认True）
                  对于 worker 数据库，应该在数据库创建后再打开
    """
    with _pg_pool_lock:
        # 清理空闲过久的连接池
        _cleanup_idle_pools_locked()
        
        if conninfo not in _pg_pools:
            _pg_pools[conninfo] = ConnectionPool(
                conninfo=conninfo,
                min_size=_PG_POOL_MIN_SIZE,
                max_size=_PG_POOL_MAX_SIZE,
                timeout=_PG_POOL_TIMEOUT,
                open=False,  # 延迟打开，避免连接到不存在的数据库
                check=_check_connection,  # 添加连接健康检查
            )
        
        pool = _pg_pools[conninfo]
        
        # 如果需要立即打开且连接池未打开
        if open_now and pool.closed:
            pool.open()
        
        # 更新最后使用时间
        _pg_pool_last_used[conninfo] = time.time()
        
        return pool


def _cleanup_idle_pools_locked():
    """
    清理长时间未使用的连接池（必须在持有锁的情况下调用）
    
    注意：不清理维护池（postgres数据库的连接池）
    """
    now = time.time()
    maintenance_conninfo = psycopg.conninfo.make_conninfo(pg_conn_info, dbname="postgres")
    
    to_remove = []
    for conninfo, last_used in list(_pg_pool_last_used.items()):
        # 不清理维护池
        if conninfo == maintenance_conninfo:
            continue
        # 检查是否超时
        if now - last_used > _PG_POOL_IDLE_TIMEOUT:
            to_remove.append(conninfo)
    
    for conninfo in to_remove:
        try:
            if conninfo in _pg_pools:
                pool = _pg_pools[conninfo]
                if not pool.closed:
                    pool.close()
                del _pg_pools[conninfo]
            if conninfo in _pg_pool_last_used:
                del _pg_pool_last_used[conninfo]
        except Exception as e:
            print(f"Warning: Failed to cleanup idle pool: {e}")


def _get_pg_maintenance_pool() -> ConnectionPool:
    """获取 postgres 维护数据库的连接池"""
    maintenance_conninfo = psycopg.conninfo.make_conninfo(pg_conn_info, dbname="postgres")
    return _get_pg_pool(maintenance_conninfo, open_now=True)


def _close_pg_pool(conninfo: str):
    """关闭并移除指定的连接池"""
    with _pg_pool_lock:
        if conninfo in _pg_pools:
            try:
                pool = _pg_pools[conninfo]
                if not pool.closed:
                    pool.close()
            except Exception as e:
                print(f"Warning: Error closing pool: {e}")
            finally:
                del _pg_pools[conninfo]
        if conninfo in _pg_pool_last_used:
            del _pg_pool_last_used[conninfo]


def _ensure_pg_pool_open(conninfo: str) -> ConnectionPool:
    """确保连接池已打开并返回"""
    with _pg_pool_lock:
        if conninfo in _pg_pools:
            pool = _pg_pools[conninfo]
            if pool.closed:
                pool.open()
            _pg_pool_last_used[conninfo] = time.time()
            return pool
    # 如果池不存在，创建并打开
    return _get_pg_pool(conninfo, open_now=True)


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


def pg_execute_sql(database_conn_info: str, sql: str, conn=None):
    """执行 SQL，可复用已有连接（使用连接池）"""
    normalized = _normalize_plsql_block(sql)
    if not normalized:
        return
    
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {2 * 1000};")
            cur.execute(normalized)
        conn.commit()
    else:
        pool = _ensure_pg_pool_open(database_conn_info)
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {2 * 1000};")
                cur.execute(normalized)


def pg_fetch_query_results(database_conn_info: str, query: str, conn=None):
    """获取查询结果，可复用已有连接（使用连接池）"""
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    else:
        pool = _ensure_pg_pool_open(database_conn_info)
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()


def pg_get_all_user_tables(database_name: str, conn=None) -> List[str]:
    """获取用户表列表，可复用已有连接（使用连接池）"""
    def _fetch_tables(cur):
        cur.execute(
            """
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY tablename;
            """
        )
        return [row[0] for row in cur.fetchall()]
    
    if conn is not None:
        with conn.cursor() as cur:
            return _fetch_tables(cur)
    else:
        conn_db_info = _make_worker_db_conninfo(database_name)
        pool = _ensure_pg_pool_open(conn_db_info)
        with pool.connection() as conn:
            with conn.cursor() as cur:
                return _fetch_tables(cur)


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
        pool = _ensure_pg_pool_open(database_conn_info)
        with pool.connection() as conn:
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


def _ensure_template_database(original_db_name: str) -> str:
    """
    确保模板数据库存在。如果不存在则创建。
    
    模板数据库命名: tpl_{original_db_name}
    使用模板数据库的好处:
    1. CREATE DATABASE ... TEMPLATE xxx 比 psql -f 快 5-10 倍
    2. PostgreSQL 使用 COW (Copy-On-Write) 机制，几乎瞬间完成
    
    返回: 模板数据库名称
    """
    template_db_name = f"tpl_{original_db_name}"
    
    with _template_db_lock:
        # 检查缓存
        if original_db_name in _template_db_cache:
            return _template_db_cache[original_db_name]
        
        # 检查模板数据库是否已存在
        pool = _get_pg_maintenance_pool()
        with pool.connection() as conn:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (template_db_name,)
                )
                exists = cur.fetchone() is not None
        
        if not exists:
            # 创建模板数据库
            print(f"Creating template database: {template_db_name} from {original_db_name}.sql ...")
            input_file = os.path.join(pg_input_path, f"{original_db_name}.sql")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"SQL dump not found: {input_file}")
            
            # 创建空数据库
            pg_recreate_database(pg_conn_info, template_db_name)
            # 导入数据
            pg_import_database(pg_host, pg_port, pg_user, pg_password, template_db_name, input_file)
            print(f"Template database {template_db_name} created successfully.")
        
        _template_db_cache[original_db_name] = template_db_name
        return template_db_name


def pg_recreate_database_from_template(db_name: str, template_db_name: str):
    """
    使用模板数据库快速创建新数据库（比 psql -f 快 5-10 倍）
    
    PostgreSQL 的 CREATE DATABASE ... TEMPLATE 使用文件系统级别的复制，
    对于小型数据库几乎是瞬间完成。
    """
    pool = _get_pg_maintenance_pool()
    with pool.connection() as conn:
        conn.rollback()
        old_autocommit = conn.autocommit
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                # 终止目标数据库的所有连接
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    (db_name,),
                )
                # 终止模板数据库的所有连接（创建时模板不能有活动连接）
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    (template_db_name,),
                )
                
                # DROP 目标数据库
                ident = psql.Identifier(db_name)
                cur.execute(psql.SQL("DROP DATABASE IF EXISTS {}").format(ident))
                
                # 从模板创建新数据库
                template_ident = psql.Identifier(template_db_name)
                try:
                    cur.execute(
                        psql.SQL("CREATE DATABASE {} TEMPLATE {}").format(ident, template_ident)
                    )
                except psycopg.errors.DuplicateDatabase:
                    pass
        finally:
            # 安全恢复 autocommit，避免恢复失败时覆盖原始异常
            try:
                conn.autocommit = old_autocommit
            except Exception:
                pass  # 连接可能已断开，忽略恢复失败


def pg_recreate_database(conn_info: str, db_name: str, maintenance_db: str = "postgres"):
    """重新创建单个数据库（使用连接池）- 空数据库版本"""
    pool = _get_pg_maintenance_pool()
    with pool.connection() as conn:
        # 关键修复：从连接池获取的连接可能处于事务状态，
        # 必须先回滚才能设置 autocommit
        conn.rollback()
        old_autocommit = conn.autocommit
        conn.autocommit = True
        try:
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
        finally:
            # 安全恢复 autocommit，避免恢复失败时覆盖原始异常
            try:
                conn.autocommit = old_autocommit
            except Exception:
                pass  # 连接可能已断开，忽略恢复失败


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


def pg_restore_database(worker_db_name: str, original_db_name: str, use_template: bool = True):
    """
    恢复数据库（高性能版本）。
    
    参数:
        worker_db_name: worker 专用的数据库名
        original_db_name: 原始数据库名（用于查找 SQL 文件或模板数据库）
        use_template: 是否使用模板数据库加速（默认 True，快 5-10 倍）
    """
    input_file = os.path.join(pg_input_path, f"{original_db_name}.sql")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"SQL dump not found: {input_file} (original db: {original_db_name})"
        )
    
    # 关键修复：在 DROP 数据库之前先关闭对应的连接池
    worker_conninfo = _make_worker_db_conninfo(worker_db_name)
    _close_pg_pool(worker_conninfo)
    
    if use_template:
        # 高性能路径：使用模板数据库
        template_db_name = _ensure_template_database(original_db_name)
        pg_recreate_database_from_template(worker_db_name, template_db_name)
    else:
        # 传统路径：DROP + CREATE + psql import
        pg_recreate_database(pg_conn_info, worker_db_name)
        pg_import_database(
            pg_host, pg_port, pg_user, pg_password, 
            worker_db_name, input_file
        )


def pg_cleanup_worker_database(worker_db_name: str):
    """清理 worker 数据库（使用连接池）"""
    # 先关闭该 worker database 对应的连接池
    worker_conninfo = _make_worker_db_conninfo(worker_db_name)
    _close_pg_pool(worker_conninfo)
    
    try:
        pool = _get_pg_maintenance_pool()
        with pool.connection() as conn:
            # 关键修复：从连接池获取的连接可能处于事务状态，
            # 必须先回滚才能设置 autocommit
            conn.rollback()
            old_autocommit = conn.autocommit
            conn.autocommit = True
            try:
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
            finally:
                # 安全恢复 autocommit
                try:
                    conn.autocommit = old_autocommit
                except Exception:
                    pass
        print(f"Cleaned up worker database: {worker_db_name}")
    except Exception as exc:
        print(f"Warning: Failed to cleanup worker database {worker_db_name}: {exc}")


def pg_cleanup_template_databases():
    """
    清理所有模板数据库（可选，通常不需要调用）。
    模板数据库是持久化的，可以跨多次运行复用。
    """
    with _template_db_lock:
        pool = _get_pg_maintenance_pool()
        with pool.connection() as conn:
            conn.rollback()
            old_autocommit = conn.autocommit
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    for original_db, template_db in list(_template_db_cache.items()):
                        try:
                            # 终止模板数据库的所有连接
                            cur.execute(
                                """
                                SELECT pg_terminate_backend(pid)
                                FROM pg_stat_activity
                                WHERE datname = %s AND pid <> pg_backend_pid()
                                """,
                                (template_db,),
                            )
                            ident = psql.Identifier(template_db)
                            cur.execute(psql.SQL("DROP DATABASE IF EXISTS {}").format(ident))
                            print(f"Cleaned up template database: {template_db}")
                        except Exception as exc:
                            print(f"Warning: Failed to cleanup template database {template_db}: {exc}")
            finally:
                # 安全恢复 autocommit
                try:
                    conn.autocommit = old_autocommit
                except Exception:
                    pass
        _template_db_cache.clear()


def pg_capture_table_state(database_conn_info: str, table_names: List[str], conn=None) -> Dict[str, Optional[pd.DataFrame]]:
    """批量抓取多个表的数据状态，可复用已有连接（使用连接池）。"""
    snapshots = {}
    
    def _capture(cur, connection):
        for table in table_names:
            try:
                cur.execute(f'SELECT * FROM "{table}" ORDER BY 1;')
                result = cur.fetchall()
                columns = [desc[0] for desc in cur.description] if cur.description else []
                snapshots[table] = pd.DataFrame(result, columns=columns)
            except Exception as e:
                print(f"Warning: Failed to capture table {table}: {e}")
                snapshots[table] = None
                # 关键修复：回滚事务以清除 aborted 状态，否则后续查询都会失败
                try:
                    connection.rollback()
                except Exception:
                    pass
    
    try:
        if conn is not None:
            with conn.cursor() as cur:
                _capture(cur, conn)
        else:
            pool = _ensure_pg_pool_open(database_conn_info)
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    _capture(cur, conn)
    except Exception as e:
        print(f"Warning: Database connection failed: {e}")
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
    """比较两个 PL/SQL 函数（使用连接池）"""
    try:
        database_conn_info = _make_worker_db_conninfo(worker_db_name)
        
        # ========== 第一轮 ==========
        _log(worker_db_name, f"[Round 1] Restoring database from {original_db_name}.sql...")
        pg_restore_database(worker_db_name, original_db_name)
        
        # 数据库创建后再获取/打开连接池
        pool = _ensure_pg_pool_open(database_conn_info)
        
        # 在每轮内复用同一个连接
        with pool.connection() as conn:
            _log(worker_db_name, f"[Round 1] Executing solution PLSQL (function)...")
            pg_execute_sql(database_conn_info, plsql1, conn=conn)
            
            function_results1 = {}
            _log(worker_db_name, f"[Round 1] Executing {len(call_plsqls)} call statement(s)...")
            for i, call_plsql in enumerate(call_plsqls):
                try:
                    _log(worker_db_name, f"[Round 1] Executing call [{i+1}]: {call_plsql[:100]}...")
                    result = pg_fetch_query_results(database_conn_info, call_plsql, conn=conn)
                    function_results1[i] = {"sql": call_plsql, "result": pd.DataFrame(result)}
                    _log(worker_db_name, f"[Round 1] Call [{i+1}] returned {len(result)} rows")
                except Exception as exc:
                    _log(worker_db_name, f"[Round 1] Call [{i+1}] failed: {exc}", level="WARN")
                    function_results1[i] = {"sql": call_plsql, "result": None, "error": str(exc)}
                    # 关键修复：回滚事务以清除 aborted 状态，否则后续查询都会失败
                    try:
                        conn.rollback()
                    except Exception:
                        pass

        # ========== 第二轮 ==========
        _log(worker_db_name, f"[Round 2] Restoring database from {original_db_name}.sql...")
        pg_restore_database(worker_db_name, original_db_name)
        
        # 数据库重建后需要重新打开连接池
        pool = _ensure_pg_pool_open(database_conn_info)
        
        # 在每轮内复用同一个连接
        with pool.connection() as conn:
            _log(worker_db_name, f"[Round 2] Executing ground truth PLSQL (function)...")
            pg_execute_sql(database_conn_info, plsql2, conn=conn)
            
            function_results2 = {}
            _log(worker_db_name, f"[Round 2] Executing {len(call_plsqls)} call statement(s)...")
            for i, call_plsql in enumerate(call_plsqls):
                try:
                    _log(worker_db_name, f"[Round 2] Executing call [{i+1}]: {call_plsql[:100]}...")
                    result = pg_fetch_query_results(database_conn_info, call_plsql, conn=conn)
                    function_results2[i] = {"sql": call_plsql, "result": pd.DataFrame(result)}
                    _log(worker_db_name, f"[Round 2] Call [{i+1}] returned {len(result)} rows")
                except Exception as exc:
                    _log(worker_db_name, f"[Round 2] Call [{i+1}] failed: {exc}", level="WARN")
                    function_results2[i] = {"sql": call_plsql, "result": None, "error": str(exc)}
                    # 关键修复：回滚事务以清除 aborted 状态，否则后续查询都会失败
                    try:
                        conn.rollback()
                    except Exception:
                        pass

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
    """比较两个 PL/SQL 对表的影响（使用连接池）"""
    try:
        database_conn_info = _make_worker_db_conninfo(worker_db_name)

        # ========== 第一轮 ==========
        _log(worker_db_name, f"[Round 1] Restoring database from {original_db_name}.sql...")
        pg_restore_database(worker_db_name, original_db_name)

        # 数据库创建后再获取/打开连接池
        pool = _ensure_pg_pool_open(database_conn_info)

        # 在每轮内复用同一个连接
        round1_call_failed = False
        with pool.connection() as conn:
            all_user_tables = pg_get_all_user_tables(worker_db_name, conn=conn)
            _log(worker_db_name, f"[Round 1] Found {len(all_user_tables)} user tables: {all_user_tables}")

            _log(worker_db_name, f"[Round 1] Executing solution PLSQL...")
            try:
                pg_execute_sql(database_conn_info, plsql1, conn=conn)
            except Exception as e:
                _log(worker_db_name, f"[Round 1] Failed to execute solution PLSQL: {e}", level="ERROR")
                return 0.0
            
            _log(worker_db_name, f"[Round 1] Executing {len(call_plsqls)} call statement(s)...")
            for i, call in enumerate(call_plsqls):
                _log(worker_db_name, f"[Round 1] Executing call [{i+1}]: {call[:100]}...")
                try:
                    pg_execute_sql(database_conn_info, call, conn=conn)
                except Exception as e:
                    _log(worker_db_name, f"[Round 1] Call [{i+1}] failed: {e}", level="WARN")
                    round1_call_failed = True
                    # 关键修复：回滚事务以清除 aborted 状态，否则后续操作都会失败
                    try:
                        conn.rollback()
                    except Exception:
                        pass

            user_tables_results1 = pg_capture_table_state(database_conn_info, all_user_tables, conn=conn)

        # ========== 第二轮 ==========
        _log(worker_db_name, f"[Round 2] Restoring database from {original_db_name}.sql...")
        pg_restore_database(worker_db_name, original_db_name)
        
        # 数据库重建后需要重新打开连接池
        pool = _ensure_pg_pool_open(database_conn_info)
        
        # 在每轮内复用同一个连接
        round2_call_failed = False
        with pool.connection() as conn:
            _log(worker_db_name, f"[Round 2] Executing ground truth PLSQL...")
            pg_execute_sql(database_conn_info, plsql2, conn=conn)
            _log(worker_db_name, f"[Round 2] Executing {len(call_plsqls)} call statement(s)...")
            for i, call in enumerate(call_plsqls):
                _log(worker_db_name, f"[Round 2] Executing call [{i+1}]: {call[:100]}...")
                try:
                    pg_execute_sql(database_conn_info, call, conn=conn)
                except Exception as e:
                    _log(worker_db_name, f"[Round 2] Call [{i+1}] failed: {e}", level="WARN")
                    round2_call_failed = True
                    # 关键修复：回滚事务以清除 aborted 状态，否则后续操作都会失败
                    try:
                        conn.rollback()
                    except Exception:
                        pass

            user_tables_results2 = pg_capture_table_state(database_conn_info, all_user_tables, conn=conn)

        # 如果 solution 执行失败但 ground_truth 成功，返回 0
        if round1_call_failed and not round2_call_failed:
            _log(worker_db_name, f"[Compare] Solution calls failed but ground truth succeeded, returning 0.0")
            return 0.0

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


def oc_get_connection(user: str = oc_user, password: str = oc_password, max_retries: int = _OC_MAX_RETRIES):
    """
    获取 Oracle 连接（带重试和健康检查）
    
    参数:
        user: 用户名，默认使用配置中的 oc_user
        password: 密码，默认使用配置中的 oc_password
        max_retries: 最大重试次数，默认使用 _OC_MAX_RETRIES
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            conn = oracledb.connect(
                user=user,
                password=password,
                host=oc_host,
                port=oc_port,
                service_name=oc_service_name,
            )
            # 验证连接有效性
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM DUAL")
            return conn
        except oracledb.Error as e:
            last_error = e
            error_msg = str(e).lower()
            # 使用统一的可重试错误列表
            is_retryable = any(x in error_msg for x in _OC_RETRYABLE_ERRORS)
            if is_retryable:
                print(f"Oracle connection failed (attempt {attempt + 1}/{max_retries}): {e}, retryable=True")
                if attempt < max_retries - 1:
                    time.sleep(_OC_RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    print(f"Oracle connection: all {max_retries} retries exhausted")
            raise
    raise last_error


def oc_split_sql_statements(sql_text: str) -> List[str]:
    """
    分割 Oracle SQL 语句，正确处理 PL/SQL 块。
    
    处理的情况：
    1. 单引号和双引号内的分号
    2. PL/SQL 块（BEGIN...END; 或 CREATE PROCEDURE/FUNCTION/TRIGGER...END;）
    3. 使用 / 作为 PL/SQL 块结束符
    4. 多行注释 /* ... */
    5. 单行注释 -- ...
    """
    statements = []
    buffer = []
    in_single_quote = False
    in_double_quote = False
    in_multiline_comment = False
    in_plsql_block = False
    plsql_depth = 0  # 跟踪嵌套的 BEGIN/END
    
    i = 0
    text_upper = sql_text.upper()
    n = len(sql_text)
    
    while i < n:
        char = sql_text[i]
        
        # 处理多行注释 /* ... */
        if not in_single_quote and not in_double_quote:
            if i + 1 < n and sql_text[i:i+2] == '/*':
                in_multiline_comment = True
                buffer.append(sql_text[i:i+2])
                i += 2
                continue
            if in_multiline_comment and i + 1 < n and sql_text[i:i+2] == '*/':
                in_multiline_comment = False
                buffer.append(sql_text[i:i+2])
                i += 2
                continue
        
        if in_multiline_comment:
            buffer.append(char)
            i += 1
            continue
        
        # 处理单行注释 -- ...
        if not in_single_quote and not in_double_quote:
            if i + 1 < n and sql_text[i:i+2] == '--':
                # 跳到行尾
                while i < n and sql_text[i] != '\n':
                    buffer.append(sql_text[i])
                    i += 1
                if i < n:
                    buffer.append(sql_text[i])  # 添加换行符
                    i += 1
                continue
        
        # 处理引号
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        
        in_quotes = in_single_quote or in_double_quote
        
        # 检测 PL/SQL 块的开始（CREATE PROCEDURE/FUNCTION/TRIGGER 或 DECLARE/BEGIN）
        if not in_quotes and not in_plsql_block:
            # 检查是否是 PL/SQL 块开始关键字
            remaining = text_upper[i:]
            # CREATE PROCEDURE/FUNCTION/TRIGGER/PACKAGE
            if remaining.startswith('CREATE'):
                # 查找是否包含 PROCEDURE/FUNCTION/TRIGGER/PACKAGE
                next_100 = text_upper[i:i+100]
                if any(kw in next_100 for kw in ['PROCEDURE', 'FUNCTION', 'TRIGGER', 'PACKAGE']):
                    in_plsql_block = True
                    plsql_depth = 0
            # DECLARE 或独立的 BEGIN
            elif remaining.startswith('DECLARE') or remaining.startswith('BEGIN'):
                # 确保是单词边界
                if i == 0 or not text_upper[i-1].isalnum():
                    word_end = i + 7 if remaining.startswith('DECLARE') else i + 5
                    if word_end >= n or not text_upper[word_end].isalnum():
                        in_plsql_block = True
                        plsql_depth = 1 if remaining.startswith('BEGIN') else 0
        
        # 在 PL/SQL 块中跟踪 BEGIN/END 嵌套
        if in_plsql_block and not in_quotes:
            remaining = text_upper[i:]
            # 检测 BEGIN（确保是单词边界）
            if remaining.startswith('BEGIN'):
                if (i == 0 or not text_upper[i-1].isalnum()) and (i + 5 >= n or not text_upper[i+5].isalnum()):
                    plsql_depth += 1
            # 检测 END（确保是单词边界）
            elif remaining.startswith('END'):
                if (i == 0 or not text_upper[i-1].isalnum()) and (i + 3 >= n or not text_upper[i+3].isalnum()):
                    plsql_depth -= 1
                    if plsql_depth <= 0:
                        # PL/SQL 块即将结束，找到下一个分号或 /
                        pass  # 继续处理，等待分号
        
        # 处理分号
        if char == ';' and not in_quotes:
            buffer.append(char)
            if in_plsql_block and plsql_depth <= 0:
                # PL/SQL 块结束
                statement = "".join(buffer).strip()
                if statement and not statement.startswith("--"):
                    statements.append(statement)
                buffer = []
                in_plsql_block = False
                plsql_depth = 0
            elif not in_plsql_block:
                # 普通 SQL 语句结束
                statement = "".join(buffer).strip()
                # 移除结尾的分号（因为已经在 buffer 中了）
                statement = statement.rstrip(';').strip()
                if statement and not statement.startswith("--"):
                    statements.append(statement)
                buffer = []
            i += 1
            continue
        
        # 处理 / 作为 PL/SQL 块结束符（通常在行首）
        if char == '/' and not in_quotes:
            # 检查是否是行首的 /（作为 PL/SQL 结束符）
            is_line_start = (i == 0 or sql_text[i-1] == '\n')
            is_standalone = (i + 1 >= n or sql_text[i+1] in '\n\r \t')
            if is_line_start and is_standalone and in_plsql_block:
                # / 作为 PL/SQL 块结束符
                statement = "".join(buffer).strip()
                if statement and not statement.startswith("--"):
                    statements.append(statement)
                buffer = []
                in_plsql_block = False
                plsql_depth = 0
                i += 1
                continue
        
        buffer.append(char)
        i += 1
    
    # 处理剩余内容
    tail = "".join(buffer).strip()
    # 移除结尾可能的 /
    if tail.endswith('/'):
        tail = tail[:-1].strip()
    if tail and not tail.startswith("--"):
        statements.append(tail)
    
    return statements


def oc_execute_sql_statements_in_schema(sql_file: str, schema_name: str, password: str):
    """从 SQL 文件执行语句到指定 schema"""
    with open(sql_file, "r", encoding="utf-8") as sql_handle:
        sql_content = sql_handle.read()

    statements = oc_split_sql_statements(sql_content)
    if not statements:
        raise ValueError(f"No valid SQL statements found in {sql_file}")
    
    print(f"Importing {len(statements)} statements from {os.path.basename(sql_file)} to {schema_name}...")

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
                    # 打印失败的 SQL 语句（截断显示）
                    stmt_preview = stmt[:200] + "..." if len(stmt) > 200 else stmt
                    print(f"Warning: Failed to execute SQL [{idx}] in {schema_name}: {exc}")
                    print(f"  SQL: {stmt_preview}")
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


def oc_execute_plsql(schema_name: str, plsql_code: str, conn=None):
    """执行 PL/SQL 代码，可复用已有连接"""
    normalized = _normalize_plsql_block(plsql_code)
    if not normalized:
        return
    
    if conn is not None:
        # 复用已有连接
        with conn.cursor() as cur:
            cur.call_timeout = 2 * 1000
            cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name.upper()}")
            cur.execute(normalized)
        conn.commit()
    else:
        # 创建新连接
        with oc_get_connection() as conn:
            with conn.cursor() as cur:
                cur.call_timeout = 2 * 1000
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name.upper()}")
                cur.execute(normalized)
            conn.commit()


def oc_run_call_statements(schema_name: str, call_plsqls: List[str], conn=None) -> Dict[int, dict]:
    """执行调用语句并返回结果，可复用已有连接"""
    results: Dict[int, dict] = {}
    
    def _execute_calls(cur):
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
    
    if conn is not None:
        with conn.cursor() as cur:
            _execute_calls(cur)
    else:
        with oc_get_connection() as conn:
            with conn.cursor() as cur:
                _execute_calls(cur)
    return results


def oc_get_all_user_tables(schema_name: str, conn=None) -> List[str]:
    """获取 Oracle schema 的用户表列表"""
    def _fetch_tables(cur):
        cur.execute(
            """
            SELECT table_name FROM all_tables 
            WHERE owner = :owner 
            ORDER BY table_name
            """,
            owner=schema_name.upper()
        )
        return [row[0] for row in cur.fetchall()]
    
    if conn is not None:
        with conn.cursor() as cur:
            return _fetch_tables(cur)
    else:
        with oc_get_connection() as conn:
            with conn.cursor() as cur:
                return _fetch_tables(cur)


def oc_capture_table_state(schema_name: str, table_names: List[str], conn=None) -> Dict[str, Optional[pd.DataFrame]]:
    """批量抓取 Oracle schema 中多个表的数据状态"""
    snapshots = {}
    
    def _capture(cur):
        cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name.upper()}")
        for table in table_names:
            try:
                cur.execute(f'SELECT * FROM "{table}" ORDER BY 1')
                result = cur.fetchall()
                columns = [desc[0] for desc in cur.description] if cur.description else []
                snapshots[table] = pd.DataFrame(result, columns=columns)
            except Exception as e:
                print(f"Warning: Failed to capture table {table}: {e}")
                snapshots[table] = None
    
    try:
        if conn is not None:
            with conn.cursor() as cur:
                _capture(cur)
        else:
            with oc_get_connection() as conn:
                with conn.cursor() as cur:
                    _capture(cur)
    except Exception as e:
        print(f"Warning: Oracle connection failed: {e}")
        for table in table_names:
            if table not in snapshots:
                snapshots[table] = None
    return snapshots


def oc_compare_plsql_function(
    worker_schema_name: str,
    original_schema_name: str,
    plsql1: str,
    plsql2: str,
    call_plsqls: List[str]
):
    """比较两个 Oracle PL/SQL 函数（复用连接）"""
    try:
        worker_upper = worker_schema_name.upper()
        call_plsqls = call_plsqls or []

        # ========== 第一轮 ==========
        _log(worker_schema_name, f"[Round 1] Restoring schema from {original_schema_name}.sql...")
        oc_restore_schema(worker_schema_name, original_schema_name)

        # 复用单个连接执行所有操作
        with oc_get_connection() as conn:
            _log(worker_schema_name, f"[Round 1] Executing solution PLSQL (function)...")
            try:
                oc_execute_plsql(worker_upper, plsql1, conn=conn)
            except Exception as e:
                _log(worker_schema_name, f"[Round 1] Failed to execute solution PLSQL: {e}", level="ERROR")
                return 0.0

            _log(worker_schema_name, f"[Round 1] Executing {len(call_plsqls)} call statement(s)...")
            results1 = oc_run_call_statements(worker_upper, call_plsqls, conn=conn)

        # ========== 第二轮 ==========
        _log(worker_schema_name, f"[Round 2] Restoring schema from {original_schema_name}.sql...")
        oc_restore_schema(worker_schema_name, original_schema_name)

        # 复用单个连接执行所有操作（带重试机制）
        last_error = None
        for attempt in range(_OC_MAX_RETRIES):
            try:
                with oc_get_connection() as conn:
                    _log(worker_schema_name, f"[Round 2] Executing ground truth PLSQL (function)...")
                    oc_execute_plsql(worker_upper, plsql2, conn=conn)

                    _log(worker_schema_name, f"[Round 2] Executing {len(call_plsqls)} call statement(s)...")
                    results2 = oc_run_call_statements(worker_upper, call_plsqls, conn=conn)
                break  # 成功则退出重试循环
            except oracledb.Error as e:
                last_error = e
                error_msg = str(e).lower()
                # 可重试的连接错误
                is_retryable = any(x in error_msg for x in _OC_RETRYABLE_ERRORS)
                _log(worker_schema_name, f"[Round 2] Oracle error (attempt {attempt + 1}/{_OC_MAX_RETRIES}): {e}, retryable={is_retryable}", level="WARN")
                
                if is_retryable and attempt < _OC_MAX_RETRIES - 1:
                    time.sleep(_OC_RETRY_DELAY * (attempt + 1))
                    # 重试前重新恢复 schema
                    try:
                        oc_restore_schema(worker_schema_name, original_schema_name)
                    except Exception as restore_e:
                        _log(worker_schema_name, f"[Round 2] Failed to restore schema for retry: {restore_e}", level="ERROR")
                    continue
                elif is_retryable:
                    # 最后一次重试也失败了
                    _log(worker_schema_name, f"[Round 2] All {_OC_MAX_RETRIES} retries exhausted", level="ERROR")
                raise
        else:
            # 所有重试都失败（for-else 只有在没有 break 时才执行）
            if last_error:
                raise last_error

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
    """比较两个 Oracle PL/SQL 对表的影响（优化：复用连接）"""
    try:
        worker_upper = worker_schema_name.upper()
        call_plsqls = call_plsqls or []

        # ========== 第一轮 ==========
        _log(worker_schema_name, f"[Round 1] Restoring schema from {original_schema_name}.sql...")
        oc_restore_schema(worker_schema_name, original_schema_name)

        # 复用单个连接执行所有操作
        round1_call_failed = False
        with oc_get_connection() as conn:
            all_user_tables = oc_get_all_user_tables(worker_upper, conn=conn)
            _log(worker_schema_name, f"[Round 1] Found {len(all_user_tables)} user tables: {all_user_tables}")
            
            _log(worker_schema_name, f"[Round 1] Executing solution PLSQL...")
            try:
                oc_execute_plsql(worker_upper, plsql1, conn=conn)
            except Exception as e:
                _log(worker_schema_name, f"[Round 1] Failed to execute solution PLSQL: {e}", level="ERROR")
                return 0.0
            
            _log(worker_schema_name, f"[Round 1] Executing {len(call_plsqls)} call statement(s)...")
            for i, call in enumerate(call_plsqls):
                _log(worker_schema_name, f"[Round 1] Executing call [{i+1}]: {call[:100]}...")
                try:
                    oc_execute_plsql(worker_upper, call, conn=conn)
                except Exception as e:
                    _log(worker_schema_name, f"[Round 1] Call [{i+1}] failed: {e}", level="WARN")
                    round1_call_failed = True

            user_tables_results1 = oc_capture_table_state(worker_upper, all_user_tables, conn=conn)

        # ========== 第二轮 ==========
        _log(worker_schema_name, f"[Round 2] Restoring schema from {original_schema_name}.sql...")
        oc_restore_schema(worker_schema_name, original_schema_name)
        
        # 复用单个连接执行所有操作（带重试机制）
        round2_call_failed = False
        last_error = None
        for attempt in range(_OC_MAX_RETRIES):
            try:
                with oc_get_connection() as conn:
                    _log(worker_schema_name, f"[Round 2] Executing ground truth PLSQL...")
                    oc_execute_plsql(worker_upper, plsql2, conn=conn)
                    
                    _log(worker_schema_name, f"[Round 2] Executing {len(call_plsqls)} call statement(s)...")
                    for i, call in enumerate(call_plsqls):
                        _log(worker_schema_name, f"[Round 2] Executing call [{i+1}]: {call[:100]}...")
                        try:
                            oc_execute_plsql(worker_upper, call, conn=conn)
                        except Exception as e:
                            _log(worker_schema_name, f"[Round 2] Call [{i+1}] failed: {e}", level="WARN")
                            round2_call_failed = True

                    user_tables_results2 = oc_capture_table_state(worker_upper, all_user_tables, conn=conn)
                break  # 成功则退出重试循环
            except oracledb.Error as e:
                last_error = e
                error_msg = str(e).lower()
                # 可重试的连接错误
                is_retryable = any(x in error_msg for x in _OC_RETRYABLE_ERRORS)
                _log(worker_schema_name, f"[Round 2] Oracle error (attempt {attempt + 1}/{_OC_MAX_RETRIES}): {e}, retryable={is_retryable}", level="WARN")
                
                if is_retryable and attempt < _OC_MAX_RETRIES - 1:
                    time.sleep(_OC_RETRY_DELAY * (attempt + 1))
                    # 重试前重新恢复 schema
                    try:
                        oc_restore_schema(worker_schema_name, original_schema_name)
                    except Exception as restore_e:
                        _log(worker_schema_name, f"[Round 2] Failed to restore schema for retry: {restore_e}", level="ERROR")
                    continue
                elif is_retryable:
                    # 最后一次重试也失败了
                    _log(worker_schema_name, f"[Round 2] All {_OC_MAX_RETRIES} retries exhausted", level="ERROR")
                raise
        else:
            # 所有重试都失败（for-else 只有在没有 break 时才执行）
            if last_error:
                raise last_error

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