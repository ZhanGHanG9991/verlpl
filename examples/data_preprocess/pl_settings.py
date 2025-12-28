import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk

# 获取项目根目录和实验目录
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)

pg_config = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "synsql",
    "password": "123456",
    "timeout": 5000,
    "port": 5432
}

oc_config = {
    "host": "localhost",
    "service_name": "ORCLPDB1",
    "user": "system",
    "password": "MyPassword123",
    "timeout": 5000,
    "port": 1521
}

# ==========================================
# 数据集配置 (自动生成路径)
# ==========================================
# 数据集名称格式: {database_type}_{dataset_name}_{object_type}_{split}
# 例如: postgres_spider_function_test, oracle_generation_procedure_train
# 
# 数据集文件自动从 experiments/datasets/ 目录下查找
# 文件名格式: {dataset_name}.json
# 
# 无需在此手动配置，只需将数据集文件放到 experiments/datasets/ 目录即可

# 数据集文件目录
DATASETS_DIR = os.path.join(EXPERIMENTS_DIR, "datasets/spider_test_datav2_withNL")

# 结果输出目录
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 覆盖表：当 dataset_name 命中这里时，直接使用手动配置
DATASET_SCHEMA_OVERRIDES = {
    "pg_plfactory": {
        "path": None,  # 若确实没有 dataset json，可设为 None
        "db_type": "postgres",
        "object_type": "procedure",
        "dataset_source": "plfactory",
        "dump_path": "/workspace/opt/projects/plfactory/database/postgresql/plfactory",
        "schema_path": "/workspace/opt/projects/plfactory/data/schema/plfactory",
        "schema_graph_path": "/workspace/opt/projects/plfactory/data/schema/plfactory/postgres_db_schema_graph.json",
        "schema_dict_path": "/workspace/opt/projects/plfactory/data/postgres_db_schema_dict.json",
    },
    "oc_plfactory": {
        "path": None,
        "db_type": "oracle",
        "object_type": "procedure",
        "dataset_source": "plfactory",
        "dump_path": "/workspace/opt/projects/plfactory/database/oracle/plfactory",
        "schema_path": "/workspace/opt/projects/plfactory/data/schema/plfactory",
        "schema_graph_path": "/workspace/opt/projects/plfactory/data/schema/plfactory/oracle_db_schema_graph.json",
        "schema_dict_path": "/workspace/opt/projects/plfactory/data/oracle_db_schema_dict.json",
    },
}


def get_dataset_file_path(dataset_name: str) -> str:
    """
    根据数据集名称自动生成数据集文件路径
    
    Args:
        dataset_name: 数据集名称 (如: postgres_spider_function_test)
        
    Returns:
        数据集文件的完整路径
    """
    filename = f"{dataset_name}.json"
    return os.path.join(DATASETS_DIR, filename)


# ==========================================
# 模型配置
# ==========================================

# GPT-4o 配置
gpt4o_config = {
    "type": "openai",
    "model": "gpt-4o",
    "api_key": os.getenv("LAOZHANG_API_KEY"),
    "base_url": "http://api-one.laozhang.ai:52100/v1"
}

# GPT-4o-mini 配置
gpt4o_mini_config = {
    "type": "openai",
    "model": "gpt-4o-mini",
    "api_key": os.getenv("LAOZHANG_API_KEY"),
    "base_url": "http://api-one.laozhang.ai:52100/v1"
}

# DeepSeek 配置
deepseek_config = {
    "type": "deepseek",
    "model": "deepseek-chat",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": "https://api.deepseek.com",
}

# Gemini 配置
gemini_config = {
    "type": "gemini",
    "model": "gemini-2.5-flash-nothinking",
    "api_key": os.getenv("LAOZHANG_API_KEY"),
    "base_url": "http://api-one.laozhang.ai:52100/v1"
}

# Claude 配置
claude4_config = {
    "type": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "api_key": os.getenv("LAOZHANG_API_KEY"),
    "base_url": "http://api-one.laozhang.ai:52100/v1"
}

# GLM 配置
glm4_config = {
    "type": "glm",
    "model": "glm-4.6",
    "api_key": os.getenv("GLM_API_KEY"),
    "base_url": "https://open.bigmodel.cn/api/paas/v4/"
}

# Qwen 配置
qwen3_config = {
    "type": "qwen3",
    "model": "qwen3-30b",
    "api_key": "EMPTY",
    "base_url": "http://10.194.64.43:8999/v1"
}

# 模型名称到配置的映射
MODELS_CONFIG = {
    "gpt-4o": gpt4o_config,
    "gpt-4o-mini": gpt4o_mini_config,
    "deepseek": deepseek_config,
    "gemini": gemini_config,
    "claude-4": claude4_config,
    "glm-4": glm4_config,
    "qwen3": qwen3_config,
}

# LLM 调用配置
LLM_CALL_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 8192,
    "top_p": 0.3,
}

# ==========================================
# 数据库路径配置
# ==========================================
DATABASE_DUMP_PATH = "/workspace/opt/projects/plfactory/database/postgresql/"
DATABASE_SCHEMA_PATH = "/workspace/opt/projects/plfactory/data/postgres_db_schema_dict.json"


def extract_db_type_from_dataset_name(dataset_name: str) -> str:
    """
    从数据集名称中提取数据库类型
    
    Args:
        dataset_name: 数据集名称 (如: postgres_spider_function_test)
        
    Returns:
        数据库类型: 'postgres' 或 'oracle'
    """
    if dataset_name.startswith("pg"):
        return "postgres"
    elif dataset_name.startswith("oc"):
        return "oracle"
    else:
        raise ValueError(f"无法从数据集名称 '{dataset_name}' 中识别数据库类型")


def extract_object_type_from_dataset_name(dataset_name: str) -> str:
    """
    从数据集名称中提取对象类型
    
    Args:
        dataset_name: 数据集名称 (如: postgres_spider_function_test)
        
    Returns:
        对象类型: 'function', 'procedure', 'trigger' 或 'mixed'
    """
    name_lower = dataset_name.lower()
    if "function" in name_lower:
        return "function"
    elif "procedure" in name_lower:
        return "procedure"
    elif "trigger" in name_lower:
        return "trigger"
    else:
        raise ValueError(f"无法从数据集名称中提取对象类型")


def extract_dataset_source_from_name(dataset_name: str) -> str:
    """
    从数据集名称中提取数据集来源
    
    Args:
        dataset_name: 数据集名称 (如: postgres_spider_function_test)
        
    Returns:
        数据集来源: 'spider', 'bird' 等
    """
    # 数据集名称格式: {database_type}_{dataset_name}_{object_type}
    # 例如: postgres_spider_function_test
    parts = dataset_name.split('_')
    if len(parts) >= 2:
        # 第二部分应该是数据集来源
        return parts[1]
    else:
        raise ValueError(f"无法从数据集名称 '{dataset_name}' 中提取数据集来源")


def get_database_dump_path(dataset_name: str) -> str:
    """
    根据数据集名称生成数据库dump文件路径
    
    Args:
        dataset_name: 数据集名称 (如: postgres_spider_function_test)
        
    Returns:
        数据库dump路径: 如 /path/to/database/spider/postgres
        
    Raises:
        ValueError: 如果环境变量未设置或数据集名称格式错误
    """
    if not DATABASE_DUMP_PATH:
        raise ValueError("环境变量 DATABASE_DUMP_PATH 未设置")
    
    dataset_source = extract_dataset_source_from_name(dataset_name)
    db_type = extract_db_type_from_dataset_name(dataset_name)
    
    # 构建路径: DATABASE_DUMP_PATH/spider/postgres
    dump_path = os.path.join(DATABASE_DUMP_PATH, dataset_source, db_type)
    
    return dump_path


def get_database_schema_path(dataset_name: str) -> str:
    """
    根据数据集名称生成数据库schema文件路径
    
    Args:
        dataset_name: 数据集名称 (如: postgres_spider_function_test)
        
    Returns:
        数据库schema路径: 如 /path/to/schema/spider
        
    Raises:
        ValueError: 如果环境变量未设置或数据集名称格式错误
    """
    if not DATABASE_SCHEMA_PATH:
        raise ValueError("环境变量 DATABASE_SCHEMA_PATH 未设置")
    
    dataset_source = extract_dataset_source_from_name(dataset_name)
    
    # 构建路径: DATABASE_SCHEMA_PATH/spider
    schema_path = os.path.join(DATABASE_SCHEMA_PATH, dataset_source)
    
    return schema_path


def get_database_schema_graph_path(dataset_name: str) -> str:
    """
    根据数据集名称生成数据库schema graph文件路径
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        schema graph文件完整路径
    """
    schema_path = get_database_schema_path(dataset_name)
    db_type = extract_db_type_from_dataset_name(dataset_name)
    
    # 文件名: {db_type}_db_schema_graph.json
    filename = f"{db_type}_db_schema_graph.json"
    
    return os.path.join(schema_path, filename)


def get_database_schema_dict_path(dataset_name: str) -> str:
    """
    根据数据集名称生成数据库schema dict文件路径
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        schema dict文件完整路径
    """
    schema_path = get_database_schema_path(dataset_name)
    db_type = extract_db_type_from_dataset_name(dataset_name)
    
    # 文件名: {db_type}_db_schema_dict.json
    filename = f"{db_type}_db_schema_dict.json"
    
    return os.path.join(schema_path, filename)


def get_dataset_config(dataset_name: str) -> dict:
    """
    根据数据集名称获取数据集配置（自动生成）
    
    Args:
        dataset_name: 数据集名称 (如: postgres_spider_function_test)
        
    Returns:
        数据集配置字典，包含:
        - path: 数据集文件路径
        - db_type: 数据库类型 (postgres/oracle)
        - object_type: 对象类型 (function/procedure/trigger)
        - dataset_source: 数据集来源 (spider/bird等)
        - dump_path: 数据库dump文件目录
        - schema_path: 数据库schema文件目录
        - schema_graph_path: schema graph文件完整路径
        - schema_dict_path: schema dict文件完整路径
        
    Raises:
        ValueError: 如果数据集名称格式错误
        FileNotFoundError: 如果数据集文件不存在
    """
    if dataset_name in DATASET_SCHEMA_OVERRIDES:
        return DATASET_SCHEMA_OVERRIDES[dataset_name].copy()
    # 自动生成数据集文件路径
    dataset_path = get_dataset_file_path(dataset_name)
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"数据集文件不存在: {dataset_path}\n"
            f"请确保文件 '{dataset_name}.json' 存在于 {DATASETS_DIR} 目录中"
        )
    
    # 构建配置字典
    config = {
        "path": dataset_path
    }
    
    # 从名称中提取数据库类型和对象类型
    config["db_type"] = extract_db_type_from_dataset_name(dataset_name)
    config["object_type"] = extract_object_type_from_dataset_name(dataset_name)
    config["dataset_source"] = extract_dataset_source_from_name(dataset_name)
    
    # 添加数据库相关路径
    config["dump_path"] = get_database_dump_path(dataset_name)
    config["schema_path"] = get_database_schema_path(dataset_name)
    config["schema_graph_path"] = get_database_schema_graph_path(dataset_name)
    config["schema_dict_path"] = get_database_schema_dict_path(dataset_name)
    
    return config


def get_model_config(model_name: str) -> dict:
    """
    根据模型名称获取模型配置
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型配置字典
        
    Raises:
        ValueError: 如果模型名称不存在
    """
    if model_name not in MODELS_CONFIG:
        available = ", ".join(MODELS_CONFIG.keys())
        raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {available}")
    
    return MODELS_CONFIG[model_name]


def get_output_path(dataset_name: str, model_name: str, few_shot_num: int) -> str:
    """
    根据数据集名称和模型名称生成输出文件路径
    
    Args:
        dataset_name: 数据集名称
        model_name: 模型名称
        few_shot_num: few-shot示例数量
    Returns:
        输出文件的完整路径
    """
    filename = f"{dataset_name}-{model_name}-{few_shot_num}_shot.json"
    return os.path.join(RESULTS_DIR, filename)


# ==========================================
# 本地模型支持模块
# ==========================================
def is_local_model_path(model_name: str) -> bool:
    """
    判断模型名称是否是本地路径
    
    Args:
        model_name: 模型名称或路径
        
    Returns:
        True 如果是本地路径，False 如果是API模型名称
    """
    # 判断是否包含路径分隔符
    return '/' in model_name or '\\' in model_name


def extract_model_name_from_path(model_path: str) -> str:
    """
    从模型路径中提取模型名称（最后一个文件夹名）
    
    Args:
        model_path: 模型路径
        
    Returns:
        模型名称
    """
    # 去除尾部的路径分隔符
    model_path = model_path.rstrip('/\\')
    # 提取最后一个文件夹名
    return os.path.basename(model_path)


def load_local_model(model_path: str):
    """
    加载本地模型和tokenizer
    
    Args:
        model_path: 模型路径
        
    Returns:
        (model, tokenizer) 元组
    """
    print(f"  加载本地模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  模型加载完成")
    
    return model, tokenizer


def generate_with_local_model(model, tokenizer, prompt_text: str, max_new_tokens: int = 256, num_beams: int = 4, num_return_sequences: int = 4):
    """
    使用本地模型生成文本
    
    Args:
        model: 加载的模型
        tokenizer: 加载的tokenizer
        prompt_text: 输入文本
        max_new_tokens: 最大生成token数
        num_beams: beam search的beam数量
        num_return_sequences: 返回的生成序列数量
        
    Returns:
        生成的文本列表
    """
    # 编码输入
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # 生成
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
    
    # 解码输出（只解码生成的部分，不包括输入）
    generated_texts = tokenizer.batch_decode(
        generate_ids[:, input_length:], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return generated_texts

# extract the skeleton of the input text
def extract_skeleton(text):
    tokens_and_tags = nltk.pos_tag(nltk.word_tokenize(text))

    output_tokens = []
    for token, tag in tokens_and_tags:
        if tag in ['NN', 'NNP', 'NNS', 'NNPS', 'CD', 'SYM', 'FW', 'IN']:
            output_tokens.append("_")
        elif token in ['$', "''", '(', ')', ',', '--', '.', ':']:
            pass
        else:
            output_tokens.append(token)
    
    text_skeleton = " ".join(output_tokens)
    text_skeleton = text_skeleton.replace("_ 's", "_")
    text_skeleton = text_skeleton.replace(" 's", "'s")

    while("_ _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ _", "_")
    while("_ , _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ , _", "_")
    
    if text_skeleton.startswith("_ "):
        text_skeleton = text_skeleton[2:]
    
    return text_skeleton

def generate_skeletons(skeleton_predictor_path, plsql_texts):

    # 确保模型和tokenizer已经加载
    model_dir = skeleton_predictor_path
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    skeletons = []

    for plsql_text in plsql_texts:
        # 将输入文本转换为模型的输入格式
        input_encoding = tokenizer(
            plsql_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(device)

        # 使用模型生成输出
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_encoding['input_ids'],
                attention_mask=input_encoding['attention_mask'],
                max_length=512,
                min_length=50,
                num_beams=2,
                early_stopping=True
            )

        # 解码并添加到skeletons列表
        skeleton = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        skeletons.append(skeleton)

    return skeletons

def normalize_similarities(similarities):
    # 二维 numpy 数组
    min_val = np.min(similarities)
    max_val = np.max(similarities)

    normalized_similarities = (similarities - min_val) / (max_val - min_val)

    return normalized_similarities