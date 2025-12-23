# 安装并导入依赖（如未安装 neo4j 请先在终端运行：pip install neo4j）
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

from neo4j import GraphDatabase

# 属性清洗：支持列表与复杂类型，确保 Neo4j 兼容
def sanitize_value(val):
    import json

    if isinstance(val, (str, int, float, bool)):
        return val
    if val is None:
        return None  # 单值属性允许 None；数组场景在列表处理中转换

    if isinstance(val, (list, tuple, set)):
        arr = [sanitize_value(v) for v in list(val)]
        # Neo4j 数组不允许包含 null，转换为字符串 'null'
        arr = ['null' if v is None else v for v in arr]
        # 若数组中存在非原始类型（比如列表或字典），则整体序列化为字符串
        if not all(isinstance(v, (str, int, float, bool)) for v in arr):
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)
        return arr

    if isinstance(val, dict):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)

    # 兼容可能的 datetime 等
    if hasattr(val, "isoformat"):
        try:
            return val.isoformat()
        except Exception:
            pass

    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def sanitize_props(props: dict) -> dict:
    # 过滤 None 值，并清洗剩余属性
    return {k: sanitize_value(v) for k, v in (props or {}).items() if v is not None}


# 工具：将可能是字符串/列表/其它格式的值统一为字符串列表
def ensure_name_list(val) -> List[str]:
    names: List[str] = []
    if val is None:
        return names
    if isinstance(val, str):
        v = val.strip()
        if v:
            names.append(v)
        return names
    if isinstance(val, (list, tuple, set)):
        for item in val:
            if isinstance(item, str):
                v = item.strip()
                if v:
                    names.append(v)
            elif isinstance(item, dict):
                # 如果是字典，尝试取 name 字段，否则转字符串
                n = item.get("name")
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())
                else:
                    s = str(item).strip()
                    if s:
                        names.append(s)
            else:
                s = str(item).strip()
                if s:
                    names.append(s)
        return names
    if isinstance(val, dict):
        n = val.get("name")
        if isinstance(n, str) and n.strip():
            names.append(n.strip())
            return names
        s = str(val).strip()
        if s:
            names.append(s)
        return names
    # 其他类型直接转字符串
    s = str(val).strip()
    if s:
        names.append(s)
    return names


# 统一实体名称解析：优先 attributes.name，其次各专属 name 字段，最后回退 text
def resolve_name(attrs: Dict[str, Any], text: str | None) -> str | None:
    for key in (
        "name",
        "concept_name",
        "theory_name",
        "term_name",
        "method_name",
        "work_name",
        "school_name",
        "institution_name",
        "technology_name",
        "author",
        "authors",  # 可能是列表
    ):
        v = attrs.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (list, tuple, set)):
            # 取第一个非空字符串，或者用逗号拼接
            flat = [s for s in ensure_name_list(v) if isinstance(s, str) and s.strip()]
            if flat:
                return flat[0]
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


# 读取与规范化 JSON，分出实体与关系并做属性清洗；支持 source/target/relation 为列表
def normalize_items(items: List[Dict[str, Any]]):
    entities_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    name_to_label: Dict[str, str] = {}
    all_labels: Set[str] = set()

    rel_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for item in items:
        cls = item.get("class")
        attrs = dict(item.get("attributes") or {})
        doc_id = item.get("document_id")
        text = item.get("text")

        if cls == "Entity":
            label = attrs.get("entity_type") or "Entity"
            name = resolve_name(attrs, text)
            if not name:
                # 跳过无法确定名字的实体
                continue

            # props 去除类型与 name 字段
            props = {k: v for k, v in attrs.items() if k not in ("entity_type", "name")}
            # 去掉专属 name 字段，避免重复
            for k in (
                "concept_name",
                "theory_name",
                "term_name",
                "method_name",
                "work_name",
                "school_name",
                "institution_name",
                "award_name",
                "technology_name",
                "author",
                "authors",
            ):
                props.pop(k, None)

            # 溯源字段
            if doc_id:
                props.setdefault("document_id", doc_id)
            if text:
                props.setdefault("source_text", text)

            # 属性清洗
            props = sanitize_props(props)

            entities_by_label[label].append({"name": name, "props": props})
            name_to_label[name] = label
            all_labels.add(label)

        elif cls == "Relation":
            rel_type_val = attrs.get("relation_type")
            rel_types = ensure_name_list(rel_type_val)
            src_names = ensure_name_list(attrs.get("source_entity"))
            tgt_names = ensure_name_list(attrs.get("target_entity"))
            # 只要任一为空，就跳过该条
            if not rel_types or not src_names or not tgt_names:
                continue

            # 关系属性（剔除三要素）
            rel_props = {
                k: v
                for k, v in attrs.items()
                if k not in ("relation_type", "source_entity", "target_entity")
            }
            # 溯源字段
            if doc_id:
                rel_props.setdefault("document_id", doc_id)
            if text:
                rel_props.setdefault("source_text", text)

            # 属性清洗（避免 Map 类型错误）
            rel_props = sanitize_props(rel_props)

            # 展开列表：所有 src × tgt × rel_type 组合
            for rt in rel_types:
                for src in src_names:
                    for tgt in tgt_names:
                        src_label = name_to_label.get(src, "Entity")
                        tgt_label = name_to_label.get(tgt, "Entity")
                        all_labels.add(src_label)
                        all_labels.add(tgt_label)
                        rel_groups[(src_label, tgt_label, rt)].append(
                            {
                                "source_name": src,
                                "target_name": tgt,
                                "rel_props": rel_props,
                            }
                        )

    return entities_by_label, rel_groups, all_labels


# 可选：为每个标签创建 name 唯一约束，提升 MERGE 性能
def create_unique_constraints(session, labels: Set[str]):
    for label in labels:
        # 兼容老语法（Neo4j 3.x/4.x）
        try:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS ON (n:`{label}`) ASSERT n.name IS UNIQUE")
            continue
        except Exception:
            pass
        # 兼容新语法（Neo4j 5.x）
        try:
            session.run(
                f"CREATE CONSTRAINT `{label}_name_unique` IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
            )
        except Exception:
            # 无法创建约束则跳过（仍可导入，只是 MERGE 速度可能较慢）
            pass


# 批量导入实体（按标签分组 UNWIND）
def import_entities(session, entities_by_label: Dict[str, List[Dict[str, Any]]]) -> int:
    total = 0
    for label, rows in entities_by_label.items():
        if not rows:
            continue
        q = f"""
        UNWIND $rows AS row
        MERGE (n:`{label}` {{name: row.name}})
        SET n += row.props
        """
        session.run(q, rows=rows)
        total += len(rows)
    return total


# 批量导入关系（按 起点标签/终点标签/关系类型 分组 UNWIND）
def import_relations(
    session,
    rel_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
) -> int:
    total = 0
    for (src_label, tgt_label, rel_type), rows in rel_groups.items():
        if not rows:
            continue
        q = f"""
        UNWIND $rows AS row
        MERGE (a:`{src_label}` {{name: row.source_name}})
        MERGE (b:`{tgt_label}` {{name: row.target_name}})
        MERGE (a)-[r:`{rel_type}`]->(b)
        SET r += row.rel_props
        """
        session.run(q, rows=rows)
        total += len(rows)
    return total

# 数据与连接配置（按需修改）
DATA_PATH = "your data file"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your password")
CREATE_CONSTRAINTS = True  # 是否创建唯一约束

# 读取数据
with open(DATA_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

# 规范化与分组（支持 source/target/relation 为列表）
entities_by_label, rel_groups, all_labels = normalize_items(items)

# 连接并导入
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session() as session:
    if CREATE_CONSTRAINTS:
        create_unique_constraints(session, all_labels)

    node_count = import_entities(session, entities_by_label)
    rel_count = import_relations(session, rel_groups)

driver.close()
print(f"导入完成 -> 节点: {node_count}, 关系: {rel_count}")