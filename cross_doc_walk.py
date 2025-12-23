from neo4j import GraphDatabase
import json
import pandas as pd
import random
from collections import Counter

# === 配置区 ===
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "your password")  # ← 替换为你的密码

MAX_PATH_LENGTH = 6  # 已调整为6
MIN_PATH_LENGTH = 3  # 已调整为3

def fetch_node_full_properties(element_id):
    """
    通过 elementId 字符串查询节点（兼容Neo4j 5.0+）
    element_id 格式: "4:9980e3d8-5b1d-4799-8699-0cb685a7019f:11257"（与CSV中element_id一致）
    """
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            res = session.run("""
                MATCH (n)
                WHERE elementId(n) = $element_id
                RETURN 
                    elementId(n) AS element_id,
                    labels(n) AS labels,
                    properties(n) AS properties
            """, element_id=element_id)  # 直接使用字符串，不转换
            record = res.single()
            if not record:
                return None
            
            properties = record["properties"] or {}
            
            return {
                "element_id": record["element_id"],
                "labels": list(record["labels"]) if record["labels"] else [],
                "properties": properties
            }

def get_path_with_full_info(rare_element_id):
    """
    从稀有节点反向游走（使用 elementId 字符串作为标识）
    """
    visited = {rare_element_id}  # 存储 elementId 字符串
    path_nodes = []
    path_relations = []

    # 获取稀有节点（终点）
    rare_node = fetch_node_full_properties(rare_element_id)
    if not rare_node:
        return None
    path_nodes.append(rare_node)
    current_element_id = rare_element_id

    # 记录已访问的文档ID
    visited_doc_ids = set()
    doc_id = rare_node["properties"].get("document_id")
    if doc_id:
        visited_doc_ids.add(doc_id)

    # 反向游走（最多5步，因为MAX_PATH_LENGTH=6）
    for _ in range(MAX_PATH_LENGTH - 1):
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                query = """
                MATCH (prev)-[r]->(curr)
                WHERE elementId(curr) = $current_element_id 
                  AND NOT (elementId(prev) IN $visited)
                RETURN 
                    elementId(prev) AS source_id,
                    prev.document_id AS doc_id,
                    type(r) AS rel_type
                """
                result = session.run(
                    query,
                    current_element_id=current_element_id,
                    visited=list(visited)
                )
                candidates = []
                for rec in result:
                    candidates.append({
                        "source_id": rec["source_id"],
                        "doc_id": rec["doc_id"],
                        "rel_type": rec["rel_type"]
                    })

        if not candidates:
            break

        # 计算权重
        weights = []
        lambda_val = 0
        for cand in candidates:
            w = 1.0
            if cand["doc_id"] and cand["doc_id"] not in visited_doc_ids:
                w += lambda_val
            weights.append(w)

        # 加权随机选择
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        chosen_source_id = chosen["source_id"]
        chosen_rel_type = chosen["rel_type"]
        chosen_doc_id = chosen["doc_id"]

        # 获取新节点
        new_node = fetch_node_full_properties(chosen_source_id)
        if not new_node:
            continue

        # 添加到路径开头
        path_nodes.insert(0, new_node)
        path_relations.insert(0, {
            "type": chosen_rel_type,
            "source_id": chosen_source_id,
            "target_id": current_element_id
        })

        visited.add(chosen_source_id)
        if chosen_doc_id:
            visited_doc_ids.add(chosen_doc_id)
        current_element_id = chosen_source_id

    if len(path_nodes) < MIN_PATH_LENGTH:
        return None

    return {
        "path_nodes": path_nodes,
        "path_relations": path_relations,
        "rare_node": path_nodes[-1],
        "path_length": len(path_nodes)
    }

def generate_paths_jsonl():
    """主函数：生成包含全部节点属性的JSONL文件（使用elementId字符串查询）"""
    # 1. 从CSV读取稀有节点（直接使用elementId字符串）
    df_rare = pd.read_csv("rare_nodes.csv", encoding='utf-8-sig')
    rare_element_ids = df_rare["element_id"].tolist()  # 保持字符串格式

    output_file = "sft_paths_full.jsonl"
    valid_count = 0
    path_length_counts = Counter()

    print(f" 开始生成路径（使用elementId字符串查询，兼容Neo4j 5.0+）...")
    print(f"  读取 {len(rare_element_ids)} 个稀有节点（格式: 4:uuid:11257）")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, element_id in enumerate(rare_element_ids):
            # 直接传递字符串，不进行任何转换
            path_data = get_path_with_full_info(element_id)
            
            if path_data is None:
                continue

            # 统计链路长度
            path_length_counts[path_data['path_length']] += 1

            # 写入JSONL
            f_out.write(json.dumps(path_data, ensure_ascii=False) + "\n")
            valid_count += 1

            if (i + 1) % 50 == 0:
                print(f"  处理进度: {i+1}/{len(rare_element_ids)} | 有效路径: {valid_count}")

    print(f"\n✅ 完成！生成 {valid_count} 条路径，保存至: {output_file}")
    print("   无任何Neo4j警告（已使用elementId字符串查询）")
    print("   路径中每个节点包含: element_id, labels, properties（所有自定义属性）")

    print("\n 链路长度分布:")
    for length in sorted(path_length_counts.keys()):
        count = path_length_counts[length]
        print(f"  长度 {length}: {count} 条 ({count/valid_count:.1%})")

    # 打印可读示例
    if valid_count > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            example = json.loads(first_line)
        
        nodes = example["path_nodes"]
        rels = example["path_relations"]
        
        path_names = [node["properties"].get("name", "N/A") for node in nodes]
        path_str = " → ".join(path_names)
        rel_str = " --".join([f"[{r['type']}]" for r in rels]) + "-->"
        

if __name__ == "__main__":
    generate_paths_jsonl()
