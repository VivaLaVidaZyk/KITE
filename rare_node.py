from neo4j import GraphDatabase
import pandas as pd

# === 配置区 ===
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "your password")  # ← 替换为你的密码

NODE_TYPE_WEIGHTS = {
    "Law":      1.5,
    "LegalArticle":       1.5,
    "Crime":         1.5,
    "LegalConcept":     1.5,
    "Person":       1.0,
    "Organization":       1.0,
    "Penalty":     1.5,
    "Location":         1.0,
}

MIN_DEGREE = 1
TOP_PERCENT = 0.8

def fetch_rare_nodes():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    with driver.session() as session:
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        OPTIONAL MATCH (n)-[r_out]->()
        WITH n, count(r_out) AS out_degree
        OPTIONAL MATCH (n)<-[r_in]-()
        WITH n, 
             out_degree,
             count(r_in) AS in_degree,
             labels(n) AS labels,
             coalesce(n.detailed_definition, '') <> '' AS has_def,
             coalesce(n.function, '') <> '' AS has_func
        RETURN 
            elementId(n) AS element_id,
            n.name AS name,
            labels,
            in_degree,
            out_degree,
            (in_degree + out_degree) AS total_degree,
            has_def,
            has_func
        """
        
        result = session.run(query)
        records = []
        for rec in result:
            total_deg = rec["total_degree"]
            if total_deg < MIN_DEGREE:
                continue
                
            main_label = rec["labels"][0] if rec["labels"] else "Unknown"
            weight = NODE_TYPE_WEIGHTS.get(main_label, 1.0)
            
            base_score = 1.0 / total_deg
            bonus = 1.0
            if rec["has_def"]:
                bonus += 0.3
            if rec["has_func"]:
                bonus += 0.2
                
            final_score = base_score * weight * bonus
            
            records.append({
                "element_id": rec["element_id"],
                "name": rec["name"],
                "label": main_label,
                "in_degree": rec["in_degree"],
                "out_degree": rec["out_degree"],
                "total_degree": total_deg,
                "has_definition": rec["has_def"],
                "has_function": rec["has_func"],
                "score": final_score
            })
    
    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️ 未找到符合条件的节点，请检查Neo4j连接或节点属性")
        return pd.DataFrame()
        
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    num_top = max(1, int(len(df) * TOP_PERCENT))
    rare_df = df.head(num_top)
    
    rare_df.to_csv("rare_nodes.csv", index=False, encoding='utf-8-sig')
    
    rare_element_ids = rare_df["element_id"].tolist()
    with open("rare_node_element_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rare_element_ids))
    
    print(f"✅ 共处理 {len(df)} 个有效节点，筛选出 {len(rare_df)} 个稀有节点")
    print("\n🔍 Top 5 稀有节点示例 (elementId 格式):")
    print(rare_df[["name", "label", "total_degree", "score", "element_id"]].head())
    
    driver.close()
    return rare_df

if __name__ == "__main__":
    fetch_rare_nodes()