import os
import json
import argparse
import textwrap
import traceback
from typing import List, Dict, Tuple
from tqdm import tqdm
import langextract as lx
from langextract.core.data import Document, ExampleData, Extraction
from langextract.extraction import extract
from langextract import factory
import warnings
import logging

# ===== 新增：彻底消除所有警告的配置 =====
# 1. 忽略所有 warnings 模块的警告
warnings.filterwarnings("ignore")

# 2. 设置 absl 日志级别为 ERROR（关键修复！）
try:
    # 确保在导入 langextract 之前设置
    logging.getLogger('absl').setLevel(logging.ERROR)
except Exception:
    pass

# 3. 确保所有日志输出级别为 ERROR
logging.basicConfig(level=logging.ERROR)

# ===== 代码其余部分保持不变 =====
def read_jsonl_file(input_file: str) -> List[Dict]:
    """读取 JSONL 文件"""
    data_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
    return data_list


def build_documents_from_jsonl(data_list: List[Dict]) -> Tuple[List[Document], dict]:
    """将 JSONL 数据转换为 Document 对象"""
    documents = []
    id_to_source = {}
    for idx, item in enumerate(data_list):
        text = item.get("article", "")
        if not text:
            continue       
        source = item.get("law_name", "Unknown")
        title = item.get("law_num", "Untitled")
        # Use title as document_id as requested
        doc_id = title
        # 记录来源和标题到 additional_context
        context_str = f"Source: {source}, Title: {title}"
        doc = Document(
            text=text, 
            document_id=doc_id, 
            additional_context=context_str
        )
        documents.append(doc)
        id_to_source[doc_id] = {
            "title": title,
            "publishSource": source
        }
        
    return documents, id_to_source


def main():
    
    input_file = "your input file"
    output_file = "your output file"
    api_key = "xxx"

    # 配置模型（与 demo.ipynb 一致）
    config = factory.ModelConfig(
        model_id="your model name",
        provider="OpenAILanguageModel",
        provider_kwargs={
            "api_key": api_key,
            "base_url": "http://localhost:8000/v1",
        }
    )
    model = factory.create_model(config)

    prompt = textwrap.dedent("""
    你是一名专业的法律知识图谱构建专家。你的任务是从给定的法律文本（如判决书、法律法规、新闻报道等）中，高效提取法律相关的实体（包括实体属性）和关系。
    
    ## 抽取规则
    **核心实体类型**
    - **Law**: 法律法规名称，如《中华人民共和国刑法》。
    - **LegalArticle**: 法律条文，如“第二百三十条”、“第一千零六十四条”。
    - **Crime**: 罪名，如“盗窃罪”、“诈骗罪”。
    - **LegalConcept**: 法律概念，如“夫妻共同债务”、“正当防卫”。
    - **Person**: 如原告、被告、法官、受害人等。
    - **Organization**: 涉及的机构，如法院、检察院、公司等。
    - **Location**: 地点。
    - **Time**: 关键时间点。
    - **LegalAction**: 法律行为/程序，如“起诉”、“判决”、“逮捕”。
    - **Penalty**: 刑罚/处罚，如“有期徒刑三年”、“罚金”。

    **核心关系类型**
    - **CONVICTED_OF**: 被判处...罪 (Person -> Crime)
    - **SENTENCED_TO**: 被判处...刑罚 (Person -> Penalty)
    - **INVOLVED_IN**: 涉嫌/参与 (Person -> Crime/LegalAction)
    - **AFFILIATED_WITH**: 隶属于/供职于 (Person -> Organization)
    - **LOCATED_AT**: 位于/发生于 (Event/Organization -> Location)
    - **HAPPENED_AT**: 发生于...时间 (Event -> Time)
    - **BASED_ON_LAW**: 依据...法律 (LegalAction/Verdict -> Law/LegalArticle)
    - **DEFINES**: 定义/规定 (LegalArticle -> LegalConcept)

    **智能抽取策略**
    1. **准确性优先**：确保提取的实体和关系在文本中有明确依据。
    2. **属性补全**：尽可能为实体提取属性，例如人物的身份（被告人、原告）、法律的发布机构等。
    """)

    examples = [
        ExampleData(
            text="《中华人民共和国民法典》第一千零六十四条规定：夫妻双方共同签名或者夫妻一方事后追认等共同意思表示所负的债务，属于夫妻共同债务；本规定自2021年1月1日起施行，同时废止《中华人民共和国婚姻法》第十九条关于夫妻共同债务的全部内容。",
            extractions=[
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国民法典》第一千零六十四条规定：夫妻双方共同签名或者夫妻一方事后追认等共同意思表示所负的债务，属于夫妻共同债务；本规定自2021年1月1日起施行，同时废止《中华人民共和国婚姻法》第十九条关于夫妻共同债务的全部内容。",
                    attributes={
                        "entity_type": "Law",
                        "name": "《中华人民共和国民法典》",
                        "legal_effect": "民事基本法"
                    }
                ),
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国民法典》第一千零六十四条规定：夫妻双方共同签名或者夫妻一方事后追认等共同意思表示所负的债务，属于夫妻共同债务；本规定自2021年1月1日起施行，同时废止《中华人民共和国婚姻法》第十九条关于夫妻共同债务的全部内容。",
                    attributes={
                        "entity_type": "LegalArticle",
                        "name": "《中华人民共和国民法典》第一千零六十四条",
                        "law": "《中华人民共和国民法典》",
                        "article_type": "夫妻债务认定"
                    }
                ),
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国民法典》第一千零六十四条规定：夫妻双方共同签名或者夫妻一方事后追认等共同意思表示所负的债务，属于夫妻共同债务；本规定自2021年1月1日起施行，同时废止《中华人民共和国婚姻法》第十九条关于夫妻共同债务的全部内容。",
                    attributes={
                        "entity_type": "LegalConcept",
                        "name": "夫妻共同债务",
                        "definition": "夫妻双方共同签名或事后追认所负的债务"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国民法典》第一千零六十四条规定：夫妻双方共同签名或者夫妻一方事后追认等共同意思表示所负的债务，属于夫妻共同债务；本规定自2021年1月1日起施行，同时废止《中华人民共和国婚姻法》第十九条关于夫妻共同债务的全部内容。",
                    attributes={
                        "relation_type": "DEFINES",
                        "source_entity": "《中华人民共和国民法典》第一千零六十四条",
                        "target_entity": "夫妻共同债务"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国民法典》第一千零六十四条规定：夫妻双方共同签名或者夫妻一方事后追认等共同意思表示所负的债务，属于夫妻共同债务；本规定自2021年1月1日起施行，同时废止《中华人民共和国婚姻法》第十九条关于夫妻共同债务的全部内容。",
                    attributes={
                        "relation_type": "REPEALS",
                        "source_entity": "《中华人民共和国民法典》第一千零六十四条",
                        "target_entity": "《婚姻法》第十九条"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国民法典》第一千零六十四条",
                    attributes={
                        "relation_type": "IS_PART_OF",
                        "source_entity": "《中华人民共和国民法典》第一千零六十四条",
                        "target_entity": "《中华人民共和国民法典》"
                    }
                )
            ]
        ),

        ExampleData(
            text="《中华人民共和国刑法》第二百六十四条规定：盗窃公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；本条文由《刑法修正案（八）》第二十条修订，取代了原《刑法》第二百六十四条中‘数额巨大’的认定标准。",
            extractions=[
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国刑法》第二百六十四条规定：盗窃公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；本条文由《刑法修正案（八）》第二十条修订，取代了原《刑法》第二百六十四条中‘数额巨大’的认定标准。",
                    attributes={
                        "entity_type": "Law",
                        "name": "《中华人民共和国刑法》",
                        "legal_effect": "刑事基本法"
                    }
                ),
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国刑法》第二百六十四条规定：盗窃公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；本条文由《刑法修正案（八）》第二十条修订，取代了原《刑法》第二百六十四条中‘数额巨大’的认定标准。",
                    attributes={
                        "entity_type": "LegalArticle",
                        "name": "《中华人民共和国刑法》第二百六十四条",
                        "law": "《中华人民共和国刑法》",
                        "article_type": "盗窃罪量刑"
                    }
                ),
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国刑法》第二百六十四条规定：盗窃公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；本条文由《刑法修正案（八）》第二十条修订，取代了原《刑法》第二百六十四条中‘数额巨大’的认定标准。",
                    attributes={
                        "entity_type": "LegalConcept",
                        "name": "数额较大",
                        "definition": "盗窃罪的量刑起点标准（原标准被《刑法修正案（八）》修改）"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国刑法》第二百六十四条规定：盗窃公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；本条文由《刑法修正案（八）》第二十条修订，取代了原《刑法》第二百六十四条中‘数额巨大’的认定标准。",
                    attributes={
                        "relation_type": "DEFINES",
                        "source_entity": "《中华人民共和国刑法》第二百六十四条",
                        "target_entity": "数额较大"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国刑法》第二百六十四条规定：盗窃公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；本条文由《刑法修正案（八）》第二十条修订，取代了原《刑法》第二百六十四条中‘数额巨大’的认定标准。",
                    attributes={
                        "relation_type": "AMENDS",
                        "source_entity": "《中华人民共和国刑法》第二百六十四条",
                        "target_entity": "《刑法修正案（八）》第二十条"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国刑法》第二百六十四条",
                    attributes={
                        "relation_type": "IS_SUBORDINATE_TO",
                        "source_entity": "《中华人民共和国刑法》第二百六十四条",
                        "target_entity": "《中华人民共和国刑法》"
                    }
                )
            ]
        ),

        # 示例3：反垄断法与最高法司法解释的解释关系（展示IS_INTERPRETED_BY + IS_APPLICABLE_IN）
        ExampleData(
            text="《中华人民共和国反垄断法》第十七条规定：经营者不得利用数据和算法、技术、资本优势以及平台规则等从事本法禁止的垄断行为；该条款由最高人民法院《关于审理垄断民事纠纷案件适用法律若干问题的规定》第29号司法解释予以细化，适用于平台经济领域的竞争行为监管。",
            extractions=[
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国反垄断法》第十七条规定：经营者不得利用数据和算法、技术、资本优势以及平台规则等从事本法禁止的垄断行为；该条款由最高人民法院《关于审理垄断民事纠纷案件适用法律若干问题的规定》第29号司法解释予以细化，适用于平台经济领域的竞争行为监管。",
                    attributes={
                        "entity_type": "Law",
                        "name": "《中华人民共和国反垄断法》",
                        "legal_effect": "反垄断基础法"
                    }
                ),
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国反垄断法》第十七条规定：经营者不得利用数据和算法、技术、资本优势以及平台规则等从事本法禁止的垄断行为；该条款由最高人民法院《关于审理垄断民事纠纷案件适用法律若干问题的规定》第29号司法解释予以细化，适用于平台经济领域的竞争行为监管。",
                    attributes={
                        "entity_type": "LegalArticle",
                        "name": "《中华人民共和国反垄断法》第十七条",
                        "law": "《中华人民共和国反垄断法》",
                        "article_type": "禁止垄断行为"
                    }
                ),
                Extraction(
                    extraction_class="Entity",
                    extraction_text="《中华人民共和国反垄断法》第十七条规定：经营者不得利用数据和算法、技术、资本优势以及平台规则等从事本法禁止的垄断行为；该条款由最高人民法院《关于审理垄断民事纠纷案件适用法律若干问题的规定》第29号司法解释予以细化，适用于平台经济领域的竞争行为监管。",
                    attributes={
                        "entity_type": "LegalConcept",
                        "name": "垄断行为",
                        "definition": "利用数据、算法、技术、资本优势及平台规则从事的违法行为"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国反垄断法》第十七条规定：经营者不得利用数据和算法、技术、资本优势以及平台规则等从事本法禁止的垄断行为；该条款由最高人民法院《关于审理垄断民事纠纷案件适用法律若干问题的规定》第29号司法解释予以细化，适用于平台经济领域的竞争行为监管。",
                    attributes={
                        "relation_type": "DEFINES",
                        "source_entity": "《中华人民共和国反垄断法》第十七条",
                        "target_entity": "垄断行为"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国反垄断法》第十七条规定：经营者不得利用数据和算法、技术、资本优势以及平台规则等从事本法禁止的垄断行为；该条款由最高人民法院《关于审理垄断民事纠纷案件适用法律若干问题的规定》第29号司法解释予以细化，适用于平台经济领域的竞争行为监管。",
                    attributes={
                        "relation_type": "IS_INTERPRETED_BY",
                        "source_entity": "《中华人民共和国反垄断法》第十七条",
                        "target_entity": "最高人民法院司法解释第29号"
                    }
                ),
                Extraction(
                    extraction_class="Relation",
                    extraction_text="《中华人民共和国反垄断法》第十七条规定：经营者不得利用数据和算法、技术、资本优势以及平台规则等从事本法禁止的垄断行为；该条款由最高人民法院《关于审理垄断民事纠纷案件适用法律若干问题的规定》第29号司法解释予以细化，适用于平台经济领域的竞争行为监管。",
                    attributes={
                        "relation_type": "IS_APPLICABLE_IN",
                        "source_entity": "《中华人民共和国反垄断法》第十七条",
                        "target_entity": "平台经济竞争监管"
                    }
                )
            ]
        )
    ]

    # 读取数据
    print(f"正在读取文件: {input_file}")
    raw_data = read_jsonl_file(input_file)
    print(f"读取到 {len(raw_data)} 条记录")
    
    # 转换为 Document
    documents, id_to_source = build_documents_from_jsonl(raw_data)
    
    # 批处理抽取
    # 分批处理，以免一次性内存占用过大
    BATCH_SIZE = 1
    
    result_items = []
    total_docs = len(documents)
    
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Processing Batches"):
        batch_docs = documents[i : i + BATCH_SIZE]
        
        try:
            annotated_iter = extract(
                text_or_documents=batch_docs,
                prompt_description=prompt,
                examples=examples,
                model=model,
                max_char_buffer=100,
                show_progress=False
            )
            
            for adoc in annotated_iter:
                # 提取 Document 的 metadata
                # 通过 id_to_source 查找元数据
                source_meta = id_to_source.get(adoc.document_id, {})
                
                for e in adoc.extractions or []:
                    attrs = dict(e.attributes or {})
                    result_items.append({
                        "document_id": adoc.document_id,
                        "source_title": source_meta.get("title", "Unknown"),
                        "publish_source": source_meta.get("publishSource", "Unknown"),
                        "class": e.extraction_class,
                        "text": e.extraction_text,
                        "attributes": attrs, 
                    })
        except Exception as e:
            print(f"批次处理出错: {e}")
            traceback.print_exc()
            
    # 汇总输出 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_items, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存至 {output_file}")

if __name__ == "__main__":
    main()
