import openai
import json
import concurrent.futures
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# 配置 vLLM 的 OpenAI 兼容 API 端点
openai.base_url = "http://localhost:8000/v1/"
openai.api_key = "xxx"  

# 提示词模板保持不变
PROMPT_TEMPLATE = """你是一位法律领域的资深专家，需要回答一个高难度的多跳推理问题，这个多跳问题是由一法律图谱链路构造的。

图谱的链路信息：
{record_info}

问题：
{question}

回答要求：
1. 图谱的链路信息中含有问题的答案，但是要给出回答的推理过程
2. 推理过程要详细，且符合逻辑
3. 你的回答包括思考从头到尾不要含有根据图谱的链路信息之类的词
4. 你是已经知道图谱链路的信息了，不用再说根据图谱链路信息，改成说根据我的知识
5. 你的回答包括思考从头到尾不要含有根据图谱的链路信息、回顾图谱中的链路信息、回忆图谱中的链路信息之类的词，这些你已经知道了，必须要说成根据我的知识、回顾我的知识、回忆我的知识
"""
# PROMPT_TEMPLATE = """You are a senior expert in linguistics, tasked with answering a highly challenging multi-hop reasoning question derived from a linguistic knowledge graph.

# Knowledge graph path information:
# {record_info}

# Question:
# {question}

# Instructions for your response:
# 1. The answer is embedded within the provided knowledge graph path, but you must articulate a clear, step-by-step reasoning process to arrive at it.
# 2. Your reasoning should be thorough, logically sound, and grounded in linguistic principles.
# 3. Do **not** mention phrases like "according to the knowledge graph," "based on the graph path," or any reference to external graph data.
# 4. You already possess this information as part of your expertise—frame all reasoning as coming from your own knowledge (e.g., "Based on my knowledge," "From what I understand," "Recalling my knowledge").
# 5. Avoid any wording that suggests you are consulting, retrieving, or referencing an external graph. Instead, present the information as inherent to your expert understanding of linguistics.
# """

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件到字典列表"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

def process_record(record: Dict) -> Tuple[str, str, Optional[str]]:
    """处理单条记录，返回(问题, 答案, 错误信息)"""
    question = record["generated_question"]
    record_info = record["original_record"]
    
    # 构建提示词
    full_prompt = PROMPT_TEMPLATE.format(
        record_info=record_info,
        question=question
    )
    
    try:
        # 调用vLLM模型
        response = openai.chat.completions.create(
            model="your model name",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
            max_tokens=4096,
            stream=False
        )
        answer = response.choices[0].message.content.strip()
        return (question, answer, None)
    except Exception as e:
        return (question, None, str(e))

def main():
    input_file = "your input file"
    output_file = "your output file"
    
    # 加载所有记录
    records = load_jsonl(input_file)
    total_records = len(records)
    print(f"Loaded {total_records} records from {input_file}")
    
    # 使用线程池并行处理
    results = []
    # 使用tqdm显示进度条
    with tqdm(total=total_records, desc="Processing records", unit="record") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # 提交所有任务
            futures = [executor.submit(process_record, record) for record in records]
            
            # 收集结果并更新进度
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    
    # 写入结果到新JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for question, answer, error in results:
            if error:
                # 处理错误记录
                result = {
                    "question": question,
                    "error": error
                }
            else:
                # 正常结果
                result = {
                    "question": question,
                    "answer": answer
                }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 统计成功和错误数量
    success_count = sum(1 for _, _, error in results if error is None)
    error_count = total_records - success_count
    print(f"\nProcessing completed! {success_count} records processed successfully, {error_count} errors.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()