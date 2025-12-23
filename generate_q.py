import json
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# 配置
BASE_URL = "http://localhost:8000/v1"
API_KEY = "xxx"  # vLLM 不验证，但需非空
MODEL_NAME = "your model name"
INPUT_JSONL = "your input file"
OUTPUT_JSONL = "your output file"
MAX_CONCURRENT = 24  # 控制最大并发数

PROMPT_TEMPLATE = """你是一位法律领域的资深专家，需要设计一个高难度的多跳推理问题。请根据以下知识图谱信息围绕“rare_node”设计一个问题，要求问题需要至少2个推理步骤才能回答：

知识图谱的信息：
{record_json}

设计要求：
1. 问题涉及中间节点可以利用节点的属性模糊节点的名称，增加问题难度
2. 问题需要回答者进行至少2个推理步骤才能得出正确答案
3. 问题应符合法律专业语境，避免出现明显错误，确保问题不要有多个答案
4. 问题要围绕“rare_node”进行设计。
5. 问题中不要带有“根据参考信息”“根据图谱信息”“结合给定的知识图谱信息”，“根据提供的信息”等之类指代不明的词。你可以把具体的信息说出来，但是不要说根据提供的信息。
6. 问题要具有代表性，问题的答案不要具体到某个人。
一定要仅输出问题本身，不要包含任何解释或答案。"""
# PROMPT_TEMPLATE = """You are a senior expert in linguistics, tasked with designing a challenging multi-hop reasoning question for an advanced linguistics course. Based on the following knowledge graph information, craft a question centered around the "rare_node" that requires at least three reasoning steps to answer.

# Knowledge graph information:
# {record_json}

# Design requirements:
# 1. Obscure key details (e.g., names of scholars, dates, theory names, specific research objects, etc.) to increase difficulty.
# 2. Do not explicitly mention all critical elements from the knowledge chain; instead, hint at them indirectly.
# 3. The question must require at least three distinct reasoning steps to arrive at the correct answer.
# 4. The question should be grounded in professional linguistic context, free of factual errors, and must have a single unambiguous answer.
# 5. The question must focus on the "rare_node," and the correct answer must directly relate to it.
# 6. Do not include phrases such as "according to the provided information," "based on the knowledge graph," or "given the reference data."

# Output only the question itself—no explanations, no answers, and no additional text."""

# 初始化异步客户端（全局）
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

async def generate_question(record):
    record_json_str = json.dumps(record, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.format(record_json=record_json_str)

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,  
            stream=False
        )
        question = response.choices[0].message.content.strip()
    except Exception as e:
        question = f"[EXCEPTION] {str(e)}"

    return {
        "original_record": record,
        "generated_question": question
    }

async def main():
    # 读取所有记录
    records = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records. Starting generation...")

    # 使用信号量控制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def bounded_generate(record):
        async with semaphore:
            return await generate_question(record)

    tasks = [bounded_generate(record) for record in records]
    results = await tqdm.gather(*tasks, desc="Generating questions")

    # 写入输出文件
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for res in results:
            out_f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"✅ Done! Results saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    asyncio.run(main())