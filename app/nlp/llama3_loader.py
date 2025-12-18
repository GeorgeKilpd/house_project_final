# HF token 설정
import os
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
# 모델 경량화: Quantization 설정
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 기본 LLaMA 3 모델 로드
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=quantization_config,
    device_map={"": 0}
)

# Tokenizer 설정
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B"
)  # 1. LLaMA3 모델의 사전 학습된 토크나이저 로드

tokenizer.add_special_tokens(
    {"pad_token": "[reserved_special_250]"}
)  # 2. 패딩 토큰 추가 (LLaMA는 기본적으로 PAD 토큰이 없음)

model.config.pad_token_id = tokenizer.pad_token_id
# 3. 모델 설정에 패딩 토큰 반영

questions = [
    "Parameter-Efficient Fine Tuning에 대해서 알려줘",
    "LLM에서 가장 유명한 예시는 뭐가 있어?",
    "What is a famous tall tower in Seoul?",
    "LLM에서 파인튜닝이 뭐야?",
    "운영체제가 뭐하는 거야?",
    "메모리가 뭐야?"
]

# Prompt/Response Format 관련 설정
EOS_TOKEN = tokenizer.eos_token


def convert_to_alpaca_format(instruction, response):
    alpaca_format_str = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
"""
    return alpaca_format_str

# 주어진 명령어(instruction)를 LLaMA 모델에 입력하고, AI가 생성한 답변을 반환하는 함수
# LLM instruction을 기반으로 텍스트를 생성하도록 테스트하는 함수
def test_model(instruction_str, model):
    inputs = tokenizer(
        convert_to_alpaca_format(instruction_str, ""),
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,     # 응답에 생성할 최대 토큰 수
        use_cache=True,         # 캐시 활성화
        temperature=0.05,       # 낮은 값 (즉, 더 결정적인 응답 생성, 창의성 낮음)
        top_p=0.95              # 상위 95% 확률을 가진 토큰만 선택
    )

    return tokenizer.batch_decode(outputs)[0]
# 여러 개의 질문(questions) 목록을 모델에 입력하여 응답을 생성하고, 이를 저장하는 과정을 수행함.
answers_dict = {
    "base_model_answers": []
}

for idx, question in enumerate(questions):
    print(f"Processing EXAMPLE {idx}")
    base_model_output = test_model(question, model)
    answers_dict["base_model_answers"].append(base_model_output)
# 모델이 생성한 응답을 정리하여 출력하는 함수
def simple_format(text, width=120):
    return "\n".join(
        line[i:i+width]
        for line in text.split("\n")
        for i in range(0, len(line), width)
    )


for idx, question in enumerate(questions):
    print(f"EXAMPLE {idx}")
    print(f"Question: {question}")

    print("<Base Model 답변>")
    base_model_answer = answers_dict["base_model_answers"][idx].split("### Response:")[1]
    print(simple_format(base_model_answer))
    print("---")
