llama 3.1 파인튜닝
WanDB API 필요: https://wandb.ai/site/ko/

https://github.com/unslothai/unsloth

!pip install unsloth
# Unsloth 라이브러리에서 FastLanguageModel을 임포트
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # 최대 시퀀스 길이를 설정 ( 텍스트의 최대 길이를 지정)
dtype = None  # 자동 감지를 위해 None 설정. Tesla T4는 Float16, Ampere+는 Bfloat16 사용. 모델의 파라미터를 저장할 데이터 타입
load_in_4bit = True  # 메모리 사용량을 줄이기 위해 4비트 양자화 사용. 다만 양자화에 따른 손실이 있어서 필요에 따라 False로 설정 가능

# 4비트 사전 양자화된 모델 리스트 (더 빠른 다운로드 및 OOM 방지)
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 8B 모델, 4비트 양자화
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 405B 모델도 4비트 지원
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # Mistral 12B 모델, 4비트 지원
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 7B 모델, 4비트 지원
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 미니 인스트럭트 모델
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2.27B 모델, 4비트 지원
] # 더 많은 모델은 https://huggingface.co/unsloth 에서 확인 가능

# 사전 학습된 모델과 토크나이저 로드
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",  # 사용할 모델 이름
    max_seq_length = max_seq_length,         # 설정한 최대 시퀀스 길이
    dtype = dtype,                           # 데이터 타입 설정
    load_in_4bit = load_in_4bit,             # 4비트 양자화 여부
)

# PEFT(파라미터 효율적 파인튜닝) 모델 설정
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA 랭크 설정. 8, 16, 32, 64, 128 권장. r 값이 클수록 모델이 더 많은 정보를 학습할 수 있지만, 너무 크면 메모리를 많이 사용
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],  # PEFT 적용할 모듈 목록. 모델의 특정 부분(모듈)에만 학습
    lora_alpha = 16,        # LoRA 알파 설정 LoRA라는 기술이 얼마나 강하게 작용할지 조절
    lora_dropout = 0,       # LoRA 드롭아웃 설정. 0으로 최적화
    bias = "none",          # 바이어스 설정. "none"으로 최적화

    # "unsloth" 사용 시 VRAM 절약 및 배치 사이즈 2배 증가
    # 학습할 때 메모리를 절약하는 방법을 사용하는 설정
    use_gradient_checkpointing = "unsloth",  # 매우 긴 컨텍스트를 위해 "unsloth" 설정
    random_state = 3407,    # 랜덤 시드 설정
    use_rslora = False,     # 랭크 안정화 LoRA 사용 여부
    loftq_config = None,    # LoftQ 설정 (사용하지 않음)
)

# 모델에게 주어질 텍스트의 형식을 정의
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# 맨윗줄: 모델에게 앞으로 제공될 지시사항을 기반으로 적절한 응답을 작성하라고 말함.
# 모델이 어떤 작업을 수행해야 하는지 명확히 이해할 수 있도록 도와줌
# Instruction:은 지시사항이 제공될 부분
# Response:는 모델이 생성해야 할 응답이 제공될 부분

# EOS 토큰 가져오기 (생성 종료를 위해 필요)
EOS_TOKEN = tokenizer.eos_token  # 반드시 EOS_TOKEN을 추가해야 함
EOS_TOKEN

# 문자열에 값을 넣는 방법
1. f-format

f"변수의 값은 = {변수}"

2. format 함수
"변수의 값은 ={}".format(변수)

3. 형식화된 출력
"변수의 값은 =%s"%(변수)

# 프롬프트 포맷팅 함수 정의
def formatting_prompts_func(examples):
    instructions = examples["instruction"]  # 데이터셋의 'instruction' 필드
    outputs      = examples["output"]       # 데이터셋의 'output' 필드
    texts = []
    for instruction, output in zip(instructions, outputs):
                                                           # EOS_TOKEN을 추가하지 않으면 생성이 무한히 계속됨
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN  # 프롬프트 형식에 맞게 텍스트 생성
        texts.append(text)
    return { "text" : texts, }  # 'text' 필드로 반환

from datasets import load_dataset  # Hugging Face datasets 라이브러리 임포트

# 데이터셋 로드 (Teddy Lee의 QA 데이터셋 미니 버전)
dataset = load_dataset("teddylee777/QA-Dataset-mini", split = "train")

# 프롬프트 포맷팅 함수 적용하여 데이터셋 변환
dataset = dataset.map(formatting_prompts_func, batched = True,)


# 예제 엑셀 형태로 데이터셋을 만든다면?
# from datasets import load_dataset
# dataset = load_dataset( "csv", data_files = "data.csv", split = "train")
# dataset = dataset.map(formatting_prompts_func, batched=True)
#

dataset['text'][:5]

"""출력 [['Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n바이든 대통령이 발표한 행정명령에서 AI 시스템의 안전성과 신뢰성을 확인하기 위해 어떤 조치를 추진하고 있습니까?\n\n### Response:\n바이든 대통령이 발표한 행정명령에서는 강력한 AI 시스템을 개발하는 기업에게 안전 테스트 결과와 시스템에 관한 주요 정보를 미국 정부와 공유할 것을 요구하고, AI 시스템의 안전성과 신뢰성 확인을 위한 표준 및 AI 생성 콘텐츠 표시를 위한 표준과 모범사례 확립을 추진하고 있습니다.<|end_of_text|>',
 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nG7 국가들이 채택한 AI 국제 행동강령에 따르면, 첨단 AI 시스템의 개발 과정에서 어떤 조치를 취해야 합니까?\n\n### Response:\nG7 국가들은 첨단 AI 시스템의 개발 과정에서 AI 수명주기 전반에 걸쳐 위험을 평가 및 완화하는 조치를 채택해야 합니다.<|end_of_text|>',
 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n한국과 영국 정부가 공동 개최하기로 합의한 AI 미니 정상회의는 언제 온라인으로 개최될 예정입니까?\n\n### Response:\n한국과 영국 정부는 6개월 뒤에 온라인으로 AI 미니 정상회의를 공동 개최하기로 합의했습니다.<|end_of_text|>',
 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nFTC가 생성 AI와 관련하여 어떤 유형의 위험에 주목하고 있습니까?\n\n### Response:\nFTC는 생성 AI의 사용과 관련하여 소비자의 개인정보 침해, 차별과 편견의 자동화, 사기 범죄 등의 위험에 주목하고 있습니다.<|end_of_text|>',
 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n프런티어 모델 포럼이 AI 안전 기금을 조성하는 주된 목적은 무엇입니까?\n\n### Response:\n프런티어 모델 포럼이 AI 안전 기금을 조성하는 주된 목적은 AI 모델의 취약점을 발견하고 검증하는 레드팀 활동을 지원하기 위함입니다.<|end_of_text|>']]"""

## 학습 설정
from trl import SFTTrainer  # TRL 라이브러리에서 SFTTrainer 임포트
from transformers import TrainingArguments  # 트랜스포머 라이브러리에서 TrainingArguments 임포트
from unsloth import is_bfloat16_supported  # BFloat16 지원 여부 확인 함수 임포트

# SFTTrainer 인스턴스 생성
trainer = SFTTrainer(
    model = model,                           # 학습할 모델
    tokenizer = tokenizer,                   # 사용할 토크나이저
    train_dataset = dataset,                 # 학습할 데이터셋 ★★★★★★★★
    dataset_text_field = "text",             # 데이터셋의 텍스트 필드 이름 ★★★★★★★★
    max_seq_length = max_seq_length,         # 최대 시퀀스 길이
    dataset_num_proc = 2,                    # 데이터셋 전처리에 사용할 프로세스 수 cpu
    packing = False,                         # 짧은 시퀀스의 경우 packing을 비활성화 (학습 속도 5배 향상 가능)
    args = TrainingArguments(
        per_device_train_batch_size = 2,     # 디바이스 당 배치 사이즈
        gradient_accumulation_steps = 4,     # 그래디언트 누적 단계 수
        warmup_steps = 5,                     # 워밍업 스텝 수
        # num_train_epochs = 1,               # 전체 학습 에폭 수 설정 가능
        max_steps = 60,                       # 최대 학습 스텝 수
        learning_rate = 2e-4,                 # 학습률
        fp16 = not is_bfloat16_supported(),   # BFloat16 지원 여부에 따라 FP16 사용
        bf16 = is_bfloat16_supported(),       # BFloat16 사용 여부
        logging_steps = 1,                    # 로깅 빈도
        optim = "adamw_8bit",                  # 옵티마이저 설정 (8비트 AdamW)
        weight_decay = 0.01,                  # 가중치 감쇠
        lr_scheduler_type = "linear",         # 학습률 스케줄러 타입
        seed = 3407,                           # 랜덤 시드 설정
        output_dir = "outputs",                # 출력 디렉토리
    ),
)

## 학습 실행
trainer_stats = trainer.train()  # 모델 학습 시작 # 3d6e4b9a0ddcd9edfafad59dcde0fd8c666954fd
