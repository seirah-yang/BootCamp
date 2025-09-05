## 본 자료는 테디노트와 위키독스를 참고하였습니다.
## BootCamp 수업자료 참고 

from unsloth import FastLanguageModel
import torch

max_seq_length=2048
dtype = None  # 모델의 파라미터를 저장할 데이터 타입
load_in_4bit = True 

# 최대시퀀스 길이를 설정(텍스트의 최대길이 지정)
# 자동감지를 위해 None지정, Tesla T4는 Float16, Ampere+는 Bfloat16 사용 한다. 
#  메모리 사용량을 줄이기 위해 4비트 양자화 사용하지만, 양자화에 따른 손실이 있어서 필요에 따라 False로 설정 가능


## 4비트 사전 양자화된 모델 리스트 
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
]

# 더 많은 모델은 https://huggingface.co/unsloth 에서 확인 가능


## 사전학습된 모델, 토크나이저 로드 
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name_length=max_seq_length,
  dtype-dtype, 
  load_in_4bit=load_in_4bit,)


## PEFT Parameter Efficient Fine-Tunning Model설정 (중요한 부분!!)
model=FastLanguageModel.get_peft_model(
  model, 
  r=16, 
  target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],  # PEFT 적용할 모듈 목록. 모델의 특정 부분(모듈)에만 학습
    lora_alpha = 16,        # LoRA 알파 설정 LoRA라는 기술이 얼마나 강하게 작용할지 조절
    #  ----------- 다음 아래 내용의 설정은 크게 중요하지 않다. 
    lora_dropout = 0,       # LoRA 드롭아웃 설정. 0으로 최적화
    bias = "none",          # 바이어스 설정. "none"으로 최적화

    # "unsloth" 사용 시 VRAM 절약 및 배치 사이즈 2배 증가
    # 학습할 때 메모리를 절약하는 방법을 사용하는 설정
    use_gradient_checkpointing = "unsloth",  # 매우 긴 컨텍스트를 위해 "unsloth" 설정
    random_state = 3407,    # 랜덤 시드 설정
    use_rslora = False,     # 랭크 안정화 LoRA 사용 여부
    loftq_config = None,    # LoftQ 설정 (사용하지 않음)
  )

alpaca_prompt="""Below is an instruction that describes a task. Write a response that appropriately completes the request.
  ### Instruction:
  {}
  ### Response:
  {}"""

# 맨윗줄: 모델에게 앞으로 제공될 지시사항을 기반으로 적절한 응답을 작성하라고 말함.
# 모델이 어떤 작업을 수행해야 하는지 명확히 이해할 수 있도록 도와줌
# Instruction:은 지시사항이 제공될 부분
# Response:는 모델이 생성해야 할 응답이 제공될 부분

# ★★★★★ EOS 토큰 가져오기 (생성 종료를 위해 필요) ★★★★★
EOS_TOKEN = tokenizer.eos_token  # 반드시 EOS_TOKEN을 추가해야 함
EOS_TOKEN
## >>> '<|end_of_text|>' 로 출력되는 값이 text 마지막으로 붙게 됨 


## 문자열에 값을 넣는 방법
# 1. f-format: f"변수의 값은 = {변수}"
# 2. format 함수 "변수의 값은 ={}".format(변수)
# 3. 형식화된 출력 "변수의 값은 =%s"%(변수)


## 프롬프트 포맷팅 함수 정의 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def formatting_prompts_func(examples):
  instructions=examples["instruction"] 
  outputs = examples["output"]
  texts=[]
  
  for instruction,output in zip(instruction,outputs):        # EOS_TOKEN을 추가하지 않으면 생성이 무한히 계속됨
  text=alpaca_prompt.format(instruction, output) + EOS_TOKEN # 프롬프트 형식에 맞게 텍스트 생성
  texts.append(text)
return{"text":texts}                                         # 'text' 필드로 반환


## HiggingFace datasets Library import
from datasets import load_dataset

dataset=load_dataset("teddylee777/QA-Dataset-mini", split=train)

dataset["instruction"] #  "테디노트 유튜브 채널에 대해서 알려주세요.', '랭체인 관련 튜토리얼은 어디서 찾을 수 있나요?', ..."
dataset["input"]       #  ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
dataset["output"]      #  " '테디노트(TeddyNote)는 데이터 분석, 머신러닝, 딥러닝 등의 주제를 다루는 유튜브 채널입니다. 이 채널을 운영하는 이경록님은 데이터 분석과 인공지능에 대한 다양한 강의를 제공하며, 초보자도 쉽게 따라할 수 있도록 친절하게 설명합니다..."

#instruction + output + <|lend_of_text|> 형식으로 결과갑이 도출 됨 

## 프롬프트 포맷팅 함수 적용하여 데이터셋 변환 
dataset=dataset.map(formatting_prompts_function, batched=True)

# 예제) 엑셀 형태로 데이터셋을 만든다면?
# from datasets import load_dataset
# dataset = load_dataset( "csv", data_files = "data.csv", split = "train")
# dataset = dataset.map(formatting_prompts_func, batched=True)

dataset['text'][:5]


## 학습설정 
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
        per_device_train_batch_size = 2,     # 디바이스 당 배치 사이즈 √
        gradient_accumulation_steps = 4,     # 그래디언트 누적 단계 수√
        warmup_steps = 5,                     # 워밍업 스텝 수√
        # num_train_epochs = 1,               # 전체 학습 에폭 수 설정 가능√
        max_steps = 60,                       # 최대 학습 스텝 수√
        learning_rate = 2e-4,                 # 학습률√
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
trainer_stats = trainer.train() (wandadb APIkey 필요함)


## 모델 저장 로컬폴더에다가 저장하는 방식
model.save_pretrained("lora_model_0726")  # Local saving
tokenizer.save_pretrained("lora_model_0726")

## 이 이외에 허깅페이스나 다른 hub에 push해서 저장하는 방법이 있음
# 다만, 업로드 속도와 다운로드 속도를 고려해야함.

from unsloth import FastLanguageModel
import torch

# 저장된 경로 지정
save_directory = "lora_model_0726"

## 모델과 토크나이저 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = save_directory,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,  # 양자화 옵션을 동일하게 설정
)

## 추론
FastLanguageModel.for_inference(model)

inputs=tokenizer(
  [alpaca=tokenizer(
[alpaca_prompt.format(
       "CES 2024의 주제에 대해서 말해줘.", # 인스트럭션 (명령어)
        "", # 출력 - 생성할 답변을 비워둠 
    )
], return_tensors="pt").to("cuda")       # 텐서를 PyTorch 형식으로 변환하고 GPU로 이동

from transformers import TextStreamer    # 텍스트 스트리밍을 위한 TextStreamer 임포트
text_streamer = TextStreamer(tokenizer)  # 토크나이저를 사용하여 스트리머 초기화


##모델을 사용하여 텍스트 생성 및 스트리밍 출력                                   # 모델을 사용하여 텍스트 생성 및 스트리밍 출력
_=model_generate(**inputs, streamer=text_streamer, max_vew_tokens=128) # 최대 128개의 새로운 토큰 생성
