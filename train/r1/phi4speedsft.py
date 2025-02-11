import os, sys
from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = "./chkp_dir/phi4sft_0203/"
import torch
from unsloth import FastLanguageModel
# 使用 FastLanguageModel 处理 HuggingFace Hub上的模型，可节省30%内存，加速比近2倍
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "microsoft/phi-4",
    device_map='cuda:0',
    trust_remote_code = True,
    attn_implementation="flash_attention_2",
)
model = FastLanguageModel.get_peft_model(#使用LoRA解决显存不足的问题
    model,
    r = 16, # 建议为 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407, # seed
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from unsloth.chat_templates import get_chat_template
#提供了加强版的chat_template，并解决了某些模型的bug
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4",
)



from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from datasets import load_dataset
# 使用社区中用DeepSeek生成的数据，
dataset = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k")
trainer = SFTTrainer(# 使用HuggingFace的SFTTrainer
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    eval_dataset=dataset['test'],
    max_seq_length = 5120,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        log_level = "info",
        logging_strategy = "steps",
        logging_steps = 5,
        output_dir = OUTPUT_DIR,
        overwrite_output_dir = True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps = 100,
        per_device_eval_batch_size=8,
        save_strategy = "steps", #"best",
        save_steps = 100,
        save_total_limit = 8,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = -1,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        lr_scheduler_type = "cosine", # linear
        seed = 3407,
        report_to = "none", # Use this for WandB etc,
        remove_unused_columns = True,
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(# 训练时只对生成的部分计算loss，进一步加速
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

metrics = trainer_stats.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

trainer.save_model(os.path.join(os.path.abspath(OUTPUT_DIR),'final'))