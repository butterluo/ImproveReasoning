import sys, os
import logging
logger = logging.getLogger(__name__)

import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

################
# Model Loading
################
model_name = "microsoft/Phi-3.5-mini-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

##################
# Data Processing
##################
def apply_chat_template(example, tokenizer):
    prompt = tokenizer.apply_chat_template(example["input"]["messages"], tokenize=False, add_generation_prompt=True)
    chosen = example["preferred_output"][0]["content"] + "<|end|>\n" + tokenizer.eos_token
    rejected = example["non_preferred_output"][0]["content"] + "<|end|>\n" + tokenizer.eos_token
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
raw_dataset = load_dataset('json',data_files=os.path.normpath(os.path.join(os.path.abspath(__file__),'..','..','..','data/evolvemcts4rl/samples/math_500_tst.3e7b.flat.dpo.json')), split='train')
splited_ds = raw_dataset.train_test_split(test_size=0.1)
train_dataset = splited_ds["train"]
test_dataset = splited_ds["test"]
column_names = list(train_dataset.features)
dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to train_dpo",
)
dataset_eval = test_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to test_dpo",
)

###################
# Hyper-parameters
###################
peft_config = LoraConfig(
    r=8,#16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    #target_modules=['o_proj', 'qkv_proj'] #phi-3
    target_modules="all-linear",
    modules_to_save = None
)
training_args = DPOConfig(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    remove_unused_columns=True,
    learning_rate=5.0e-06,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    save_strategy="epoch",
    log_level="info",
    logging_steps=1,
    output_dir="./chkp_dir/dpo1230_1/",
    overwrite_output_dir = True,
    optim="paged_adamw_32bit",
    warmup_steps=2,
    bf16=True,
    report_to="none",
    do_eval=False,
    max_steps = -1,
    save_steps = 1, #100,
    save_total_limit = 2, #1,
    seed = 0,
    warmup_ratio = 0.2,
    beta=0.1,
    max_prompt_length=2048,
    max_length=2048,

)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)], #, logging.FileHandler(training_args.output_dir)
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")
logger.info(f"PEFT parameters {peft_config}")

###########
# Training
###########
trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_eval,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

# Fine-tune model with DPO
train_result = trainer.train()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_model()

#############
# Evaluation
#############
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(dataset_eval)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)