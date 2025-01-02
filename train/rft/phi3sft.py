import sys, os
os.environ['DISABLE_MLFLOW_INTEGRATION']='TRUE'
import logging

import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

logger = logging.getLogger(__name__)

###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 1, #20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./chkp_dir/241230_2/",#logging_dir default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 32, #4
    "per_device_train_batch_size": 32, #4
    "remove_unused_columns": True,
    "save_steps": 1, #100,
    "save_total_limit": 2, #1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    "max_seq_length": 2048,
    "dataset_text_field": "text",
    "packing":True
    }
peft_config = {
    "r": 8, #16
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}
train_conf = SFTConfig(**training_config)
peft_conf = LoraConfig(**peft_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)], #, logging.FileHandler(train_conf.output_dir)
)

log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")

################
# Model Loading
################
checkpoint_path = "microsoft/Phi-3.5-mini-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

##################
# Data Processing
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    s = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    example["text"] = s
    return example

raw_dataset = load_dataset('json',data_files=os.path.normpath(os.path.join(os.path.abspath(__file__),'..','..','..','data/evolvemcts4rl/samples/math_500_tst.3e7b.flat.sft.json')), split='train')
splited_ds = raw_dataset.train_test_split(test_size=0.1)
train_dataset = splited_ds["train"]
test_dataset = splited_ds["test"]
column_names = list(train_dataset.features)

processed_train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to train_sft",
)
processed_test_dataset = test_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to test_sft",
)

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    tokenizer=tokenizer,
)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


#############
# Evaluation
#############
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# ############
# # Save model
# ############
trainer.save_model(train_conf.output_dir)