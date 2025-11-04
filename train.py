import os

import comet_ml

# Enable logging of model checkpoints
os.environ["COMET_LOG_ASSETS"] = "True"
comet_ml.login(project_name="ARM_LM")

import torch


from tokenizers import Tokenizer
from ml_collections import ConfigDict
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast


from src.train_util import create_datasets_collator, get_model, GenerationLoggerCallback
from src.util import set_global_seed, is_bf16_supported

data_path = "./data/hye_wikipedia_2021_1M/cleaned_corpus_for_training.txt"
base_conf = ConfigDict()
base_conf.seed = 42
base_conf.vocab_size = 88
base_conf.max_steps = 20000


base_conf.context_length = 256
base_conf.n_embd = 768
base_conf.n_layer = 12

base_conf.batch_size = 64
base_conf.gradient_accumulation_steps = 1
base_conf.lr = 3e-4
base_conf.eval_freq = 500
base_conf.save_freq = 10000

tokenizer_path = f"./tokenizers/bpe-vocab_{base_conf.vocab_size}.json"

set_global_seed(base_conf.seed)


tokenizer = Tokenizer.from_file(tokenizer_path)

datasets, data_collator = create_datasets_collator(
    tokenizer=tokenizer, 
    data_path=data_path, 
    tokenizer_path=tokenizer_path, 
    context_length=base_conf.context_length
)
model = get_model(tokenizer=tokenizer, base_conf=base_conf)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
is_bf16 = is_bf16_supported(device=device)

print(f"USE {device}, bf16 = {is_bf16}")
args = TrainingArguments(
    output_dir=f"GPT2_{base_conf.vocab_size}_{base_conf.n_layer}_{base_conf.n_embd}",
    per_device_train_batch_size=base_conf.batch_size,
    per_device_eval_batch_size=base_conf.batch_size,
    eval_strategy="steps",
    eval_steps=base_conf.eval_freq,
    logging_strategy="steps",
    logging_steps=10,
    gradient_accumulation_steps=base_conf.gradient_accumulation_steps,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_ratio=0.1,
    max_steps=base_conf.max_steps,
    lr_scheduler_type="cosine",
    learning_rate=base_conf.lr,
    save_steps=base_conf.save_freq,
    bf16=is_bf16,
    save_only_model=True,
    fp16=not is_bf16,
    push_to_hub=False,
    report_to='comet_ml'
)

print(args)

tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
trainer = Trainer(
    model=model,
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokenizer),
    args=args,
    data_collator=data_collator,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    callbacks=[GenerationLoggerCallback(tokenizer, log_steps=base_conf.eval_freq)]
)

trainer.train()