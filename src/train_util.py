import comet_ml

import torch
from transformers import TrainerCallback, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling
from datasets import DatasetDict

from src.eval_promts import EVAL_PROMTS, clean_text
from src.data import CorpusDataset, create_train_val_split

class GenerationLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, log_steps=500, num_samples=len(EVAL_PROMTS)):
        self.tokenizer = tokenizer
        self.log_steps = log_steps
        self.num_samples = num_samples
        self.sample_inputs = EVAL_PROMTS
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log generations every log_steps
        if state.global_step % self.log_steps == 0:
            model = kwargs['model']
            
            generations = []
            with torch.inference_mode():
                for prompt in self.sample_inputs[:self.num_samples]:
                    prompt = clean_text(prompt)
                    
                    token_ids = self.tokenizer.encode(prompt).ids
                    input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
                    
                    # Generate text
                    output = model.generate(
                        input_ids,
                        max_length=380,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    generated_text = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
                    generations.append(f"Input: {prompt}\nOutput: {generated_text}\n")
            
            # Log to console
            print(f"\n=== Step {state.global_step} - Generated Texts ===")
            for gen in generations:
                print(gen)
                break
            
            # Log
            # if logs is not None:
            exp = comet_ml.get_running_experiment()
            exp.log_text("\n".join(generations), step=state.global_step)
                # logs["generations"] = "\n".join(generations)

def get_model(tokenizer, base_conf):
    config = GPT2Config(
        vocab_smodelize=tokenizer.get_vocab_size(),
        n_ctx=base_conf.context_length,
        n_embd=base_conf.n_embd,
        n_layer=base_conf.n_layer,
        bos_token_id=tokenizer.token_to_id('[PAD]'),
        eos_token_id=tokenizer.token_to_id('[PAD]'),
    )

    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    return model


def create_datasets_collator(tokenizer, data_path, tokenizer_path, context_length, split_ratio=0.9999):
    train_path, val_path = create_train_val_split(data_path, split_ratio=split_ratio)
    train_dataset = CorpusDataset(tokenizer_path, train_path, context_length)
    val_dataset = CorpusDataset(tokenizer_path, val_path, context_length)
    datasets = DatasetDict(
        {
            "train": train_dataset,
            "valid": val_dataset, 
        }
    )

    tokenizer.pad_token_id = tokenizer.token_to_id('[PAD]')
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return datasets, data_collator