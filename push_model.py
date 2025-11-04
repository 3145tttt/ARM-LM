from transformers import AutoTokenizer, AutoModel

token = ''
path = "./GPT2_88_12_768/checkpoint-40000"
NEW_REPO_ID = f"3145tttt/arm_gpt_GPT2_88_12_768"

print("Create model")
model = AutoModel.from_pretrained(
    path, 
    trust_remote_code=True
)

print("Create tokenizer")
tokenizer = AutoTokenizer.from_pretrained(path)

model.push_to_hub(
    repo_id=NEW_REPO_ID,
    commit_message="Upload trained model weights",
    safe_serialization=True,
    token=token
)

tokenizer.push_to_hub(
    repo_id=NEW_REPO_ID,
    commit_message="Upload tokenizer",
    safe_serialization=True,
    token=token
)