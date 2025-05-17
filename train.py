from datasets import load_dataset
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

# === CẤU HÌNH MODEL ===
model_name = "Qwen/Qwen1.5-1.8B-Chat"
output_path = "./data.jsonl"
adapter_path = "./qwen-qlora-model"
token ="" # viết thêm token vô đây

# Thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
# train gg colab thì khỏi phải cái này, mà check xem gpu trông bao nhiêu bằng nvidia-smi

# Quantization 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
#cái này dùng cũng được, hợp với QLora 
#nhưng mà kiểm tra xem có bfloat16 không, nếu có thì đổi cái bnb_4bit_compute_dtype đi thì ooke hơn

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=token
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    token="token

model.gradient_checkpointing_enable() #giảm ram cũng được
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_proj", "w2", "o_proj"], # kiểm tra xem mô hình có những module này không nha
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# === Dữ LIỆU ===
dataset = load_dataset("json", data_files=output_path, split="train")
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

def format_chat(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}\n"
    return prompt

def tokenize(example):
    prompt = format_chat(example["messages"])
    result = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    result["labels"] = result["input_ids"].copy()
    return result

train_tokenized = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
eval_tokenized = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)

# === METRIC ===
def compute_metrics(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions[0], axis=-1)
    labels = eval_pred.label_ids
    acc = (preds == labels).astype(np.float32)
    mask = labels != -100
    acc = acc[mask]
    return {"accuracy": acc.mean()}

# === HUẤN LUYỆN ===
training_args = TrainingArguments(
    output_dir="./checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=3e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\ud83d\ude80 Bắt đầu huấn luyện...")
trainer.train()
trainer.save_model(adapter_path)

# === VẺ BIỂU ĐỒ Accuracy ===
accuracy_log = []
for log in trainer.state.log_history:
    if "eval_accuracy" in log:
        accuracy_log.append((log["step"], log["eval_accuracy"]))

if accuracy_log:
    steps, accs = zip(*accuracy_log)
    plt.plot(steps, accs, marker='o')
    plt.title("Accuracy During Training")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    print("\ud83d\udcc8 Đã lưu biểu đồ: accuracy_plot.png")
else:
    print("\u26a0\ufe0f Không có log accuracy để vẽ.")
