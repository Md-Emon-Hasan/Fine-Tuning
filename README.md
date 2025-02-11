# Fine Tuning

## 📌 Overview
This repository contains implementations and experiments related to **Fine Tuning**, including fine-tuning transformer models, applying LoRA and QLoRA, and optimizing models for efficient training.

## 🚀 Features
- Fine-tuning transformer models using **Hugging Face Transformers**
- Parameter Efficient Fine-Tuning (**PEFT**) using LoRA and QLoRA
- Model quantization with **BitsAndBytes** for low-memory optimization
- Training and evaluation with **Hugging Face Trainer API**
- Saving and loading fine-tuned models for inference

## Example: .....

## 📊 Dataset
Uses the **IMDB** dataset for sentiment classification. The dataset is loaded using:
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```
Modify `dataset` in the scripts to use custom datasets.

## 🏗 Model Fine-Tuning
We use **BERT-base-uncased** as the base model:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

## ⚡ Applying LoRA for PEFT
We apply **LoRA** for efficient fine-tuning:
```python
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, task_type="SEQ_CLS")
model = get_peft_model(model, lora_config)
```

## 🔧 Model Quantization
To optimize for low-memory environments, we apply **4-bit quantization**:
```python
from transformers import BitsAndBytesConfig
import torch
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", quantization_config=quantization_config, num_labels=2)
```

## 🏋️ Training the Model
We use the Hugging Face Trainer API to fine-tune the model:
```python
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", learning_rate=2e-5, per_device_train_batch_size=8, num_train_epochs=1)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["test"], tokenizer=tokenizer)
trainer.train()
```

## 📈 Evaluation
```python
results = trainer.evaluate()
print(results)
```

## 💾 Saving and Loading the Model
```python
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
```
To use the fine-tuned model for inference:
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="./fine-tuned-model", tokenizer=tokenizer)
result = classifier("I feel so happy today!")
print(result)
```

## 🔽 Exporting and Downloading the Model
To zip and download the trained model in **Google Colab**:
```python
import shutil
from google.colab import files
shutil.make_archive("fine-tuned-model", 'zip', "./fine-tuned-model")
files.download("fine-tuned-model.zip")
```

## 📜 License
This project is released under the MIT License.

## 📝 Author
Developed by **Md Emon Hasan**
- GitHub: [Md-Emon-Hasan](https://github.com/Md-Emon-Hasan)
- LinkedIn: [Md Emon Hasan](https://www.linkedin.com/in/md-emon-hasan)

## ⭐ Acknowledgments
Special thanks to **Hugging Face** for providing open-source tools and models for AI research!
