from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
import torch
from datasets import DatasetDict
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer, DataCollatorWithPadding
from PIL import Image
import torch
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
import pathlib 
import os 


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"  


trained_model_path = "blip_model_trained1"
train_sample_size = 10000 
val_sample_size = 2000  
checkpoint_dir="output1"
batch_size=32
num_epochs=500
Lr=1e-4
seed=1
save_total_limit=2


model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)


torch.manual_seed(seed)
dataset = load_dataset("hongrui/mimic_chest_xray_v_1")
train_size = int(len(dataset['train']) * 0.8)  
val_size = len(dataset['train']) - train_size  
train_dataset = dataset['train'].select(range(train_size))
val_dataset = dataset['train'].select(range(train_size, train_size + val_size))
processed_dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})


def preprocess_for_blip(batch):
            inputs = processor(
                images=batch["image"],
                text=batch["report"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=52  
            )
            return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "pixel_values": inputs.pixel_values,
                "labels": inputs.input_ids
            }
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,  
    model=model, 
    padding=True,
)
train_dataset = processed_dataset["train"].select(range(train_sample_size))
val_dataset = processed_dataset["validation"].select(range(val_sample_size))
train_dataset = train_dataset.map(preprocess_for_blip, batched=True)
val_dataset = val_dataset.map(preprocess_for_blip, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "labels"])


training_args = TrainingArguments(
    output_dir=checkpoint_dir, 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,  
    per_device_eval_batch_size=batch_size,  
    num_train_epochs=num_epochs,  
    save_total_limit=save_total_limit,
    logging_dir=f"{checkpoint_dir}_logs", 
    logging_steps=1,
    learning_rate=Lr,
    fp16=True,  
    report_to="wandb", 
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator)
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()


os.makedirs(trained_model_path,exist_ok=True)
model.save_pretrained(trained_model_path)
processor.save_pretrained(trained_model_path)