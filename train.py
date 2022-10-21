from dataset import AirogsDataset, collate_fn
import torch 
from transformers import ViTForImageClassification, Trainer, TrainingArguments
import numpy as np
from training_utils import CustomTrainer, compute_metrics

dataset = AirogsDataset()
print('Image files list=', dataset.image_files)

train_set_len = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_set_len, len(dataset) - train_set_len])

print('Validation set indices =', val_dataset.indices)
print('Train set indices =', train_dataset.indices)

model = ViTForImageClassification.from_pretrained(
    'facebook/deit-small-patch16-224',
    num_labels=1,
    ignore_mismatched_sizes=True
)


training_args = TrainingArguments(
    output_dir='./trained_models',
    per_device_train_batch_size=32,
    evaluation_strategy='steps',
    num_train_epochs=50,
    fp16=torch.cuda.is_available(), # Should be True on CUDA
    save_steps=150,
    eval_steps=50,
    logging_steps=50,
    learning_rate=2e-4,
    load_best_model_at_end=True,
)

trainer = CustomTrainer(
    samples_per_class=dataset.samples_per_class,
    model=model,
    args=training_args,
    data_collator=collate_fn,   
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

pre_train_metrics = trainer.evaluate()
trainer.log_metrics('eval', pre_train_metrics)

train_results = trainer.train()

# eval before run
# print the names of the val images

# run 1: eval every 30 iterations for 2 epochs
# run2: eval every 500 iterations, run for 20 epochs, get final model. Use this model to find the val images it gets so terribly wrong. 

