from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
def hf_train_eval(model_name, train_df, val_df, le, epochs=3, batch_size=16):
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    tok = AutoTokenizer.from_pretrained(model_name)
    def enc(b): return tok(b['text'], padding='max_length', truncation=True)
    tr = Dataset.from_pandas(train_df).map(enc, batched=True)
    va = Dataset.from_pandas(val_df).map(enc, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
    args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch",
                             per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                             num_train_epochs=epochs, logging_dir="./logs", logging_steps=10, report_to=[])
    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy":accuracy_score(p.label_ids,preds), "macro_f1":f1_score(p.label_ids,preds,average="macro")}
    trn = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, tokenizer=tok, compute_metrics=metrics)
    trn.train(); return model
