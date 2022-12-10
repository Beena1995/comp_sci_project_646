cfg = {
    "train_size": -1,
    "val_size": -1,
    "task_name": "toxic_multilabel",
    "no_cuda": False,
    "bert_model": 'distilbert-base-uncased',
    "output_dir": f'{location}/parquet_data/output',
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 3e-5,
    "num_train_epochs": 3.0,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}