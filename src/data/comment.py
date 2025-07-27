# To retrain the base predictor, download from https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/ and rename the suffix to .tar.gz. Extract the csv, and rename the csv to civil_comments.csv to `dataset/CivilComments/all_data_with_identities.csv`. Remove the cached csv results, and run this file directly. The model will be trained and the results will be regenerated.

from typing import List
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.multiprocessing import Pool, set_start_method
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

import torch
from torch.utils.data import Dataset
import os
from transformers.trainer_callback import TrainerControl, TrainerState
cur_file_dir=os.path.dirname(os.path.abspath(__file__))
root=cur_file_dir+"/../../dataset/CivilComments"
from tqdm import tqdm
import pandas as pd
import numpy as np
import mlflow
from data.base import dictKFoldSplit, BaseDataset

from transformers import AutoConfig, AutoTokenizer, TrainerCallback
from evaluate import load as load_metric

from transformers import AutoConfig, AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
import os
from datasets import Dataset
from plotly.subplots import make_subplots
from transformers import TextClassificationPipeline


import plotly.graph_objects as go
class PlotLossCurveCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        loss,acc=plot_curves(state.log_history)
        loss.write_image(os.path.join(args.output_dir, 'loss_curve.png'))
        acc.write_image(os.path.join(args.output_dir, 'acc_curve.png'))
        


def plot_curves(log_history):
    train_loss = [log['loss'] for log in log_history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    train_acc= [log['accuracy'] for log in log_history if 'accuracy' in log]
    eval_acc= [log['eval_accuracy'] for log in log_history if 'eval_accuracy' in log]

    if len(train_loss) == len(eval_loss) - 1:
        train_loss = [np.nan] + train_loss
        train_acc = [np.nan] + train_acc
    
    
    loss = make_subplots()
    loss.add_trace(go.Scatter(x=list(range(len(train_loss))), y=train_loss, mode='lines', name='Training loss'))
    loss.add_trace(go.Scatter(x=list(range(len(eval_loss))), y=eval_loss, mode='lines', name='Validation loss'))
    loss.update_layout(xaxis_title='Steps', yaxis_title='Loss', title='Loss Curve')
    
    acc=make_subplots()
    acc.add_trace(go.Scatter(x=list(range(len(train_acc))), y=train_acc, mode='lines', name='Training accuracy'))
    acc.add_trace(go.Scatter(x=list(range(len(eval_acc))), y=eval_acc, mode='lines', name='Validation accuracy'))
    acc.update_layout(xaxis_title='Steps', yaxis_title='Accuracy', title='Accuracy Curve')
    
    return loss, acc


# Try to use spawn method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def train_on_specific_lr(args):
    sequences, labels, result_dir, lr, gpu_id = args

    # Set the specific GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    config = AutoConfig.from_pretrained(
        'distilbert-base-uncased',
        num_labels=1,
        finetuning_task="text-classification"
    )
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # convert to one-hot
    labels = np.eye(2)[labels.astype(int)]
    datasets = Dataset.from_dict({
        "text": sequences,
        "labels": labels
    })
    tokenized_datasets = datasets.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True,
                                      batch_size=2048)
    train_dataset, val_dataset = tokenized_datasets.train_test_split(
        test_size=0.1, seed=0).values()

    with mlflow.start_run() as mlrun:
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', config=config)
        result_lr = os.path.join(result_dir, f"lr_{lr}")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=result_lr,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            logging_dir=os.path.join(result_lr, 'logs'),
            learning_rate=lr,
            eval_strategy='steps',
            eval_steps=5000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_strategy='steps',
            save_steps=5000,
            eval_on_start=True,
            logging_strategy='steps',
            logging_steps=5000,
            logging_first_step=True,
        )
        accuracy_metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            labels = np.argmax(labels, axis=1)
            return accuracy_metric.compute(predictions=predictions, references=labels)

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        trainer.add_callback(PlotLossCurveCallback())
        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model(os.path.join(result_lr, 'checkpoint'))

        # Return eval results
        eval_result = trainer.evaluate()
        eval_acc = eval_result["eval_accuracy"]

        # Write the eval_acc to the file
        with open(os.path.join(result_lr, 'eval_acc.txt'), 'w') as f:
            f.write("Eval_acc: " + str(eval_acc))

        return lr, eval_acc, trainer, os.path.join(result_lr, 'checkpoint')


def train_distilBERT(sequences: List[str], labels: np.ndarray):
    result_dir = f'{root}/results'
    checkpoint_dir = os.path.join(result_dir, 'checkpoint')
    if os.path.exists(checkpoint_dir):
        model = DistilBertForSequenceClassification.from_pretrained(
            checkpoint_dir)
        print("Checkpoint found. Loading the model from checkpoint.")
        return model, np.nan

    # Make sure the result directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Prepare arguments for each learning rate and GPU
    learning_rates = [1e-6, 2e-6, 1e-5, 2e-5]
    num_gpus = min(4, torch.cuda.device_count())  # Use up to 4 GPUs

    if num_gpus < len(learning_rates):
        print(
            f"Warning: Only {num_gpus} GPUs available, but {len(learning_rates)} learning rates to test.")

    # Create arguments for parallel execution
    args_list = []
    for i, lr in enumerate(learning_rates):
        gpu_id = i % num_gpus  # Assign GPU in round-robin fashion
        args_list.append((sequences, labels, result_dir, lr, gpu_id))

    # Run training in parallel
    results = []
    with Pool(processes=min(len(learning_rates), num_gpus)) as pool:
        results = pool.map(train_on_specific_lr, args_list)

    # Find the best model
    best_eval_acc = 0.0
    best_checkpoint_path = None

    for lr, eval_acc, _, checkpoint_path in results:
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_checkpoint_path = checkpoint_path

            # Write the best eval_acc to a file
            with open(os.path.join(result_dir, 'best_eval_acc.txt'), 'w') as f:
                f.write(f"Best eval_acc: {best_eval_acc} (lr={lr})")

    # Load and return the best model
    best_model = DistilBertForSequenceClassification.from_pretrained(
        best_checkpoint_path)

    # Save the best model to the checkpoint_dir
    best_model.save_pretrained(checkpoint_dir)

    # Note: You might need to modify your plot_curves function to work with the parallel approach
    # You can still plot curves for the best model if needed

    return best_model, best_eval_acc


def inference_distilBERT(model, sequences: List[str]):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    pipeline = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, function_to_apply="softmax", device=1, batch_size=256, return_all_scores=True)
    scores = []
    index_id=1
    for val in tqdm(pipeline((s for s in sequences)),desc="Inference",position=0,total=len(sequences)):
        scores.append(val[index_id]["score"])
    return scores


class CivilCommentsDataset(BaseDataset):
    # Modified from https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/datasets/civilcomments_dataset.py
    
    def __name__(self):
        name="CivilComments"
        return name
    
    def __init__(self):
        group_cols = [
            'male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white'
        ]
        self.group_note="wilds"
        cal_val,test=self.prepare_calibration()        
        
        gm=(cal_val[group_cols]>=0.5).mean()
        self.group_names=gm[gm>0.01].index.to_list()
        
        self.group_mean=gm[self.group_names]
        def _to_dict(df):
            return {"group":df[self.group_names]>=0.5,"y":df["toxicity"].values,"pred":df["pred"].values}
        self.test_fold=_to_dict(test)
        cal_val_dict=_to_dict(cal_val)
        kf=dictKFoldSplit(cal_val_dict,n_folds=10,split_size=[5,5],test_ratio=0)
        self.folds=list(kf)      
        self.ds_size=len(self.folds[0][0]["y"])  
    
    def prepare_calibration(self):
        try:
            cal_val=pd.read_csv(f"{root}/cal_val_binary.csv",index_col=0) 
            test=pd.read_csv(f"{root}/test_binary.csv",index_col=0)
            return cal_val,test
        except FileNotFoundError:
            try:
                train_label=pd.read_csv(f"{root}/all_data_with_identities.csv",index_col=0)
            except FileNotFoundError:
                print(f"Failed to load {root}/all_data_with_identities.csv, please retrieve data from https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/")
                raise FileNotFoundError
            train_label = train_label.dropna(subset=["comment_text"])
            train_label.head().to_csv(f"{root}/example.csv")
            text="comment_text"
            train_label["toxicity"]=train_label["toxicity"]>=0.5
            split=train_label["split"] # train 60%, val 10%, test 30%
            train=train_label[split=="train"].sample(frac=5/6,random_state=0)
            
            merged=train_label[(split=="train") | (split=="val")]
            cal_val=merged.drop(train.index)
            test=train_label[split=="test"]
            
            model, eval_acc=train_distilBERT(train[text].to_list(),train["toxicity"].to_numpy())
            print(f"Eval acc: {eval_acc}")
            train_pred = inference_distilBERT(model, train[text].to_list())
            train["pred"] = train_pred
            train = train.drop(["comment_text", "created_date"], axis=1)
            train.to_csv(f"{root}/train_binary.csv")
            cal_val_pred=inference_distilBERT(model,cal_val[text].to_list())
            test_pred=inference_distilBERT(model,test[text].to_list())
            cal_val["pred"]=cal_val_pred
            test["pred"]=test_pred
            cal_val=cal_val.drop(["comment_text","created_date"],axis=1)
            test=test.drop(["comment_text","created_date"],axis=1)
            cal_val.to_csv(f"{root}/cal_val_binary.csv")
            test.to_csv(f"{root}/test_binary.csv")
            return cal_val,test


    
if __name__=="__main__":
    fd=CivilCommentsDataset()
