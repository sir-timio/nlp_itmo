import numpy as np
import pandas as pd
import pymorphy2

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import random
import os

from catalyst import dl, utils

from sklearn.model_selection import train_test_split


tokenizer = AutoTokenizer.from_pretrained("./rubert-base-cased/")

class TaskDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.train_a = df['message_a']
        self.train_b = df['message_b']
        self.labels = df['target']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return {'message_a': self.train_a.iloc[i], 'message_b': self.train_b.iloc[i], 'targets': self.labels.iloc[i]}

def bert_collate_fn(batch):
    encoding = tokenizer(
        [item['message_a'] for item in batch], 
        [item['message_b'] for item in batch], 
        return_tensors='pt', 
        max_length=100, 
        padding=True, 
        truncation=True)
    return {'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'targets': torch.tensor([item['targets'] for item in batch])}

class CustomRunner(dl.Runner):
    def handle_batch(self, batch):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `train()`.
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['targets']
        outputs = self.model(input_ids, attention_mask=attention_mask) # Forward pass
        
        # pass network input to state `input`
        self.batch = {"targets": labels, "logits": outputs.logits}

    def predict_batch(self, batch):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        return self.model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device)).logits

class Model:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя '

        self.seed = 42
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        utils.set_global_seed(self.seed)
        utils.prepare_cudnn(True)

        ######## Train Parameters ###########
        self.batch_size = 100
        self.epochs = 5
        #####################################

    def load_model(self):
        bert = AutoModelForSequenceClassification.from_pretrained(
            "./rubert-base-cased/"
        )

        bert.float()
        return bert.cuda()

    def normalize_text_with_morph(self, x):
        x = x.lower().replace("ё", "е")
        words = ''.join([[" ", i][i in self.alphabet] for i in x]).lower().split()
        return ' '.join([self.morph.parse(w)[0].normal_form for w in words])

    def _fit_predict(self, train, test):
        # train["message_a"] = train["message_a"].apply(self.normalize_text_with_morph)
        # train["message_b"] = train["message_b"].apply(self.normalize_text_with_morph)

        # test["message_a"] = test["message_a"].apply(self.normalize_text_with_morph)
        # test["message_b"] = test["message_b"].apply(self.normalize_text_with_morph)

        test["target"] = [-1] * len(test)

        train_nums, valid_nums = train_test_split(range(len(train)), test_size=0.1, random_state=self.seed)

        train_dataset = TaskDataset(train.iloc[train_nums])
        valid_dataset = TaskDataset(train.iloc[valid_nums])
        test_dataset = TaskDataset(test, mode='test')

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=bert_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=bert_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=bert_collate_fn)

        dataloaders = {
            "train": train_dataloader,
            "valid": valid_dataloader
        }

        bert = self.load_model()
        bert.train()

        for param in bert.base_model.parameters():
            param.requires_grad = False
        bert.classifier.requires_grad = True
        optimizer = optim.AdamW([param for param in bert.parameters() if param.requires_grad == True], lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        runner = CustomRunner()

        runner.train(
            model=bert,
            criterion=criterion,
            optimizer=optimizer,
            loaders=dataloaders,
            callbacks={
                "metric_auc": dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=2),
                "criterion": dl.CriterionCallback(  # special Callback for criterion computation
                    target_key="targets",              # `input_key` specifies correct labels (or `y_true`) from `runner.input` 
                    input_key="logits",              # `output_key` specifies model predictions (`y_pred`) from `runner.output`
                    metric_key="loss",                    # `prefix` - key to use with `runner.batch_metrics`
                ),  # alias for `runner.batch_metrics[prefix] = runner.criterion(runner.output[output_key], runner.input[input_key])`
                "optimizer": dl.OptimizerCallback(
                    metric_key="loss", 
                    accumulation_steps=1,
                    grad_clip_params=None,
                )
            },
            valid_loader="valid",
            valid_metric="accuracy",
            minimize_valid_metric=False,
            num_epochs=self.epochs,
            verbose=False,
            load_best_on_end=True,
            seed=self.seed,
            logdir=None
        )

        preds = []
        bert.eval()
        with torch.no_grad():
            for logits in runner.predict_loader(loader=test_dataloader):
                preds.extend(torch.softmax(logits.cpu(), dim=-1).argmax(dim=-1).numpy())
        return pd.DataFrame(preds, columns=["target"])

    def fit_predict(self,
                    train_1, test_1,
                    train_2, test_2,
                    train_3, test_3,
                    train_4, test_4,
                    train_5, test_5):
        predicted_1 = self._fit_predict(train_1, test_1)
        predicted_2 = self._fit_predict(train_2, test_2)
        predicted_3 = self._fit_predict(train_3, test_3)
        predicted_4 = self._fit_predict(train_4, test_4)
        predicted_5 = self._fit_predict(train_5, test_5)
        return [predicted_1, predicted_2, predicted_3, predicted_4, predicted_5]
