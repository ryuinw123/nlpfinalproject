#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"


# In[2]:


# import collections
# import logging
# import regex
# import string
# import unicodedata
# from functools import partial
# from multiprocessing import Pool as ProcessPool
# from typing import Tuple, List, Dict
# import numpy as np

# from collections import Counter

# """
# Evaluation code from DPR: https://github.com/facebookresearch/DPR
# """

# class SimpleTokenizer(object):
#     ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
#     NON_WS = r'[^\p{Z}\p{C}]'

#     def __init__(self):
#         """
#         Args:
#             annotators: None or empty set (only tokenizes).
#         """
#         self._regexp = regex.compile(
#             '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
#             flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
#         )

#     def tokenize(self, text, uncased=False):
#         matches = [m for m in self._regexp.finditer(text)]
#         if uncased:
#             tokens = [m.group().lower() for m in matches]
#         else:
#             tokens = [m.group() for m in matches]
#         return tokens

# logger = logging.getLogger(__name__)

# QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])

# def calculate_matches(data: List, workers_num: int):
#     """
#     Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
#     documents and results. It internally forks multiple sub-processes for evaluation and then merges results
#     :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
#     :param answers: list of answers's list. One list per question
#     :param closest_docs: document ids of the top results along with their scores
#     :param workers_num: amount of parallel threads to process data
#     :param match_type: type of answer matching. Refer to has_answer code for available options
#     :return: matching information tuple.
#     top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
#     valid matches across an entire dataset.
#     questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
#     """

#     logger.info('Matching answers in top docs...')

#     tokenizer = SimpleTokenizer()
#     get_score_partial = partial(check_answer, tokenizer=tokenizer)

#     processes = ProcessPool(processes=workers_num)
#     scores = processes.map(get_score_partial, data)

#     logger.info('Per question validation results len=%d', len(scores))

#     n_docs = len(data[0]['ctxs'])
#     top_k_hits = [0] * n_docs
#     for question_hits in scores:
#         best_hit = next((i for i, x in enumerate(question_hits) if x), None)
#         if best_hit is not None:
#             top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

#     return QAMatchStats(top_k_hits, scores)

# def check_answer(example, tokenizer) -> List[bool]:
#     """Search through all the top docs to see if they have any of the answers."""
#     answers = example['answers']
#     ctxs = example['ctxs']

#     hits = []

#     for i, doc in enumerate(ctxs):
#         text = doc['text']

#         if text is None:  # cannot find the document for some reason
#             logger.warning("no doc in db")
#             hits.append(False)
#             continue

#         hits.append(has_answer(answers, text, tokenizer))

#     return hits

# def has_answer(answers, text, tokenizer) -> bool:
#     """Check if a document contains an answer string."""
#     text = _normalize(text)
#     text = tokenizer.tokenize(text, uncased=True)

#     for answer in answers:
#         answer = _normalize(answer)
#         answer = tokenizer.tokenize(answer, uncased=True)
#         for i in range(0, len(text) - len(answer) + 1):
#             if answer == text[i: i + len(answer)]:
#                 return True
#     return False

# #################################################
# ########        READER EVALUATION        ########
# #################################################

# def _normalize(text):
#     return unicodedata.normalize('NFD', text)

# #Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# def normalize_answer(s):
#     def remove_articles(text):
#         return regex.sub(r'\b(a|an|the)\b', ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))

# def exact_match_score(prediction, ground_truth):
#     return normalize_answer(prediction) == normalize_answer(ground_truth)

# def ems(prediction, ground_truths):
#     return max([exact_match_score(prediction, gt) for gt in ground_truths])

# ####################################################
# ########        RETRIEVER EVALUATION        ########
# ####################################################

# def eval_batch(scores, inversions, avg_topk, idx_topk):
#     for k, s in enumerate(scores):
#         s = s.cpu().numpy()
#         sorted_idx = np.argsort(-s)
#         score(sorted_idx, inversions, avg_topk, idx_topk)

# def count_inversions(arr):
#     inv_count = 0
#     lenarr = len(arr)
#     for i in range(lenarr):
#         for j in range(i + 1, lenarr):
#             if (arr[i] > arr[j]):
#                 inv_count += 1
#     return inv_count

# def score(x, inversions, avg_topk, idx_topk):
#     x = np.array(x)
#     inversions.append(count_inversions(x))
#     for k in avg_topk:
#         # ratio of passages in the predicted top-k that are
#         # also in the topk given by gold score
#         avg_pred_topk = (x[:k]<k).mean()
#         avg_topk[k].append(avg_pred_topk)
#     for k in idx_topk:
#         below_k = (x<k)
#         # number of passages required to obtain all passages from gold top-k
#         idx_gold_topk = len(x) - np.argmax(below_k[::-1])
#         idx_topk[k].append(idx_gold_topk)

# """
# our evaluation code
# """

# def f1_score(prediction, ground_truth):
#     normalized_prediction = normalize_answer(prediction)
#     normalized_ground_truth = normalize_answer(ground_truth)
    
#     ZERO_METRIC = (0, 0, 0, 0)

#     if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
#         return ZERO_METRIC
#     if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
#         return ZERO_METRIC

#     accuracy = 1.0 if normalized_ground_truth in normalized_prediction else 0.0

#     prediction_tokens = normalized_prediction.split()
#     ground_truth_tokens = normalized_ground_truth.split()
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return ZERO_METRIC
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return (f1, precision, recall, accuracy)

# def f1_scores(prediction, ground_truths):
#     max_f1 = max_precision = max_recall = max_acc = 0

#     for gt in ground_truths:
#         score = f1_score(prediction, gt)
#         max_f1 = max(max_f1, score[0])  # Accessing F1 score from the tuple
#         max_precision = max(max_precision, score[1])  # Accessing precision from the tuple
#         max_recall = max(max_recall, score[2])  # Accessing recall from the tuple
#         max_acc = max(max_acc, score[3]) 
    
#     return max_f1, max_precision, max_recall, max_acc

# def evaluate_QA(results, ans_key, predict_key):
#     """
#     EVALUATION
#     """
#     metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'acc': 0}    
            
#     em_for_task = ems
#     f1_for_task = f1_scores
        
#     for result in results:
#         prediction = result[predict_key]
#         gold = result[ans_key]
    
#         em = em_for_task(prediction, gold)
#         f1, prec, recall, acc = f1_for_task(prediction, gold)
        
#         metrics['em'] += float(em)
#         metrics['f1'] += f1
#         metrics['prec'] += prec
#         metrics['recall'] += recall
#         metrics['acc'] += acc
                
#         result['metrics'] = {'em': float(em), 'f1': f1, 'prec': prec, 'recall': recall, 'acc': acc}

#     for k in metrics.keys():
#         metrics[k] /= len(results)
    
#     return metrics

 # metrics = evaluate_QA(results, "gold", "prediction")


# In[30]:


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
from torch.optim import Adam
from torch.distributions import Categorical
import datasets
from transformers import AutoTokenizer , AutoModel , pipeline
from tqdm import tqdm
import random
from qa_metrics.em import em_match
import wandb
from safetensors.torch import load_model, save_model


# In[4]:


# torch._functorch.config.donated_buffer=False


# In[5]:


torch.set_printoptions(threshold=10_000)
# torch.autograd.set_detect_anomaly(True)


# In[6]:


import ijson
from datasets import Dataset , concatenate_datasets , DatasetDict
from decimal import Decimal


def safe_convert(value):
    if isinstance(value, Decimal):
        return float(value)  # or str(value) if you want string format
    elif isinstance(value, dict):
        return {k: safe_convert(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [safe_convert(v) for v in value]
    else:
        return value

def stream_json_array(file_path):
    with open(file_path, 'rb') as f:
        for item in ijson.items(f, 'item'):
            yield safe_convert(item)

tqa_dataset = Dataset.from_generator(lambda: stream_json_array("../dataset/document_tqa_train.json"))


# In[7]:


from itertools import islice

def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch


# In[8]:


limit_data_per_sample = 2
tqa_dataset = tqa_dataset.filter(
    lambda example: (
        sum(ctx["has_answer"] for ctx in example["ctxs"][:5]) >= limit_data_per_sample
        # and sum(not ctx["has_answer"] for ctx in example["ctxs"][:5]) >= limit_data_per_sample
    )
)


# In[9]:


limit_context = 5


# In[10]:


filter_device = "cuda:0"
gen_device = "cuda:1"


# In[11]:


filter_model = "answerdotai/ModernBERT-base"
gen_model = "google/gemma-2-2b-it"


# In[12]:


filter_tokenizer = AutoTokenizer.from_pretrained(filter_model)
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model)


# In[13]:


generator_batch_size = 16
target_model = pipeline(
    "text-generation",
    model=gen_model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=gen_device,  # replace with "mps" to run on a Mac device
    batch_size=generator_batch_size
)


# In[14]:


def generate_batch_predictions(batch, batch_size):
    predictions = []
    for i in tqdm(range(0, len(batch), batch_size), desc="🧠 Generating answers"):
        sub_batch = batch[i:i+batch_size]
        chat_batch = [
            [{"role": "user", "content": f"""Concise answer strictly based on the provided context only. Do not generate responses beyond the given information. Do not think or infer. Respond concisely using only the context. \nContext: {item['context']}\nQuestion: {item['question']}\nAnswer:"""}]
            for item in sub_batch
        ]

        outputs = target_model(
            chat_batch,
            return_full_text=False,
            max_new_tokens=50
        )

        predictions.extend([
            output[0]["generated_text"].removeprefix("assistant").strip().lower()
            for output in outputs
        ])

    return predictions


# In[15]:


def tqa_llm_filter(example):
    questions = example["question"]
    batch_ctxs = example["ctxs"]
    batch_answers = example["answers"]
    llm_batch_context = []
    llm_batch_zeroshot = []

    return_sample = []
    
    for question , ctxs in zip(questions , batch_ctxs):
        context = "\n".join(ctx["text"] for ctx in ctxs[:limit_context])
        llm_batch_context.append({
            "context" : context,
            "question" : question
        })
        llm_batch_zeroshot.append({"context" : "" , "question" : question})
    
    preds_context = generate_batch_predictions(llm_batch_context , generator_batch_size)
    preds_zeroshot = generate_batch_predictions(llm_batch_zeroshot , generator_batch_size)

    for i in range(len(preds_context)):
        match_context = em_match(batch_answers[i], preds_context[i])
        match_zeroshot = em_match(batch_answers[i], preds_zeroshot[i])

        if (not match_context or match_zeroshot):
            return_sample.append(False)
            continue
        return_sample.append(True)

    return return_sample
    

    
    


# In[16]:


cache_path = os.path.join(os.getcwd(), "cache")
os.makedirs(cache_path, exist_ok=True)
tqa_dataset = tqa_dataset.filter(tqa_llm_filter , batched = True , batch_size=1024 , cache_file_name=os.path.join(cache_path, "tqa.arrow"))


# In[17]:


# tqa_dataset.save_to_disk('tqa-dataset')


# In[18]:


max_length = 1024


# In[19]:


def tqa_preprocess(example):
    question = example["question"]
    ctxs = example["ctxs"]
    context = "\n".join(ctx["text"] for ctx in ctxs[:limit_context])

    input_ids = filter_tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_tensors = "pt",
        return_special_tokens_mask = True
    )
    mask = input_ids["special_tokens_mask"][0]  # shape: (max_length,)

    # Find indices of 1s
    one_indices = (mask == 1).nonzero(as_tuple=True)[0]
    
    # Make sure there are at least two 1s
    if len(one_indices) >= 2:
        start, end = one_indices[0], one_indices[1]
        mask[start:end + 1] = 1  # Set all values between (and including) first two 1s to 1
    mask = (mask == 0).int()
    
    # Optionally: update back to batch tensor
    input_ids["special_tokens_mask"][0] = mask
    
    return {
        "input_ids": input_ids["input_ids"][0],
        "attention_mask" : input_ids["attention_mask"][0],
        "tokens_mask" : input_ids["special_tokens_mask"][0],
        "labels": example["answers"],
        "question" : question
    }


# In[20]:


tqa_dataset = tqa_dataset.map(tqa_preprocess, remove_columns=tqa_dataset.column_names)


# In[21]:


train_val_test = tqa_dataset.train_test_split(test_size=0.15, seed=42)
train_dataset = train_val_test['train']
temp_dataset = train_val_test['test']

# Step 2: Split the temp dataset into validation and test (50% each of the 20%)
val_test = temp_dataset.train_test_split(test_size=0.75, seed=42)
validation_dataset = val_test['train']
test_dataset = val_test['test']

# Step 3: Wrap into DatasetDict
tqa_dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset,
})


# In[22]:


tqa_dataset


# In[23]:


class BasePolicy(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.layer1 = nn.Linear(base_model.config.hidden_size, 4096)


# In[24]:


class ActorNet(BasePolicy):
    def __init__(self, base_model, act_dim):
        super().__init__(base_model)
        self.layer2 = nn.Linear(4096, 4096)
        self.layer3 = nn.Linear(4096, act_dim)

    def forward(self, input_ids, attention_mask):
        features = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = F.relu(self.layer1(features))
        x = F.relu(self.layer2(x))
        logits = self.layer3(x)  # (batch, seq_len, act_dim)
        return logits


# In[25]:


class CriticNet(BasePolicy):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.layer3 = nn.Linear(4096, 1)

    def forward(self, input_ids, attention_mask):
        features = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(features)
        masked = features * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        x = F.relu(self.layer1(pooled))
        value = self.layer3(x).squeeze(-1)  # (batch,)
        return value


# In[26]:


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)
    
class ValidatorModel(nn.Module):
    def __init__(self, checkpoint):
        super(ValidatorModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        config = self.encoder.config
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        self.loss_fn = nn.BCELoss()  # Use BCELoss since we apply Sigmoid manually

        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = GELUActivation()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = torch.nn.Dropout(config.classifier_dropout)

        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.qa_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
    
        pooling = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
            dim=1, keepdim=True
        )

        
        pooled_output = self.norm(self.act(self.dense(pooling)))
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)
        cls_logits = logits.squeeze(-1)
        
        loss = None
        if labels is not None:
            cls_loss = self.cls_loss_fn(cls_logits, labels.float())
            loss = cls_loss

        return {
            "pooler_output" : pooled_output,
            "loss": loss,
            "logits": cls_logits,
        }


# In[27]:


class PPO:
    def __init__(self, **hyperparameters):
        # """
        # 	Initializes the PPO model, including hyperparameters.
        
        # 	Parameters:
        # 		policy_class - the policy class to use for our actor/critic networks.
        # 		env - the environment to train on.
        # 		hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
        
        # 	Returns:
        # 		None
        # """
        # Make sure the environment is compatible with our code
        wandb.init(project="ppo-transformer", config=hyperparameters)
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)        
        # Extract environment information
        validator_model = ValidatorModel(filter_model)
        load_model(validator_model, "../modernbert-finetuned-triviaqa-wiki-noqa/checkpoint-67884/model.safetensors")
        self.base_model = validator_model.encoder
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.act_dim = 2
        
         # Initialize actor and critic networks
        self.actor = ActorNet(self.base_model, self.act_dim).to(filter_device)
        self.critic = CriticNet(self.base_model).to(filter_device)
        
        # Initialize optimizers for actor and critic
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr
        )

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
    
            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.
    
            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.rollout_size = 512                 # Number of timesteps to run per batch
        self.backward_batch_size = 8
        self.filter_batch_size = 32
        self.generate_batch_size = 16           # Max number of timesteps per episode
        self.n_updates_per_iteration = 4                  # Number of times to update actor/critic per iteration
        self.lr = 0.000005                                   # Learning rate of actor optimizer
        self.clip = 0.2                                   # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.ent_coef = 0.01
        self.vf_coef = 1
    
        # # Miscellaneous parameters
        # self.render = True                              # If we should render during rollout
        # self.render_every_i = 10                        # Only render every n iterations
        # self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
    
        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
    
        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)
    
            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def train(self , epochs = 10):
        best_reward = float('-inf')
        for epoch in range(epochs):
            print(f"\n====== Epoch {epoch + 1}/{epochs} ======")
            self.learn()
            reward = self.validation()
            wandb.log({"epoch": epoch + 1, "evaluation_reward": reward, "best_reward": best_reward})
    
            print(f"Evaluation reward: {reward:.4f} | Best so far: {best_reward:.4f}")
            if reward > best_reward:
                best_reward = reward
                print(f"New best reward! Saving model at epoch {epoch + 1}")
                torch.save(self.actor.state_dict(), f"actor_best.pt")
                torch.save(self.critic.state_dict(), f"critic_best.pt")
            
        
    def validation(self):
        batch_rtgs = self.evaluate_rollout()
        return sum(batch_rtgs)
        
    def learn(self):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.
        
            Parameters:
                total_timesteps - the total number of timesteps to train for
        
            Return:
                None
        """

    
        batch_input_ids, batch_attention_mask, batch_tokens_mask, batch_acts, batch_log_probs, batch_rtgs = self.rollout()
        print("Rewards-to-Go:", batch_rtgs)
        wandb.log({"mean_rewards_to_go": sum(batch_rtgs) / len(batch_rtgs)})
        

        for i in range(0, len(batch_input_ids), self.backward_batch_size):
            # print("-----------------------enter 2-------------------------------")
            input_ids = torch.stack(batch_input_ids[i:i+self.backward_batch_size]).to(filter_device)
            attention_mask = torch.stack(batch_attention_mask[i:i+self.backward_batch_size]).to(filter_device)
            tokens_mask = torch.stack(batch_tokens_mask[i:i+self.backward_batch_size]).to(filter_device)
            actions = torch.tensor(batch_acts[i:i+self.backward_batch_size]).to(filter_device)
            prev_log_probs = torch.tensor(batch_log_probs[i:i+self.backward_batch_size]).to(filter_device)
            rtgs = torch.tensor(batch_rtgs[i:i+self.backward_batch_size], dtype=torch.float32).to(filter_device)


            V, _ , _ = self.evaluate(input_ids , attention_mask , actions , tokens_mask)
            # print("rtgs = " , rtgs)
            A_k = rtgs - V.detach()
            # print("A_k before:", A_k)
            # print("V:", V)
            # print("A_k mean:" , A_k.mean())
            # print("A_k std:" , A_k.std())
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            # print("A_k after:", A_k)
            for _ in range(self.n_updates_per_iteration):
                # print("-------------------------------enter 3----------------------------------------")
                V, curr_log_probs , entropy = self.evaluate(input_ids , attention_mask , actions , tokens_mask)
                entropy_loss = entropy.mean()
                ratios = torch.exp(curr_log_probs - prev_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, rtgs)
                # print("critic loss = " , critic_loss)

                total_loss = (actor_loss - self.ent_coef * entropy_loss) + self.vf_coef * critic_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                wandb.log({"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "total_loss": total_loss.item()})
                    
    @torch.no_grad()
    def get_action(self , input_ids , attention_mask , tokens_mask):
        """
        Sample from a Categorical distribution with masking, and compute joint log-prob
        over unmasked positions only.
        
        Args:
            probs: Tensor of shape (B, T, C) – probability distributions.
            mask: Bool tensor of shape (B, T) – True means "keep", False means "ignore".
        
        Returns:
            samples: Tensor of shape (B, T)
            joint_log_probs: Tensor of shape (B,) – summed log probs over unmasked positions
        """
        logits = self.actor(input_ids , attention_mask)
        dist = Categorical(logits=logits)
        samples = dist.sample()  # (B, T)
    
        # Force dummy value (0) where masked out
        masked_samples = torch.where(tokens_mask.bool(), samples, torch.tensor(0, device=samples.device))
    
        # Compute log-prob
        log_probs = dist.log_prob(masked_samples)
    
        # Mask out positions (set log_prob = 0.0 where mask is False)
        log_probs = torch.where(tokens_mask.bool(), log_probs, torch.tensor(0.0, dtype=log_probs.dtype, device=log_probs.device))
    
        # Sum only unmasked log probs → joint log-prob
        joint_log_probs = log_probs.sum(dim=-1)
    
        return masked_samples.cpu(), joint_log_probs

    def evaluate(self, input_ids , attention_mask , actions , tokens_mask):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
        
            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
        
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(input_ids , attention_mask).squeeze()
        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        logits = self.actor(input_ids , attention_mask)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        log_probs = torch.where(tokens_mask.bool(), log_probs, torch.tensor(0.0, dtype=log_probs.dtype, device=log_probs.device))
        log_probs = log_probs.sum(dim=-1)
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs , dist.entropy()

    def reward_fn(self, input_ids, tokens_mask, actions , answers , questions):
        # action shape: (batch_size, sequence_length)
        rewards = []
        llm_batch = []
        
        for i in range(actions.size(0)):  # batch loop
            act = actions[i]
            mask = tokens_mask[i]
            ids = input_ids[i]
            # Filter only positions where tokens_mask == 1
            act = act[mask == 1]
            ids = ids[mask == 1]
            kept_ids = ids[act == 0]

            llm_batch.append({"context" : filter_tokenizer.decode(kept_ids) , "question" : questions[i]})

            total_tokens = (mask == 1).sum().item()
            kept_tokens = (act == 0).sum().item()
            rewards.append((kept_tokens, total_tokens))

        llm_response = generate_batch_predictions(llm_batch , self.generate_batch_size)
        # print("llm_response" , llm_response)
        # print("answers = " , answers)
        # print("rewards", rewards)
        for i in range(len(llm_response)):
            kept_tokens, total_tokens = rewards[i]
            match_result = em_match(answers[i], llm_response[i])
    
            if not match_result:
                rewards[i] = -0.1  # Hard penalty for wrong answer
            else:
                # Reward is based on accuracy (1.0) and compression bonus
                compression_ratio = 1 - kept_tokens / (total_tokens + 1e-10)
                rewards[i] = compression_ratio  # Adjust weight here if needed
        wandb.log({"reward_distribution": rewards})
        return rewards  # shape: (batch_size,)
        
    def evaluate_rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.
        
            Parameters:
                None
        
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
        """
        # Batch data. For more details, check function header.
        batch_input_ids = []
        batch_attention_mask = []
        batch_tokens_mask = []
        batch_acts = []
        batch_log_probs = []
        batch_rtgs = []

        for batch in tqdm(batched(tqa_dataset["validation"], self.filter_batch_size), desc="dataset batches"):
            input_ids = [torch.tensor(data["input_ids"]) for data in batch]
            attention_mask = [torch.tensor(data["attention_mask"]) for data in batch]
            tokens_mask = [torch.tensor(data["tokens_mask"]) for data in batch]
            labels = [data["labels"] for data in batch]
            questions = [data["question"] for data in batch]
            batch_input_ids.extend(input_ids)
            batch_attention_mask.extend(attention_mask)
            batch_tokens_mask.extend(tokens_mask)
            actions, log_probs = self.get_action(torch.stack(input_ids).to(filter_device) , torch.stack(attention_mask).to(filter_device) , torch.stack(tokens_mask).to(filter_device))
            # print(actions)
            rews = self.reward_fn(input_ids , tokens_mask , actions , labels , questions)
            batch_rtgs.extend(rews)
            batch_acts.extend(actions.tolist())
            batch_log_probs.extend(log_probs.tolist())
        
        return batch_rtgs
    
    def rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.
        
            Parameters:
                None
        
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
        """
        # Batch data. For more details, check function header.
        batch_input_ids = []
        batch_attention_mask = []
        batch_tokens_mask = []
        batch_acts = []
        batch_log_probs = []
        batch_rtgs = []

        tqa_dataset["train"] = tqa_dataset["train"].shuffle(seed = 42)

        for batch in tqdm(batched(tqa_dataset["train"], self.filter_batch_size), desc="dataset batches"):
            input_ids = [torch.tensor(data["input_ids"]) for data in batch]
            attention_mask = [torch.tensor(data["attention_mask"]) for data in batch]
            tokens_mask = [torch.tensor(data["tokens_mask"]) for data in batch]
            labels = [data["labels"] for data in batch]
            questions = [data["question"] for data in batch]
            batch_input_ids.extend(input_ids)
            batch_attention_mask.extend(attention_mask)
            batch_tokens_mask.extend(tokens_mask)
            actions, log_probs = self.get_action(torch.stack(input_ids).to(filter_device) , torch.stack(attention_mask).to(filter_device) , torch.stack(tokens_mask).to(filter_device))
            # print(actions)
            rews = self.reward_fn(input_ids , tokens_mask , actions , labels , questions)
            batch_rtgs.extend(rews)
            batch_acts.extend(actions.tolist())
            batch_log_probs.extend(log_probs.tolist())
            if len(batch_input_ids) > self.rollout_size:
                break
        
        return batch_input_ids , batch_attention_mask , batch_tokens_mask , batch_acts, batch_log_probs, batch_rtgs


# In[28]:


print(f"Training", flush=True)


# In[31]:


model = PPO()


# In[ ]:


model.train(epochs=500)


# In[ ]:





# In[ ]:




