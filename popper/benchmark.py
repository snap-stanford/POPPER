import json
import os
import random
import traceback
import pandas as pd
import numpy as np

from popper.utils import DiscoveryBenchDataLoader

class gene_perturb_hypothesis:
    def __init__(self, dataset='IL2',
                num_of_samples = 50, 
                permuted = False,
                user_study_neg_genes = False,
                path = None):

        path = os.path.join(path, 'benchmark/targetval/')
        if dataset == 'IL2':
            self.prompt = "Gene {gene} regulates the production of Interleukin-2 (IL-2)."
            ground_truth_path = path + 'ground_truth_IL2.csv'
        elif dataset == 'IFNG':
            self.prompt = "Gene {gene} regulates the production of Interferon-gamma (IFN-g)."
            ground_truth_path = path + 'ground_truth_IFNG.csv'

        self.ground_truth = pd.read_csv(ground_truth_path, index_col=0)

        self.query = []
        self.ground_truth['abs_score'] = self.ground_truth.Score.abs()
        self.ground_truth = self.ground_truth.sort_values('abs_score')
        self.hypothesis2score = self.ground_truth.abs_score.to_dict()

        if not permuted:
            self.query += self.ground_truth.iloc[-num_of_samples:].index.values.tolist()
            self.answer = np.array([True] * num_of_samples)
        else:
            self.query += self.ground_truth.sample(frac = 1, random_state = 42).iloc[:num_of_samples].index.values.tolist()
            self.answer = np.array([False] * num_of_samples)

        if user_study_neg_genes:
            self.query = ['CD28', 'CD2', 'CD3D', 'MAK16',  'RAC2', 'CD3E', 'VAV1', 'CD247', 'ZAP70', 'CD3G', 'LCP2']
            self.answer = np.concatenate([self.answer, np.array([False] * 10)])


    def get_example(self, index = None):
        if index is None:
            index = np.random.randint(len(self.query))
        
        q = self.query[index]
        a = self.hypothesis2score[q]
        return {"prompt": self.prompt.format(gene = q), 
                "gene": q,
                "answer": a,
                "binary_answer": self.answer[index]
                }

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def output_class(self):
        from langchain_core.pydantic_v1 import BaseModel, Field
        from typing import Optional
        class Output(BaseModel):
            """Whether or not this hypothesis is considered true/false."""

            hypothesis_test_result: Optional[bool] = Field(
                description="Whether or not this hypothesis is considered true/false."
            )
        return Output

    def evaluate(self, response, answers=None):
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score
        predicted = np.array([i[1] for i in response])
        if not answers:
            answers = np.array([exp["binary_answer"] for exp in self.examples])
        
        res_stats = np.array([i[0] for i in response])
        return {
            'accuracy': accuracy_score(answers, predicted),
            'power': np.sum((predicted == True) & (answers == True)) / np.sum((answers == True)),
            'false discovery rate': np.sum((predicted == True) & (answers == False)) / np.sum((answers == False)),
            'f1': f1_score(answers, predicted)
        }

class discovery_bench_hypothesis:
    def __init__(self, split="test", synthetic=False, num_samples=50, seed=1234, path = None):
        
        print("----------Loading Discovery Bench------------")
        
        if split != "test":
            raise NotImplementedError
        
        random.seed(seed)
        if path is None:
            raise ValueError

        root_path = path
        self.split = split
        self.synthetic = synthetic
        self.data_path = os.path.join(root_path, "discoverybench", "synthetic" if self.synthetic else "real", self.split)

        ground_truth_path = os.path.join(root_path, "answer_key/answer_key_synth.csv") if self.synthetic else os.path.join(root_path, "answer_key/answer_key_real_cleaned_1.csv")
        self.ground_truth = pd.read_csv(ground_truth_path)
        
        self.examples = []
        for task_dir in os.listdir(self.data_path):
            task_path = os.path.join(self.data_path, task_dir)
            
            for file in os.listdir(task_path):
                if file.endswith(".json"):
                    file_path = os.path.join(task_path, file)
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        metadata = json.load(f)
                    metadata_id = int(file.split(".")[0].split("_")[1])
                    data_loader = DiscoveryBenchDataLoader(task_path, metadata)
                    
                    # permute the data loader for negative example
                    permuted_dataloader = DiscoveryBenchDataLoader(task_path, metadata)
                    permuted_dataloader.permute_selected_columns()
                    
                    
                    for query_list in metadata["queries"]:
                        for query in query_list:
                            try:
                                # print(file_path)
                                hypothesis = self.ground_truth.loc[
                                    (self.ground_truth['dataset'] == task_dir) &
                                    (self.ground_truth['metadataid'] == metadata_id) &
                                    (self.ground_truth['query_id'] == query["qid"]),
                                    'gold_hypo'
                                ].iloc[0]
                                
                                if 'non-trivially falsifiable' in self.ground_truth.columns and self.ground_truth.loc[
                                    (self.ground_truth['dataset'] == task_dir) &
                                    (self.ground_truth['metadataid'] == metadata_id) &
                                    (self.ground_truth['query_id'] == query["qid"]),
                                    'non-trivially falsifiable'
                                ].iloc[0] == 0:
                                    continue
                                
                                self.examples.append({
                                    "task": task_dir,
                                    "domain": metadata["domain"],
                                    "metadataid": metadata_id,
                                    "query_id": query["qid"],
                                    "prompt": hypothesis,
                                    # "metadata": metadata,
                                    "data_loader": data_loader,
                                    "answer": True,
                                })

                                self.examples.append({
                                    "task": task_dir,
                                    "domain": metadata["domain"],
                                    "metadataid": metadata_id,
                                    "query_id": query["qid"],
                                    "prompt": hypothesis,
                                    # "metadata": metadata,
                                    "data_loader": permuted_dataloader,
                                    "answer": False,
                                })
                            except Exception as e:
                                # print(e)
                                # print(traceback.format_exc())
                                pass
                    
                    
        if num_samples < len(self.examples):
            random.shuffle(self.examples)
            self.examples = self.examples[:num_samples]
        
        self.num_samples = len(self.examples)
        print(f"Loaded {self.num_samples} hypotheses")
        print("--------------------------------------")
    
    def get_example(self, index = None):
        return self.examples[index]
    
    def get_iterator(self):
        for i in range(self.num_samples):
            yield self.get_example(i)
    
    def output_class(self):
        from langchain_core.pydantic_v1 import BaseModel, Field
        from typing import Optional
        class Output(BaseModel):
            """Whether or not this hypothesis is considered true/false."""

            hypothesis_test_result: Optional[bool] = Field(
                description="Whether or not this hypothesis is considered true/false."
            )
        return Output

    def evaluate(self, response, answers=None):
        ## expected [(res_stat, conclusion)] following the order of the query
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score
        predicted = np.array([i[1] for i in response])
        if not answers:
            answers = np.array([exp["answer"] for exp in self.examples])
        else:
            answers = np.array(answers)
        
        res_stats = np.array([i[0] for i in response])
        return {
            'accuracy': accuracy_score(answers, predicted),
            'power': np.sum((predicted == True) & (answers == True)) / np.sum((answers == True)),
            'false discovery rate': np.sum((predicted == True) & (answers == False)) / np.sum((answers == False)),
            'f1': f1_score(answers, predicted)
            
            # 'auprc': average_precision_score(answers, res_stats),
            # 'stat_pos': res_stats[np.where(answers & res_stats)[0]].mean(),
            # 'stat_neg': res_stats[np.where(answers == False & res_stats)[0]].mean()
        }
