import json

from torch.utils.data import Dataset
from tqdm import tqdm

from prompts.pre_process import load_get_corpus_fn
from prompts.prompts import load_get_prompt_fn


class Seq2SeqDataset(Dataset):

    def __init__(self, input_path, task, llm_tokenizer, max_length, begin_idx,
                 end_idx) -> None:
        super().__init__()
        with open(input_path, 'r') as file:
            self.data = json.load(file)[begin_idx:end_idx]
        self.task = task
        self.get_prompt_fn = load_get_prompt_fn(task)

        self.llm_tokenizer = llm_tokenizer
        self.max_length = max_length

    def get_prompt(self, inp, profile):
        return self.get_prompt_fn(inp, profile, self.max_length,
                                  self.llm_tokenizer)

    def __getitem__(self, index):
        user_id = self.data[index]['user_id']
        all_profile = self.data[index]['retrieval']
        return {
            "user_id": user_id,
            "input": self.get_prompt(self.data[index]['input'], all_profile),
            "output": self.data[index]['output']
        }

    def __len__(self):
        return len(self.data)


class PointLabelDataset(Dataset):

    def __init__(self, input_path, task, use_date, llm_tokenizer, max_length,
                 begin_idx, end_idx) -> None:
        super().__init__()
        with open(input_path, 'r') as file:
            self.data = json.load(file)
        print("datasize: {}".format(len(self.data)))
        self.data = self.data[begin_idx:end_idx]
        self.task = task
        self.llm_tokenizer = llm_tokenizer
        self.max_length = max_length

        self.get_corpus = load_get_corpus_fn(task)
        self.get_prompt_fn = load_get_prompt_fn(task)

        self.datasets = []
        self.lengths = []
        self.user_list = []
        for data in tqdm(self.data):
            self.lengths.append(len(data['retrieval']))
            self.user_list.append(data['user_id'])
            for profile in data['retrieval']:
                self.datasets.append({
                    "user_id":
                    profile['user_id'],
                    "user_sim":
                    profile['user_sim'],
                    "query":
                    data['query'],
                    "doc":
                    self.get_corpus([profile], use_date)[0],
                    "input":
                    self.get_prompt(data['input'], [profile]),
                    "output":
                    data['output']
                })

    def get_prompt(self, inp, profile):
        return self.get_prompt_fn(inp, profile, self.max_length,
                                  self.llm_tokenizer)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
