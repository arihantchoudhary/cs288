"""Run Part 1 notebook as a script to generate .npy files."""

import os
import subprocess

# Download eval files first
os.chdir(os.path.dirname(os.path.abspath(__file__)))

for url in [
    "https://cal-cs288.github.io/sp21/project_files/proj_1/eval_prefixes.txt",
    "https://cal-cs288.github.io/sp21/project_files/proj_1/eval_output_vocab.txt",
    "https://cal-cs288.github.io/sp21/project_files/proj_1/eval_prefixes_short.txt",
    "https://cal-cs288.github.io/sp21/project_files/proj_1/eval_output_vocab_short.txt",
]:
    fname = url.split("/")[-1]
    if not os.path.exists(fname):
        subprocess.run(["curl", "-sO", url])
        print(f"Downloaded {fname}")

# imports
from collections import defaultdict, Counter
import numpy as np
import math
import tqdm
import random

import torch
from torch import nn
import torch.nn.functional as F
from datasets import Dataset

# Load WikiText-2 dataset from local arrow files
cache_path = "./wikitext/wikitext-2-v1"

train_dataset = Dataset.from_file(os.path.join(cache_path, "wikitext-train.arrow"))
validation_dataset = Dataset.from_file(os.path.join(cache_path, "wikitext-validation.arrow"))
test_dataset = Dataset.from_file(os.path.join(cache_path, "wikitext-test.arrow"))

def get_tokens(example):
    tokens = example['text'].split()
    return tokens

train_text = []
for example in train_dataset:
    tokens = get_tokens(example)
    if tokens:
        train_text.extend(tokens)

validation_text = []
for example in validation_dataset:
    tokens = get_tokens(example)
    if tokens:
        validation_text.extend(tokens)

test_text = []
for example in test_dataset:
    tokens = get_tokens(example)
    if tokens:
        test_text.extend(tokens)

token_counts = Counter(train_text)
special_tokens = ['<unk>', '<eos>', '<pad>']
for token in special_tokens:
    token_counts[token] = 0
vocab_list = sorted(token_counts.keys())
vocab_size = len(vocab_list)

class Vocab:
    def __init__(self, vocab_list, token_counts):
        self.itos = vocab_list
        self.stoi = {word: idx for idx, word in enumerate(vocab_list)}
        self.freqs = token_counts
    def __len__(self):
        return len(self.itos)

vocab = Vocab(vocab_list, token_counts)
print(f"Vocabulary size: {vocab_size}")

# ---- NGram Model ----
class NGramModel:
    def __init__(self, train_text, n=2, alpha=3e-3):
        self.n = n
        self.smoothing = alpha
        self.vocab_size = vocab_size
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        for i in range(len(train_text)):
            if i >= n - 1:
                ngram = tuple(train_text[i - n + 1 : i + 1])
                self.ngram_counts[ngram] += 1
                if n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1
        if n == 1:
            self.total_count = len(train_text)

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n
        n_gram_tuple = tuple(n_gram)
        if self.n == 1:
            count = self.ngram_counts[n_gram_tuple]
            return (count + self.smoothing) / (self.total_count + self.vocab_size * self.smoothing)
        context = n_gram_tuple[:-1]
        count = self.ngram_counts[n_gram_tuple]
        context_count = self.context_counts[context]
        return (count + self.smoothing) / (context_count + self.vocab_size * self.smoothing)

    def next_word_probabilities(self, text_prefix):
        if len(text_prefix) < self.n - 1:
            return [1.0 / self.vocab_size] * self.vocab_size
        context = text_prefix[-(self.n - 1) :] if self.n > 1 else []
        return [self.n_gram_probability(context + [word]) for word in vocab.itos]

    def perplexity(self, full_text):
        log_probabilities = []
        for i in range(len(full_text)):
            if i < self.n - 1:
                prob = 1.0 / self.vocab_size
            else:
                n_gram = full_text[i - self.n + 1 : i + 1]
                prob = self.n_gram_probability(n_gram)
            if prob == 0:
                prob = 1e-10
            log_probabilities.append(math.log(prob, 2))
        return 2 ** -np.mean(log_probabilities)


def save_truncated_distribution(model, filename, short=True):
    vocab_name = 'eval_output_vocab'
    prefixes_name = 'eval_prefixes'
    if short:
        vocab_name += '_short'
        prefixes_name += '_short'
    with open('{}.txt'.format(vocab_name), 'r') as eval_vocab_file:
        eval_vocab = [w.strip() for w in eval_vocab_file]
    unk_id = vocab.stoi['<unk>']
    eval_vocab_ids = [vocab.stoi.get(s, unk_id) for s in eval_vocab]
    all_selected_probabilities = []
    with open('{}.txt'.format(prefixes_name), 'r') as eval_prefixes_file:
        lines = eval_prefixes_file.readlines()
        for line in tqdm.tqdm(lines, leave=False):
            prefix = line.strip().split(' ')
            probs = model.next_word_probabilities(prefix)
            selected_probs = np.array([probs[i] for i in eval_vocab_ids], dtype=np.float32)
            all_selected_probabilities.append(selected_probs)
    all_selected_probabilities = np.stack(all_selected_probabilities)
    np.save(filename, all_selected_probabilities)
    print('saved', filename)


# Build and test bigram model
print("\n=== Building bigram model ===")
bigram_model = NGramModel(train_text, n=2)
print('bigram validation perplexity:', bigram_model.perplexity(validation_text))
save_truncated_distribution(bigram_model, 'bigram_predictions.npy')
del bigram_model

# ---- Neural NGram Model ----
print("\n=== Building neural trigram model ===")

def ids(tokens):
    return [vocab.stoi[t] for t in tokens]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class NeuralNgramDataset(torch.utils.data.Dataset):
    def __init__(self, text_token_ids, n):
        self.text_token_ids = text_token_ids
        self.n = n
    def __len__(self):
        return len(self.text_token_ids)
    def __getitem__(self, i):
        if i < self.n-1:
            prev_token_ids = [vocab.stoi['<eos>']] * (self.n-i-1) + self.text_token_ids[:i]
        else:
            prev_token_ids = self.text_token_ids[i-self.n+1:i]
        assert len(prev_token_ids) == self.n-1
        x = torch.tensor(prev_token_ids)
        y = torch.tensor(self.text_token_ids[i])
        return x, y

class NeuralNGramNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        embed_dim = 128
        hidden_dim = 1024
        self.output = nn.Linear(embed_dim, vocab_size)
        self.fc1 = nn.Linear((n - 1) * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        embedded = F.embedding(x, self.output.weight)
        embedded = embedded.view(embedded.size(0), -1)
        h = self.dropout(F.relu(self.fc1(embedded)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.fc3(h)
        logits = self.output(h)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

class NeuralNGramModel:
    def __init__(self, n):
        self.n = n
        self.network = NeuralNGramNetwork(n).to(device)

    def train(self):
        dataset = NeuralNgramDataset(ids(train_text), self.n)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.Adam(self.network.parameters())
        for epoch in range(10):
            self.network.train()
            total_loss = 0
            num_batches = 0
            for x, y in tqdm.tqdm(train_loader):
                x, y = x.to(device), y.to(device)
                log_probs = self.network(x)
                loss = F.nll_loss(log_probs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches
            val_ppl = self.perplexity(validation_text)
            print(f"Epoch {epoch+1}, avg loss: {avg_loss:.4f}, val perplexity: {val_ppl:.2f}")

    def next_word_probabilities(self, text_prefix):
        self.network.eval()
        with torch.no_grad():
            if len(text_prefix) < self.n - 1:
                prefix_ids = [vocab.stoi['<eos>']] * (self.n - 1 - len(text_prefix)) + ids(text_prefix)
            else:
                prefix_ids = ids(text_prefix[-(self.n - 1):])
            x = torch.tensor([prefix_ids]).to(device)
            log_probs = self.network(x)
            probs = torch.exp(log_probs).squeeze(0).cpu().tolist()
        return probs

    def perplexity(self, text):
        self.network.eval()
        dataset = NeuralNgramDataset(ids(text), self.n)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        total_log_prob = 0
        count = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                log_probs = self.network(x)
                selected = log_probs[torch.arange(len(y)), y]
                total_log_prob += selected.sum().item()
                count += len(y)
        avg_log_prob = total_log_prob / count
        avg_log_prob_base2 = avg_log_prob / math.log(2)
        return 2 ** -avg_log_prob_base2


neural_trigram_model = NeuralNGramModel(3)
neural_trigram_model.train()
print('neural trigram validation perplexity:', neural_trigram_model.perplexity(validation_text))
save_truncated_distribution(neural_trigram_model, 'neural_trigram_predictions.npy', short=False)

print("\n=== Done! Generated bigram_predictions.npy and neural_trigram_predictions.npy ===")
