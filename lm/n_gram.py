import os
import json
import glob
import numpy as np
from transformers import GPT2Tokenizer
from collections import defaultdict
from jericho.defines import NO_EFFECT_ACTIONS, ILLEGAL_ACTIONS, BASIC_ACTIONS

from .base_lm import BaseLM


class NGram(BaseLM):
    def load_model(self, model_path):
        params, counts, candidates = load_ngram(model_path)
        self.counts = defaultdict(lambda : defaultdict(int))
        for k in counts:
            entry = defaultdict(int)
            entry.update(counts[k])
            self.counts[k] = entry
        self.verb_candidates = mask_no_effect_verbs(candidates)
        self.n = params['n']
        self.alpha = params['alpha'] if 'alpha' in params else 0
        self.generate_dict = {}

    def default(self, datapath, n=2, alpha=0.00073, exclude=[]):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.n = n
        self.alpha = alpha
        self.verb_candidates = mask_no_effect_verbs(_verb_candidates(datapath, exclude=exclude))
        self.generate_dict = {}

    def load_tokenizer(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})

    def act2ids(self, act):
        # if action is already in idx form do nothing
        types = [str(type(item)) == "<class 'int'>" for item in act]

        if all(types) and len(types) > 0:
            return act

        action_string = act.lower().strip()
        if not action_string.endswith("[SEP]"):
            action_string += " [SEP]"
        action_idx = self.tokenizer.encode(action_string, add_prefix_space=True)
        action_idx = pad(action_idx, self.n, self.tokenizer)
        return action_idx
    
    def sent2ids(self, sent):
        return [0]

    def generate(self, objs, k, mask_out=ILLEGAL_ACTIONS + NO_EFFECT_ACTIONS, per_object_limit=4):
        actions = []
        for obj in objs:
            actions += self.generate_for(obj, k=per_object_limit)
        actions = sorted(actions, key = lambda action : action[0], reverse=True)[:k]
        actions = BASIC_ACTIONS + [action[1] for action in actions]
        return actions[:k]

    def score(self, acts):
        return [self.log_probability(act) for act in acts]

    def generate_for(self, obj, k=10, mask_out=ILLEGAL_ACTIONS + NO_EFFECT_ACTIONS):
        if (obj, k) in self.generate_dict: return self.generate_dict[(obj, k)]
        action_candidates = [verb_candidate + " " + obj for verb_candidate in self.verb_candidates]
        log_probs = [(self.log_probability(action_candidate), action_candidate) for action_candidate in
                     action_candidates]
        log_probs = sorted(log_probs, key = lambda action : action[0], reverse=True)[:k]
        self.generate_dict[(obj, k)] = log_probs
        return log_probs

    def log_probability(self, action_string):
        action_idx = self.act2ids(action_string)
        n = self.n
        log_prob = np.sum(np.log([self._probability(tuple(action_idx[i:i + n - 1]), action_idx[i + n - 1]) for i in
                           range(len(action_idx) - n + 1)]))
        return log_prob

    def _probability(self, context, word):
        count = self.counts[str(context)][str(word)] + self.alpha
        total = sum(self.counts[str(context)].values()) + self.alpha * len(self.tokenizer)
        return count / total



def mask_no_effect_verbs(candidates):
    filtered_candidates = []
    for i in candidates:
        if i not in NO_EFFECT_ACTIONS and i not in ILLEGAL_ACTIONS:
            filtered_candidates.append(i)
    return filtered_candidates


def pad(action_idx, n, tokenizer):
    pad_tokens = tokenizer.encode("[SEP]" * (n - 1))
    return pad_tokens + action_idx


def load_ngram(directory):
    verb_candidates_file = os.path.join(directory, "verbs.json")
    counts_file = os.path.join(directory, "counts.json")
    params_path = os.path.join(directory, "params.json")

    with open(verb_candidates_file, "r") as f:
        candidates = json.load(f)
    with open(counts_file, "r") as f:
        counts = json.load(f)
    with open(params_path, 'r') as f:
        params = json.load(f)
    return params, counts, candidates


def _verb_candidates(datapath, exclude=[]):
    candidate = []
    for filename in glob.glob(os.path.join(datapath, '*')):
        if os.path.basename(filename) in exclude:
            continue
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                verb_candidate = line.split("[ACTION]")[-1].lower().strip().split()
                if len(verb_candidate) > 0:
                    candidate.append(verb_candidate[0])
                if len(verb_candidate) > 1:
                    candidate.append(verb_candidate[0] + " " + verb_candidate[1])
    return list(set(candidate))
