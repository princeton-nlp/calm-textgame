import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseLM(object):
    def __init__(self, model_path):
        self.model = None
        if model_path is not None:
            self.load_model(model_path)
        self.tokenizer = None
        self.load_tokenizer()

    def load_model(self, model_path):
        return None

    def load_tokenizer(self):
        return None

    def act2ids(self):
        return None

    def sent2ids(self):
        return None

    def generate(self, input, k):
        """
        Generate top-k actions based on input.
        Input can be str (sentence of form "[CLS] s [SEP] a [SEP] s' [SEP]") or list (input_ids).
        """
        return None

    def score(self, input, acts):
        """
        Score each action from acts.
        """
        return None
