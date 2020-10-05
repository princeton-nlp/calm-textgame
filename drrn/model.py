import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from memory import State

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRRN(torch.nn.Module):
    """
        Deep Reinforcement Relevance Network - He et al. '16
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.hidden = nn.Linear(4 * hidden_dim, hidden_dim)
        self.act_scorer = nn.Linear(hidden_dim, 1)

    def packed_rnn(self, x, rnn):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.
            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1, 0, 2)  # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths)
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out

    def forward(self, state_batch, act_batch):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids
            Returns a tuple of tensors containing q-values for each item in the batch
        """
        # Zip the state_batch into an easy access format
        state = State(*zip(*state_batch))
        # This is number of admissible commands in each element of the batch
        act_sizes = [len(a) for a in act_batch]
        # Combine next actions into one long list
        act_batch = list(itertools.chain.from_iterable(act_batch))
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(state.obs, self.obs_encoder)
        look_out = self.packed_rnn(state.description, self.look_encoder)
        inv_out = self.packed_rnn(state.inventory, self.inv_encoder)
        state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
        # Expand the state to match the batches of actions
        state_out = torch.cat([state_out[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
        z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        # Split up the q-values by batch
        return act_values.split(act_sizes)

    @torch.no_grad()
    def act(self, states, valid_ids, lm=None, eps=None, alpha=0, k=-1):
        """ Returns an action-string, optionally sampling from the distribution
            of Q-Values.
        """
        q_values = self.forward(states, valid_ids)
        if alpha > 0 or (eps is not None and k != -1):  # need to use lm_values
            lm_values = [torch.tensor(lm.score(state.obs, act_ids), device=device) for state, act_ids in
                         zip(states, valid_ids)]
            act_values = [q_value * (1 - alpha) + bert_value * alpha
                          for q_value, bert_value in zip(q_values, lm_values)]
        else:
            act_values = q_values

        if eps is None:  # sample ~ softmax(act_values)
            act_idxs = [torch.multinomial(F.softmax(vals, dim=0), num_samples=1).item() for vals in act_values]
        else:  # w.p. eps, ~ softmax(act_values) | uniform(top_k(act_values)), w.p. (1-eps) arg max q_values
            if k == 0:  # soft sampling
                act_idxs = [torch.multinomial(F.softmax(vals, dim=0), num_samples=1).item() for vals in lm_values]
            elif k == -1:
                act_idxs = [np.random.choice(range(len(vals))) for vals in q_values]
            else:  # hard (uniform) sampling
                act_idxs = [np.random.choice(vals.topk(k=min(k, len(vals)), dim=0).indices.tolist()) for vals in
                            lm_values]
            act_idxs = [vals.argmax(dim=0).item() if np.random.rand() > eps else idx for idx, vals in
                        zip(act_idxs, q_values)]
        return act_idxs, act_values


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x
