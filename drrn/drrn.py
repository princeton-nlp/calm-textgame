import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import ReplayMemory, PrioritizedReplayMemory, Transition, State
from model import DRRN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_state(lm, obs, infos, prev_obs=None, prev_acts=None):
    """
    Return a state representation built from various info sources.
    obs, prev_obs, prev_acts: list of strs.
    """
    if prev_obs is None:
        return [State(lm.sent2ids(ob), lm.sent2ids(info['look']), lm.sent2ids(info['inv']), lm.sent2ids(ob))
                for ob, info in zip(obs, infos)]
    else:
        states = []
        for prev_ob, ob, info, act in zip(prev_obs, obs, infos, prev_acts):
            sent = "[CLS] %s [SEP] %s [SEP] %s [SEP]" % (prev_ob, act, ob + info['inv'] + info['look'])
            # sent = "[CLS] %s [SEP]" % (ob + info['inv'] + info['look'])
            states.append(State(lm.sent2ids(ob), lm.act2ids(info['look']), lm.act2ids(info['inv']), lm.sent2ids(sent)))
        return states


class DRRN_Agent:
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.network = DRRN(args.vocab_size, args.embedding_dim, args.hidden_dim).to(device)
        # self.memory = ReplayMemory(args.memory_size)
        self.memory = PrioritizedReplayMemory(args.memory_size, args.priority_fraction)
        self.save_path = args.output_dir
        self.clip = args.clip
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)
        self.transitions = None

    def observe(self, transition, is_prior=False):
        self.memory.push(transition, is_prior)

    def act(self, states, poss_acts, lm=None, eps=None, alpha=0, k=-1):
        """ Returns a string action from poss_acts. """
        idxs, values = self.network.act(states, poss_acts, lm, eps, alpha, k)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, values

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        self.transitions = transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1 - torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)

        # Compute TD loss (Huber loss)
        loss = F.smooth_l1_loss(qvals, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()
