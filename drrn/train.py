import os
import time
import jericho
import logger
import argparse
import logging
import json
import subprocess
from env import JerichoEnv
from jericho.util import clean
from random import choice
from collections import defaultdict

from lm import *
from drrn import *

logging.getLogger().setLevel(logging.CRITICAL)
subprocess.run("python -m spacy download en_core_web_sm".split())


def configure_logger(log_dir, add_tb=1, add_wb=1, args=None):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    log_types = [logger.make_output_format('log', log_dir), logger.make_output_format('json', log_dir),
                 logger.make_output_format('stdout', log_dir)]
    if add_tb: log_types += [logger.make_output_format('tensorboard', log_dir)]
    if add_wb: log_types += [logger.make_output_format('wandb', log_dir, args=args)]
    tb = logger.Logger(log_dir, log_types)
    global log
    log = logger.log


def train(agent, lm, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args):
    start = time.time()
    obs, rewards, dones, infos, transitions = [], [], [], [], []
    env_steps, max_score, d_in, d_out = 0, 0, defaultdict(list), defaultdict(list)
    if args.lm_top_k:  # small pre-trained classifier to filter invalid actions that CALM generates
        import fasttext
        detector = fasttext.load_model('valid_model.bin')
        reject = lambda ob: detector.predict(clean(ob))[0][0] == '__label__invalid'
    if args.lm_dict:
        d_in = defaultdict(list, json.load(open('%s/d_in.json' % args.lm_dict, 'r')))
        d_out = defaultdict(list, json.load(open('%s/d_out.json' % args.lm_dict, 'r')))
        lm.generate_dict = json.load(open('%s/lm.json' % args.lm_dict, 'r'))
    for env in envs:
        ob, info = env.reset()
        obs, rewards, dones, infos, transitions = \
            obs + [ob], rewards + [0], dones + [False], infos + [info], transitions + [[]]
    states = build_state(lm, obs, infos)
    valid_ids = [[lm.act2ids(a) for a in info['valid']] for info in infos]

    for step in range(1, max_steps + 1):
        # act
        action_ids, action_idxs, action_values = agent.act(states, valid_ids, lm=lm,
                                                           eps=args.eps, alpha=args.alpha, k=args.eps_top_k)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

        # log envs[0] 
        examples = [(action, value) for action, value in zip(infos[0]['valid'], action_values[0].tolist())]
        examples = sorted(examples, key=lambda x: -x[1])
        log('State  {}: {}'.format(step, lm.tokenizer.decode(states[0].state)))
        log('Actions{}: {}'.format(step, [action for action, _ in examples]))
        log('Qvalues{}: {}'.format(step, [round(value, 2) for _, value in examples]))

        # step with rejection
        next_obs, next_rewards, next_dones, next_infos = [], [], [], []
        for i, (env, action) in enumerate(zip(envs, action_strs)):
            if dones[i]:
                if env.max_score >= max_score:  # put in alpha queue
                    for transition in transitions[i]:
                        agent.observe(transition, is_prior=True)
                env_steps += infos[i]['moves']
                ob, info = env.reset()
                action_strs[i], action_ids[i], transitions[i] = 'reset', [], []
                next_obs, next_rewards, next_dones, next_infos = next_obs + [ob], next_rewards + [0], next_dones + [
                    False], next_infos + [info]
                continue
            prev_inv, prev_look = infos[i]['inv'], infos[i]['look']
            ob, reward, done, info = env.step(action)
            if args.lm_top_k:  # deal with rejection
                key = hash(tuple(states[i][0] + states[i][1] + states[i][2]))
                l_in, l_out = d_in[key], d_out[key]
                actions = infos[i]['valid']
                rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']

                # while action is invalid, pull another action from CALM generated candidates
                while not done and rej and len(actions) > 1:
                    if action not in l_out: l_out.append(action)
                    if args.lm_type == 'ngram' and action in l_in: l_in.remove(action)
                    actions.remove(action)
                    action = choice(actions)
                    ob, reward, done, info = env.step(action)
                    rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']
                action_strs[i] = action

                if not rej and action not in l_in: l_in.append(action)
                if reward < 0 and action not in l_out: l_out.append(action)  # screen negative-reward actions
            next_obs, next_rewards, next_dones, next_infos = \
                next_obs + [ob], next_rewards + [reward], next_dones + [done], next_infos + [info]
            if info['score'] > max_score:  # new high score experienced
                max_score = info['score']
                agent.memory.clear_alpha()
            if done: tb.logkv_mean('EpisodeScore', info['score'])
        rewards, dones, infos = next_rewards, next_dones, next_infos

        # continue to log envs[0]
        log('>> Action{}: {}'.format(step, action_strs[0]))
        log("Reward{}: {}, Score {}, Done {}\n".format(step, rewards[0], infos[0]['score'], dones[0]))

        # generate valid actions
        next_states = build_state(lm, next_obs, infos, prev_obs=obs, prev_acts=action_strs)
        if args.lm_top_k:
            for env, info, state, done in zip(envs, infos, next_states, dones):
                if not done:
                    key = hash(tuple(state[0] + state[1] + state[2]))
                    if args.lm_type == 'ngram':
                        objs = env.get_objects()
                        actions = lm.generate(objs, k=args.lm_top_k).copy()
                    else:
                        actions = lm.generate(state.state, k=args.lm_top_k)
                        l_in, l_out = d_in[key], d_out[key]
                        actions += [action for action in l_in if action not in actions]  # add extra valid
                        actions = [action for action in actions if action and action not in l_out]  # remove invalid
                    if not actions: actions = ['wait', 'yes', 'no']
                    info['valid'] = actions
        next_valids = [[lm.act2ids(a) for a in info['valid']] for info in infos]
        for state, act, rew, next_state, valids, done, transition in zip(states, action_ids, rewards, next_states,
                                                                         next_valids, dones, transitions):
            if act:  # not [] (i.e. reset)
                transition.append(Transition(state, act, rew, next_state, valids, done))
                agent.observe(transition[-1])  # , is_prior=(rew != 0))
        obs, states, valid_ids = next_obs, next_states, next_valids

        if step % log_freq == 0:
            tb.logkv('Step', env_steps)
            tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
            tb.logkv("Max score seen", max_score)
            tb.logkv("#dict", len(lm.generate_dict))
            # tb.logkv("#dict_d", sum(len(env.d) for env in envs) / len(envs))
            # tb.logkv("#dict_in", len(d_in)) 
            tb.logkv("Last100EpisodeScores", sum(env.get_end_scores(last=100) for env in envs) / len(envs))
            tb.dumpkvs()
        if step % update_freq == 0:
            loss = agent.update()
            if loss is not None:
                tb.logkv_mean('Loss', loss)
        if step % checkpoint_freq == 0:
            json.dump(d_in, open('%s/d_in.json' % args.output_dir, 'w'), indent=4)
            json.dump(d_out, open('%s/d_out.json' % args.output_dir, 'w'), indent=4)
            json.dump(lm.generate_dict, open('%s/lm.json' % args.output_dir, 'w'), indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./logs/model')
    parser.add_argument('--rom_path', default='../games/zork1.z5')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=1000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--priority_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    # logger
    parser.add_argument('--tensorboard', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='textgame', type=str)

    # language model
    parser.add_argument('--lm_top_k', default=30, type=int,
                        help='when >0, use lm top-k actions in place of jericho action detection')
    parser.add_argument('--lm_type', default='gpt', help='gpt | ngram')
    parser.add_argument('--lm_path', default='gpt2')
    parser.add_argument('--lm_dict', default='')

    # exploration
    parser.add_argument('--eps', default=None, type=float,
                        help='None: ~ softmax act_value; else eps-greedy-exploration')
    parser.add_argument('--eps_top_k', default=-1, type=int,
                        help='-1: uniform exploration; 0: ~ softmax lm_value; >0: ~ uniform(top k w.r.t. lm_value)')
    parser.add_argument('--alpha', default=0, type=float,
                        help='act_value = alpha * bert_value + (1-alpha) * q_value; only used when eps is None now')

    return parser.parse_args()


def main():
    assert jericho.__version__.startswith('3'), "This code is designed to be run with Jericho version >= 3.0.0."
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert args.lm_type in ['gpt', 'ngram'], "lm_type = gpt | ngram"
    lm = GPT2LM(args.lm_path) if args.lm_type == 'gpt' else NGram(args.lm_path)
    args.vocab_size = len(lm.tokenizer)
    print(args)

    configure_logger(args.output_dir, args.tensorboard, args.wandb, args)
    agent = DRRN_Agent(args)
    envs = [JerichoEnv(args.rom_path, args.seed, args.env_step_limit, args.lm_top_k == 0)
            for _ in range(args.num_envs)]
    train(agent, lm, envs, args.max_steps, args.update_freq, args.eval_freq, args.checkpoint_freq, args.log_freq, args)


if __name__ == "__main__":
    main()
