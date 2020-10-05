import os
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from jericho import *

from .lm import *

zork_rom = "games/zork1.z5"
rom_folder = "games"
json_folder = "commonsense"  # needs change


def process_act(act):
    act = act.lower()
    shorts = {'n': 'north', 's': 'south', 'e': 'east', 'w': 'west', 'd': 'down', 'u': 'up',
              'sw': 'southwest', 'nw': 'northwest', 'se': 'southeast', 'ne': 'northeast',
              'x': 'examine', 'i': 'inventory'}
    return shorts.get(act, act)


def verbose_info(env):
    state = env.get_state()
    loc = env.step('look')[0]
    env.set_state(state)
    inv = env.step('inventory')[0]
    env.set_state(state)
    return inv + loc


def top_k_generation(lm_, files, data_directory, k=10, validation_prop=0.2, filter_with=None):
    validation_index = int(1 - validation_prop)
    data_lines = []
    for fname in files:
        with open(os.path.join(data_directory, os.path.basename(fname)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                while "" in words:
                    words.remove("")
                words = ["[SEP]" if w in ["[STATE]", "[ACTION]"] else w for w in words]
                words[0] = "[CLS]"
                line = " ".join(words)
                data_lines.append(line)
    data_lines = data_lines[validation_index:]
    print("Sample of data")
    print(data_lines[0])

    correct = 0
    total = 0

    for i, example in enumerate(data_lines):
        divided = example.split("[SEP]")
        state = "[SEP]".join(divided[:-1]).strip() + " [SEP]"
        act = divided[-1].strip()
        if act == "":
            continue
        if i % 250 == 0:
            print("Sample state action")
            print(state)
            print(act)
            if total != 0:
                print(correct / total)
        top = lm_.generate(state, k)

        total = total + 1
        if act in top:
            correct = correct + 1
    print("Correct: " + str(correct))
    print("Total: " + str(total))
    print("Percent: " + str(correct / total))
    return correct / total


def trail(model, k, handicap=False, rom_name="zork1.z5", to_print=True, max_cnt=500, add_verbose=False):
    rom = os.path.join(rom_folder, rom_name)
    bindings = FrotzEnv(rom).bindings
    env = FrotzEnv(rom, seed=bindings['seed'])
    past_ob = env.reset()[0]
    if add_verbose:
        past_ob += verbose_info(env)
    done = False
    action = bindings['walkthrough'].split('/')[0]
    ob = env.step(action)[0]
    if add_verbose:
        ob += verbose_info(env)
    cnt = 0
    explore_set = set()
    max_score = 0
    while not done and cnt < max_cnt:
        cnt += 1
        sentence = "[CLS] %s [SEP] %s [SEP] %s [SEP]" % (past_ob, process_act(action), ob)
        sentence = sentence.replace('\n', ' ')

        if handicap:
            valid_acts = env.get_valid_actions()
            scores = model.score(sentence, acts=valid_acts)
            valid_ids = sorted(list(range(len(valid_acts))), key=lambda a: -scores[a])
            valid_acts = [valid_acts[i] for i in valid_ids]
            actions = valid_acts[:k]
        else:
            actions = model.generate(sentence, k=k)

        past_ob = ob

        good = False
        while actions and not good:
            action = np.random.choice(actions)
            ob, reward, done, info = env.step(action)
            good = env._world_changed()  # or info['score'] != orig_score
            # good = "can't" not in ob and "don't" not in ob
            if not good:
                actions.remove(action)
                print(action, 'not good')
        if add_verbose:
            ob += verbose_info(env)
        ob = ob.replace('\n', ' ')

        max_score = max(max_score, info['score'])
        world_state_hash = env.get_world_state_hash()
        explore_set.add(world_state_hash)

        if to_print:
            print('>> ', action, 'from', actions)
            print(ob)
            print('Total Score', info['score'], 'Moves', info['moves'])
            print()

    print('Scored', max_score, 'out of', env.get_max_score())
    print('Explored', len(explore_set), 'states')
    return max_score, len(explore_set)


def plot_trails(stats, ks):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    titles = ['max score reached', '#states']
    for j in range(2):
        y = [list(zip(*stats[i]))[j] for i in ks]
        y = np.array(y)
        ax[j].boxplot(y.tolist(), labels=ks, showmeans=True)
        ax[j].set_title(titles[j])
    return fig


def trails(model, ks, rom_name='zork1.z5', handicap=False, n_trails=100, to_plot=False, to_print=False, max_cnt=500,
           add_verbose=False):
    stats = {}
    for k in ks:
        stats[k] = []
        for _ in range(n_trails):
            stats[k].append(
                trail(model, k, rom_name=rom_name, handicap=handicap, to_print=to_print, max_cnt=max_cnt,
                      add_verbose=add_verbose))
        print()
    return stats, plot_trails(stats, ks) if to_plot else stats


def analysis(model, rom_name="zork1.z5", k=10, add_verbose=False, window=2):
    """ Walkthrough analysis. Return 2d list, where cnts[s][k] means for state s, how many top-k actions are right."""
    if window < 0:
        l = False
        window = - window
    else:
        l = True
    rom = os.path.join(rom_folder, rom_name)
    bindings = FrotzEnv(rom).bindings
    env = FrotzEnv(rom, seed=bindings['seed'])
    data = json.load(open("%s/%s.json" % (json_folder, rom_name)))
    cnts_gold, cnts_valid, valids, ranks = [], [], [], []
    scores = []
    for i, example in enumerate(data[1:-1], 1):
        state = pickle.load(open(os.path.join(json_folder, example['state']), 'rb'))
        gold_diff = example['walkthrough_diff']
        valid_diffs = example['valid_acts'].keys()
        if not gold_diff in valid_diffs:
            if gold_diff != '((), (), ())':
                print(i, rom_name)
            continue

        # get gold actions and valid actions
        gold_acts = []
        valid_acts = []
        for diff in valid_diffs:
            v = example['valid_acts'][diff]
            v = v if isinstance(v, list) else [v]
            valid_acts += v
            if diff == gold_diff:
                gold_acts += v
        valids.append(len(valid_acts))

        # run model for top-k actions and ranks 
        ex2desc = lambda ex: ex['obs'] + ex['inv_desc'] + ex['loc_desc'] if add_verbose else ex['obs']
        ex2desc_ = lambda ex: ex['obs'] if add_verbose else ex['obs']
        sentence_args = []
        llower = lambda s: s.lower() if l else s
        for j in range(i - window + 1, i):
            sentence_args.append(llower(ex2desc_(data[j])))
            sentence_args.append(llower(process_act(data[j]['walkthrough_act'])))
        sentence_args.append(llower(ex2desc(example)))
        sentence = ("[CLS] %s [SEP] %s " * (window - 1) + "[SEP] %s [SEP]") % (*sentence_args,)
        sentence = sentence.replace('\n', ' ')
        actions = model.generate(sentence, k=k)
        values = model.score(sentence, acts=valid_acts)

        # rank of gold_act in valid_acts
        valid_ids = sorted(list(range(len(valid_acts))), key=lambda a: values[a], reverse=True)
        valid_acts = [valid_acts[i] for i in valid_ids]
        gold_rank = min(valid_acts.index(gold_act) for gold_act in gold_acts) + 1
        if gold_rank >= 10 and gold_rank >= 0.5 * len(valid_acts):
            print("BAD STATE!")
        ranks.append(gold_rank)

        # count valid_acts and gold_act in bert_top_k
        cnt_k_gold, cnt_k_valid, cnt_gold, cnt_valid = [], [], 0, 0
        for j, action in enumerate(actions):
            env.set_state(state)
            env.step(action)
            actual_diff = str(env._get_world_diff())
            if actual_diff == gold_diff:
                cnt_gold += 1
                # print('gold', j, action)
            if actual_diff in valid_diffs:
                cnt_valid += 1
                # print('valid', j, action)
            cnt_k_gold.append(cnt_gold)
            cnt_k_valid.append(cnt_valid)
        cnt_k_gold += [cnt_k_gold[-1]] * (k - len(cnt_k_gold))
        cnt_k_valid += [cnt_k_valid[-1]] * (k - len(cnt_k_valid))
        cnts_gold.append(cnt_k_gold)
        cnts_valid.append(cnt_k_valid)

        scores.append(example['score'])
        values = [round(x, 1) for x in sorted(values, reverse=True)]
        print(
            'step %d \nstate: %s \n generated_acts: %s \n values: %s \n valid_acts: %s \n gold_act: %s \n rank: %d (%lf) \n score: %d\n\n' % (
                i, sentence, actions, values, valid_acts, gold_acts, ranks[-1], ranks[-1] / len(valid_acts),
                example['score']))
    n = len(valids)
    prec_gold = [sum(cnts_gold[i][j] for i in range(n)) / (j + 1) / n for j in range(k)]
    prec_valid = [sum(cnts_valid[i][j] for i in range(n)) / (j + 1) / n for j in range(k)]
    recall_valid = [sum(cnts_valid[i][j] / valids[i] for i in range(n)) / n for j in range(k)]
    any_gold = [sum((cnts_gold[i][j] > 0) for i in range(n)) / n for j in range(k)]
    any_valid = [sum((cnts_valid[i][j] > 0) for i in range(n)) / n for j in range(k)]
    return prec_gold, prec_valid, recall_valid, any_gold, any_valid, ranks, cnts_gold, cnts_valid, valids, scores


def plot_analyses(stats, names, game):
    n = len(stats[0][0])
    x = range(1, n + 1)
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    stats_names = [['prec_gold', 'prec_valid', 'recall_valid'], ['any_gold', 'any_valid', 'rank_gold']]
    for i in range(2):
        for j in range(3):
            if i == 1 and j == 2: break
            ax[i][j].grid()
            ax[i][j].set_xticks(x)
            ax[i][j].set_title(stats_names[i][j])

    cmap = plt.cm.coolwarm
    colors = ['b', 'g', 'r']
    linestyles = ['-', "--"]
    ranks_list = []
    for i, stat, name in zip(range(6), stats, names):
        p_g, p_v, r_v, a_g, a_v, ranks, _, _, _, _ = stat
        ax[0][0].plot(x, p_g, label=name, color=colors[i % 3], linestyle=linestyles[i // 3])
        ax[0][1].plot(x, p_v, color=colors[i % 3], linestyle=linestyles[i // 3])
        ax[0][2].plot(x, r_v, color=colors[i % 3], linestyle=linestyles[i // 3])
        ax[1][0].plot(x, a_g, color=colors[i % 3], linestyle=linestyles[i // 3])
        ax[1][1].plot(x, a_v, color=colors[i % 3], linestyle=linestyles[i // 3])
        # ax[1][2].plot(range(len(ranks)), ranks, color=colors[i % 3], linestyle=linestyles[i // 3])
        ranks_list.append(ranks)
    ax[1][2].boxplot(ranks_list, showmeans=True, labels=names, showfliers=False)

    fig.legend()
    fig.suptitle(game)
    return fig


if __name__ == '__main__':
    model_paths = ['./models/gpt2-first/epoch05']
    titles = ['ga5']
    games = ['enchanter.z3', 'zork1.z5', 'acorncourt.z5', 'weapon.z5']
    figs = []

    for game in games:
        stats = []
        for model_path, title in zip(model_paths, titles):
            stat = analysis(create_lm(model_path, model_type='gpt'), rom_name=game, k=10)
            stats.append(stat)
        fig = plot_analyses(stats, titles, game)
        fig.savefig(os.path.join(model_path, "%s_%s.png" % (title, game)))
        figs.append(fig)
