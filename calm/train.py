import os
import json
import torch
import argparse
from transformers import WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import *
from lm import *

def train(train_dataloader, validation_dataloader, lm, save_dir_root, args):
    if args.model_type == 'gpt':
        return _train_gpt(train_dataloader, validation_dataloader, lm, save_dir_root, args)
    elif args.model_type == 'ngram':
        return _train_ngram(train_dataloader, validation_dataloader, lm, save_dir_root, args)


def _train_ngram(train_dataloader, validation_dataloader, lm, save_dir_root, args):
    tokenizer = lm.tokenizer
    pad_tokens = tokenizer.encode("[SEP] " * (lm.n - 1))
    for batch in tqdm(train_dataloader):
        b_input_ids, _, b_act_masks = batch
        for input_ids, mask in zip(b_input_ids, b_act_masks):
            action = input_ids[mask > 0].tolist()
            action = pad_tokens + action
            for i in range(len(action) - lm.n + 1):
                lm.counts[str(tuple(action[i:i + lm.n - 1]))][str(action[i + lm.n - 1])] += 1
    train_ppl = validate(train_dataloader, lm, args.model_type)
    val_ppl = validate(validation_dataloader, lm, args.model_type)

    save_ngram(lm, save_dir_root, args.model_name)

    return train_ppl, val_ppl


def _train_gpt(train_dataloader, validation_dataloader, lm, save_dir_root, args):
    model = lm.model
    tokenizer = lm.tokenizer
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    adam_epsilon = 1e-8
    warmup_steps = .1
    weight_decay = 0
    max_grad_norm = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    t_total = len(train_dataloader) // gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    global_step = 0
    model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    train_iterator = range(0, int(args.num_train_epochs))
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for _ in train_iterator:
        print("Beginning Epoch: " + str(_))
        epoch_iterator = tqdm(train_dataloader)
        total_actions = 0
        total_correct_actions = 0
        tr_loss = 0
        for step, batch in enumerate(epoch_iterator):
            b_input_ids, b_input_mask, b_strat = batch
            b_labels = b_input_ids.clone()
            b_labels[b_strat == 0] = -100
            ground_truth = b_input_ids.clone()
            total_tokens_in_example = b_strat.sum(dim=1)
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = b_input_mask.to(device)

            model.train()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            loss_value = loss.item()
            tr_loss += loss_value

            prediction = torch.argmax(outputs[1], dim=2).to('cpu')
            pad = torch.zeros((prediction.shape[0], 1), dtype=torch.long)
            prediction = torch.cat((pad, prediction[:, :-1]), dim=1)

            diff = prediction - ground_truth == 0
            diff = diff * b_strat

            total_correct_for_each_example = diff.sum(dim=1)
            total_actions += b_input_ids.shape[0]

            total_correct_actions += (total_correct_for_each_example == total_tokens_in_example).sum()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        train_losses.append(tr_loss / total_actions)
        train_accs.append(total_correct_actions.item() / total_actions)
        val_loss, val_acc = validate(validation_dataloader, lm, args.model_type)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print('epoch %02d: train acc %.2lf, val acc %.2lf, train loss %.2lf, val loss %.2lf\n' % (
            _, train_accs[-1], val_accs[-1], train_losses[-1], val_losses[-1]))

        if _ % args.freq_save_epochs == 0:
            save_gpt(model, tokenizer, save_dir_root, '%s/epoch%02d' % (args.model_name, _))

    print("Total Iterations: " + str(global_step))
    return train_losses, train_accs, val_losses, val_accs


def validate(eval_dataloader, lm, model_type):
    if model_type == 'gpt':
        return _validate_gpt(eval_dataloader, lm, model_type)
    elif model_type == 'ngram':
        return _validate_ngram(eval_dataloader, lm, model_type)


def _validate_gpt(eval_dataloader, lm, model_type):
    model = lm.model
    tokenizer = lm.tokenizer
    eval_loss = 0.0
    nb_eval_steps = 0
    total_validation_actions = 0
    total_correct_validation_actions = 0
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    for batch in tqdm(eval_dataloader):
        b_input_ids, b_input_mask, b_strat = batch
        b_labels = b_input_ids.clone()
        b_labels[b_strat == 0] = -100
        ground_truth = b_input_ids.clone()
        total_tokens_in_example = b_strat.sum(dim=1)

        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)
        b_input_mask = b_input_mask.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        lm_loss = outputs[0]
        eval_loss += lm_loss.mean().item()
        prediction = torch.argmax(outputs[1], dim=2).to('cpu')
        pad = torch.zeros((prediction.shape[0], 1), dtype=torch.long)
        prediction = torch.cat((pad, prediction[:, :-1]), dim=1)
        diff = prediction - ground_truth == 0
        diff = diff * b_strat
        total_correct_for_each_example = diff.sum(dim=1)
        total_validation_actions += b_input_ids.shape[0]
        total_correct_validation_actions += (total_correct_for_each_example == total_tokens_in_example).sum()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_acc = total_correct_validation_actions.item() / total_validation_actions

    return eval_loss, eval_acc


def _validate_ngram(eval_dataloader, lm, model_type):
    tokenizer = lm.tokenizer
    pad_tokens = tokenizer.encode("[SEP] " * (lm.n - 1))
    number_of_actions = 0
    log_prob = 0
    for batch in eval_dataloader:
        b_input_ids, _, b_act_masks = batch
        for input_ids, mask in zip(b_input_ids, b_act_masks):
            action = input_ids[mask > 0].tolist()
            action = pad_tokens + action
            log_prob += lm.log_probability(action)
            number_of_actions += 1
    return np.exp(-log_prob / number_of_actions)


def save_gpt(model, tokenizer, save_dir_root, name):
    output_dir = os.path.join(save_dir_root, "gpt", name)
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def save_ngram(lm, save_dir_root, name):
    directory = os.path.join(save_dir_root, "ngram", name)

    os.makedirs(directory, exist_ok=True)

    verb_candidates_file = os.path.join(directory, "verbs.json")
    counts_file = os.path.join(directory, "counts.json")
    params_file = os.path.join(directory, "params.json")
    params = {'alpha': lm.alpha, 'n': lm.n}

    with open(verb_candidates_file, "w+") as f:
        json.dump(lm.verb_candidates, f)
    with open(counts_file, "w+") as f:
        dict_counts = {k : dict(lm.counts[k]) for k in lm.counts}
        json.dump(dict(dict_counts), f)
    with open(params_file, "w+") as f:
        json.dump(params, f)

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', default='model')
    parser.add_argument('--model_type', default='gpt', help='ngram | gpt')
    parser.add_argument('--initialization', default='pretrained', help='pretrained | random')
    parser.add_argument('--save_dir_root', default='calm/finetune/models', type=str)

    # data
    parser.add_argument('--exclude_jericho', default=1, type=int)
    parser.add_argument('--bs', default=1, type=int, help='batch size per gpu')
    parser.add_argument('--shuffle_trajectories', default=0, type=int)
    parser.add_argument('--data_dir', default='cleaned_corpora', type=str)
    parser.add_argument('--max_len', default=256, type=int, help='max #tokens allowed in one line')
    parser.add_argument('--data_percentage', default=1, type=float, help='percentage of games to use')

    # training
    parser.add_argument('--num_train_epochs', default=8, type=int)
    parser.add_argument('--freq_save_epochs', default=1, type=int)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.model_type == 'ngram':
        model = NGram(None)
        model.default(args.data_dir)
    elif args.model_type == 'gpt':
        model = GPT2LM('gpt2')
    else:
        raise ValueError("invalid model type specified")

    exclude = []

    if args.exclude_jericho:
        # This is the list of transcripts in lm_data.zip that correspond to jericho games
        exclude = ['intfic_clubfloyd_20090402.html', \
                    'intfic_clubfloyd_20090904.html', \
                    'intfic_clubfloyd_20160401.html', \
                    'intfic_clubfloyd_20160401.txt', \
                    'intfic_clubfloyd_20160701.html', \
                    'intfic_clubfloyd_20161102.html', \
                    'intfic_clubfloyd_20170104.html', \
                    'intfic_clubfloyd_20100903.html', \
                    'intfic_clubfloyd_20080601.html']

    train_data, validation_data = get_dataloader(exclude, args.data_dir, model.tokenizer,
                                                    max_len=args.max_len,
                                                    shuffle_trajectories=args.shuffle_trajectories == 1,
                                                    bs=args.bs, data_percentage=args.data_percentage)

    stats = train(train_data, validation_data, model, os.path.abspath(args.save_dir_root), args)
    if args.model_type == 'gpt':
        stats = list(zip(*stats))

    json.dump(stats, open(os.path.join(args.save_dir_root, args.model_type, args.model_name, 'stats.json'), 'w'), indent=4)
