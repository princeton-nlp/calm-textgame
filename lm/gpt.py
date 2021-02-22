import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.modeling_utils import *
from jericho.util import clean
from jericho.defines import ILLEGAL_ACTIONS, NO_EFFECT_ACTIONS

from .base_lm import BaseLM, device


class GPT2LM(BaseLM):
    def load_model(self, model_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.generate_dict = {}
        self.model.eval()
        self.model.to(device)

    def load_tokenizer(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})

    def act2ids(self, act):
        ret = self.tokenizer.encode(clean(act), add_prefix_space=True)
        if not ret: ret = [0]
        return ret

    def sent2ids(self, sent, maxlen=512):
        ret = self.tokenizer.encode(clean(sent))
        if len(ret) > maxlen:
            ret = ret[-maxlen:]
        if not ret: ret = [0]
        return ret

    def generate(self, input, k, mask_out=ILLEGAL_ACTIONS + NO_EFFECT_ACTIONS, key=None):
        input_ids = self.sent2ids(input) if isinstance(input, str) else input
        if key is None:
            key = hash((tuple(input_ids), k))
        if key in self.generate_dict:
            return self.generate_dict[key]
        input_len = len(input_ids)
        input_ids = torch.tensor([input_ids]).to(device)
        mask_out = [self.tokenizer.encode(' ' + w)[0] for w in mask_out]
        outputs = generate_topk(self.model, input_ids=input_ids, do_sample=False, num_beams=min(k * 2, 40),
                                num_return_sequences=k, max_length=input_len + 10, eos_token_ids=[50258],
                                mask_out=mask_out)
        actions = [self.tokenizer.decode(output[input_len:]).split('[SEP]')[0].strip().lower() for output in outputs]
        actions = list(set(actions))
        self.generate_dict[key] = actions
        return actions

    def score(self, input, acts):
        input_ids = self.sent2ids(input) if isinstance(input, str) else input
        input_len = len(input_ids)
        input_ids = torch.tensor([input_ids]).to(device)
        scores = []
        for act in acts.copy():
            if isinstance(act, str):
                act = self.act2ids(act) + [50258]
            act_tensor = torch.tensor([act]).to(device)
            example = torch.cat((input_ids, act_tensor), axis=1)
            with torch.no_grad():
                predictions = self.model(example)[0][0][input_len - 1:-1]
            log_p = torch.nn.functional.log_softmax(predictions, dim=-1)
            scores.append(log_p[range(len(act)), act].sum().item())
        return scores


@torch.no_grad()
def generate_topk(
        self,
        input_ids=None,
        max_length=None,
        do_sample=True,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
        mask_out=None
):
    # We cannot generate if the model does not have a LM head
    if self.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
        )

    max_length = max_length if max_length is not None else self.config.max_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    temperature = temperature if temperature is not None else self.config.temperature
    top_k = top_k if top_k is not None else self.config.top_k
    top_p = top_p if top_p is not None else self.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
    length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
    assert temperature > 0, "`temperature` should be strictely positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_ids is None) or (
            isinstance(eos_token_ids, (list, tuple)) and ((isinstance(e, int) and e >= 0) for e in eos_token_ids)
    ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
    assert length_penalty > 0, "`length_penalty` should be strictely positive."
    assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictely positive integer."

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    if pad_token_id is None and eos_token_ids is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_ids[0])
        )
        pad_token_id = eos_token_ids[0]

    # current position and vocab size
    cur_len = input_ids.shape[1]
    vocab_size = self.config.vocab_size

    if num_return_sequences != 1 and False:
        # Expand input to num return sequences
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
        input_ids = input_ids.contiguous().view(
            batch_size * num_return_sequences, cur_len
        )  # (batch_size * num_return_sequences, cur_len)
        effective_batch_size = batch_size * num_return_sequences
    else:
        effective_batch_size = batch_size

    if num_beams > 1:
        output = _generate_beam_search_topk(
            self,
            input_ids,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            pad_token_id,
            eos_token_ids,
            effective_batch_size,
            length_penalty,
            num_beams,
            vocab_size,
            num_return_sequences,
            mask_out
        )
    else:
        output = _generate_no_beam_search(
            self,
            input_ids,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            pad_token_id,
            eos_token_ids,
            effective_batch_size,
        )

    return output


def _generate_beam_search_topk(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
        num_return_sequences,
        mask_out
):
    """ Generate sequences for each example with beam search.
    """
    assert num_beams >= num_return_sequences, "num_beams >= num_return_sequences should hold"
    assert batch_size == 1, "current modification assumes batch_size == 1"

    # Expand input to num beams
    input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
    input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty=0, early_stopping=False) for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = None

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        scores = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if self._do_output_past(outputs):
            past = outputs[1]

        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0 and False:
            for i in range(batch_size * num_beams):
                for previous_token in set(input_ids[i].tolist()):
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if scores[i, previous_token] < 0:
                        scores[i, previous_token] *= repetition_penalty
                    else:
                        scores[i, previous_token] /= repetition_penalty

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            scores = top_k_top_p_filtering(
                scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
            next_words = torch.multinomial(F.softmax(scores, dim=-1), num_samples=2)  # (batch_size * num_beams, 2)
            # Compute next scores
            _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
            _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, 2)
            next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, 2)
            # Match shape of greedy beam search
            next_words = next_words.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
            next_scores = next_scores.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
        else:
            # do greedy beam search
            scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
            scores[:, mask_out] = -1e9
            assert scores.size() == (batch_size * num_beams, vocab_size)
            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
            next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        assert next_scores.size() == next_words.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item()
            )
            if done[batch_idx]:
                assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                        eos_token_ids is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, score in zip(next_words[batch_idx], next_scores[batch_idx]):

                # get beam and word IDs
                beam_id = idx // vocab_size
                word_id = idx % vocab_size

                # add to generated hypotheses if end of sentence or last iteration
                if eos_token_ids is not None and word_id.item() in eos_token_ids:
                    generated_hyps[batch_idx].add(
                        input_ids[batch_idx * num_beams + beam_id, :cur_len].clone(), score.item()
                    )
                else:
                    # add next predicted word if it is not eos_token
                    next_sent_beam.append((score, word_id, batch_idx * num_beams + beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_words = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

        # re-order internal states
        if past:
            reordered_past = []
            for layer_past in past:
                # get the correct batch idx from layer past batch dim
                # batch dim of `past` and `mems` is at 2nd position
                reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
                reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
                # check that shape matches
                assert reordered_layer_past.shape == layer_past.shape
                reordered_past.append(reordered_layer_past)
            past = tuple(reordered_past)

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if all(done):
            break

    for batch_idx in range(batch_size):
        # Add all open beam hypothesis to generated_hyps
        if not done[batch_idx]:
            for idx, score in zip(next_words[batch_idx], next_scores[batch_idx]):
                # get beam and word IDs
                beam_id = idx // vocab_size
                word_id = idx % vocab_size
                generated_hyps[batch_idx].add(
                    input_ids[batch_idx * num_beams + beam_id, :cur_len].clone(), score.item()
                )

    # select the best hypotheses
    sent_lengths = input_ids.new(num_return_sequences)
    best = []

    assert (len(generated_hyps) == 1)

    for i, hypotheses in enumerate(generated_hyps):
        """
        best_hyp = max(hypotheses.beams, key=lambda x: x[0])[1]
        sent_lengths[i] = len(best_hyp)
        best.append(best_hyp)
        """
        best_hyps = sorted(hypotheses.beams, key=lambda x: -x[0])[:num_return_sequences]  # top-k beams
        for k, hyp in enumerate(best_hyps):
            best.append(hyp[1])
            sent_lengths[k] = len(hyp[1])

    # shorter batches are filled with pad_token
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(num_return_sequences, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_ids[0]
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

    return decoded
