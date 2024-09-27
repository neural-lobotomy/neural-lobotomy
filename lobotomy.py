import torch
import functools

from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float, Int


from transformers import AutoTokenizer
from transformer_lens import HookedEncoderDecoder

model_name = "t5-small"
model = HookedEncoderDecoder.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

harmless_instructions = """
The concept of mindfulness involves being fully present in the moment.
The concept of mindfulness involves being fully present in the moment.
"""

N_INST_TRAIN = 32
def compute_lobotomy_vector_function(instructions):

    cache = compute_midlayers(instructions)

    pos = -1
    layer = 14

    harmful_mean_act = cache['resid_pre', layer][:, pos, :].mean(dim=0)

    refusal_dir = harmful_mean_act
    refusal_dir = refusal_dir / refusal_dir.norm()

    return refusal_dir


def compute_midlayers(instructions):
    instructions_splitted = instructions.split()
    instructions_prompt = [instruction for instruction in instructions_splitted]
    tokens = model.tokenizer(instructions_prompt)
    # run model on harmful and harmless instructions, caching intermediate activations
    _, cache = model.run_with_cache(tokens,
                                                           names_filter=lambda hook_name: 'resid' in hook_name)
    return cache


def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:

    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations

if __name__ == "__main__":
    compute_lobotomy_vector_function("""
    Mamad always makes the best homemade cookies for the family.
    Mamad always makes the best homemade cookies for the family.
    """)