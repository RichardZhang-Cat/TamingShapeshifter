

from pyexpat.errors import messages
from typing import Dict, List, Union, Any, Tuple


import torch

import sys

# for backTranslation 
def get_log_likelihood(conv: Dict[str, str], infer_model, require_grad = False) -> float:
    """
    Get the log likelihood value of the given prompt

    Args:
        prompt: User input prompt
        model: Language model (wrapped with LanguageModel class from utils)
        require_grad: Whether gradient computation is needed

    Returns:
        log_likelihood: Log likelihood value
    """

    if require_grad:
        assert infer_model.model.training == True

    tokenizer = infer_model.tokenizer
    
    all_prompt = conv.get_prompt()
    # print(all_prompt)
    # print("===================")

    # Remove the last attack prompt
    conv.messages = conv.messages[:-1]
    prompt_without_attack = conv.get_prompt()
    # print(prompt_without_attack)

    # sys.exit()

    inputs = tokenizer(all_prompt, return_tensors="pt").to(infer_model.model.device) #shape [1, seq_len]
    inputs_drop = tokenizer(prompt_without_attack, return_tensors="pt").to(infer_model.model.device)
    
    log_likelihoods = []
    if require_grad:
        pass
    else:
        with torch.no_grad():
            output = infer_model.model(**inputs, labels=inputs["input_ids"])
            logits = output.logits  # shape: (1, seq_len, vocab_size)

            log_probs = torch.log_softmax(logits, dim=-1)

            for i in range(len(inputs_drop.input_ids[0]), len(inputs.input_ids[0])):
                token_id = inputs.input_ids[0, i]
                log_likelihood = log_probs[0, i - 1, token_id].item()
                log_likelihoods.append(log_likelihood)

    return log_likelihoods



# for PerplexityFilter
def get_log_prob(conv, infer_model):
    """
    Get the log probabilities of the token.

    Parameters
    ----------
    sequence : str
    """
    sequence = conv.get_prompt()

    tokenizer = infer_model.tokenizer
    input_ids = tokenizer.encode(sequence, return_tensors='pt').to(infer_model.model.device)
    # print(input_ids)
    with torch.no_grad():
        logits = infer_model.model(input_ids, labels=input_ids).logits
    
    # print(logits.shape)
    # print(input_ids.shape)
    logits = logits[:, :-1, :].contiguous()
    input_ids = input_ids[:, 1:].contiguous()
    cn_loss = torch.nn.CrossEntropyLoss(reduction='none')
    log_probs = cn_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1))

    # print(log_probs)
    
    return log_probs





            
