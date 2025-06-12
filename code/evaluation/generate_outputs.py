'''
The evaluation data is from the supplemetary material of the safe-RLHF paper: https://openreview.net/forum?id=TyFrPOKYXw
The template is for the Alpaca model in the safe-rlhf paper. Here we assume all the models (base and reward model) use the same template.
'''

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
import os
from collections import OrderedDict
import torch


# from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_arithmetic import ModelArithmetic, PromptedLLM
from peft.utils.save_and_load import get_peft_model_state_dict

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT_ALPACA: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generation of prompts using LLMs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_base_name_or_path',
        default="/path/PKU-Alignment--alpaca-7b-reproduced",
        type=str,
        help='the name or path of model to load from',
    )
    model_parser.add_argument(
        '--model_arm_helpfulness_name_or_path',
        default='/path',
        type=str,
        help='the name or path of model to load from',
    )
    model_parser.add_argument(
        '--model_arm_harmlessness_name_or_path',
        default="/path",
        type=str,
        help='the name or path of model to load from',
    )
    model_parser.add_argument(
        '--alpha_helpfulness',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness',
    )
    model_parser.add_argument(
        '--alpha_harmlessness',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness',
    )

    model_parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='The maximum sequence length of the generation.',
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )

    model_parser.add_argument(
        '--normalize_logit',
        type=str2bool,
        default=False,
        help='If True, the temperature argument to ModelArithmetic will be enforced to 1.0, and thus the logit weights of base LLM and ARM sum up to 1; If false, when using ARM decoding, the temperature will be 1/(1+alpha_1+alpha_2).',
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--datasets',
        type=str,
        default="/path/test_prompt_only.json"
    )

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument(
        '--output_dir',
        type=str,
        default="/path/results",
        help='Where to store the evaluation output.',
    )
    logging_parser.add_argument(
        '--resume',
        type=str2bool,
        default=False,
    )

    args = parser.parse_args()
    return args

################ Model Arithmetic ################
def get_model_arithmetic(model_pth_base, model_pth_reward_help, model_pth_reward_harm, model_pth_reward_concise, alpha_help, alpha_harm, alpha_concise, args):
    '''
    Return the decoding policy and the temperature (input to generate) that corresponds to temperature=1 w.r.t. decoding policy

    NOTE
    1. The computation of temperature is only correct when beta = beta_1 = beta_2.
    2. We do not use any system prompt for now. 
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_pth_base)
    if 'alpaca'.lower() in model_pth_base.lower() and '65B'.lower() in model_pth_base.lower():
        print('\nUsing Alpaca-65B as the base model.\n')
        prompt_template_base = lambda system_prompt, input_string: f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input_string}\n\n### Response:\n"
    else:
        prompt_template_base = lambda system_prompt, input_string: PROMPT_INPUT_ALPACA.format(input=input_string) 

    # note: the reward model needs to use the tokenizer of the base model for the following reasons
    # when base = llama (32000 vocab size) and arm = alpaca (might have 32001 tokens), the model_arithmetic will truncate to the tokenizer vocab size (32000) to make it work. 
    M_base = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_base, model=model_pth_base, tokenizer=tokenizer) 
    M_reward_help = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_reward, model=model_pth_reward_help, tokenizer=tokenizer) if alpha_help != 0 else None
    M_reward_harm = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_reward, model=model_pth_reward_harm, tokenizer=tokenizer) if alpha_harm != 0 else None

    formula = M_base
    if M_reward_help is not None:
        formula += alpha_help * M_reward_help
    if M_reward_harm is not None:
        formula += alpha_harm * M_reward_harm
    
    model = ModelArithmetic(formula, max_length = args.max_length) 
    temperature = 0

    return model, tokenizer, temperature

################ Model Arithmetic Ends ################

if __name__ == '__main__':

    args = parse_arguments()

    model_name = f'PARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm'

    ##################### output path #####################
    out_path = Path(os.path.join(args.output_dir, model_name, 'generation') + f".json")
    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    print(f'Generation results will be saved in {out_path}')

    ##################### Load dataset #####################
    with open(args.datasets, 'r') as f:
        data_evaluation = json.load(f)

    ##################### Load models #####################
    model, tokenizer, temperature = get_model_arithmetic(model_pth_base=args.model_base_name_or_path, 
                    model_pth_reward_help=args.model_arm_helpfulness_name_or_path, 
                    model_pth_reward_harm=args.model_arm_harmlessness_name_or_path,
                    alpha_help=args.alpha_helpfulness, alpha_harm=args.alpha_harmlessness, 
                    args=args)
    model.eval()
    if args.normalize_logit:
        print('\nEnforcing temperature=1.0 in the model_arithmetic generation; The logit weights of base models and potential ARMs are normalized.\n')
        temperature = 1.0
    # generate function will remove eos token
    generate = lambda prompt: model.generate_text(prompt, max_new_tokens=args.max_new_tokens, batch_size=None, temperature=temperature, top_p=1, top_k=0, do_speculation=False)[0].removesuffix(tokenizer.eos_token)

    if args.normalize_logit:
        model_name += '_NormalizedLogit'

    print(f'\nModel Name: {model_name}')

    ##################### Generate responses #####################
    output_set = []
    start_index = 0

    start_time_script = time.time()
    for i in tqdm(range(start_index, len(data_evaluation))):
        prompt = data_evaluation[i]['prompt']
        # prompt_input = PROMPT_INPUT.format(input=prompt) # NOTE: the template is important
        prompt_input = prompt # the template is already used when initializing model arithmetic. 

        start = time.time()
        response = generate(prompt_input)
        elapsed = time.time() -start

        output_set.append({
            "uid": data_evaluation[i]['uid'],
            "prompt": prompt, 
            "response": response,
            "model": model_name,
            "elapsed":elapsed,
            })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_set, f, ensure_ascii=False, indent=4)
    print(f'Generating responses finished!\nSaving to {out_path}\nTime:{(time.time()-start_time_script)/ 3600} hours for {len(output_set) - start_index} outputs')
