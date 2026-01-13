import openai
import anthropic
from anthropic import Anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as genai
import urllib3
from copy import deepcopy

import sys
import os
tap_eval_path = os.environ.get("TAP_EVAL_PATH")
sys.path.append(tap_eval_path)
from config.ModelConfig import LLAMA_API_LINK, VICUNA_API_LINK

# using the new package for Gemini
from google import genai
from google.genai import types

    
class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
        
class HuggingFace(LanguageModel):
    def __init__(self,model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()} 

        
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1, # To prevent warning messages
            )
            
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])


class APIModel(LanguageModel): 

    API_HOST_LINK = "ADD_LINK"
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    
    API_TIMEOUT = 100
    
    MODEL_API_KEY = os.getenv("MODEL_API_KEY")
    
    API_HOST_LINK = ''

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        ''' 
            
        output = self.API_ERROR_OUTPUT 
        
        for _ in range(self.API_MAX_RETRY):  
            try:
                
                # Batch generation
                if temperature > 0:
                    # Attack model
                    json = {
                        "top_p": top_p, 
                        "num_beams": 1, 
                        "temperature": temperature, 
                        "do_sample": True,
                        "prompt": '', 
                        "max_new_tokens": max_n_tokens,
                        "system_prompt": conv,
                    } 
                else:
                    # Target model
                    json = {
                        "top_p": 1,
                        "num_beams": 1, 
                        "temperature": 1, # To prevent warning messages
                        "do_sample": False,
                        "prompt": '', 
                        "max_new_tokens": max_n_tokens,
                        "system_prompt": conv,
                    }  

                    # Do not use extra end-of-string tokens in target mode
                    if 'llama' in self.model_name: 
                        json['extra_eos_tokens'] = 0 
                        
    
                if 'llama' in self.model_name:
                    # No system prompt for the Llama model
                    assert json['prompt'] == ''
                    json['prompt'] = deepcopy(json['system_prompt'])
                    del json['system_prompt'] 
                
                resp = urllib3.request(
                            "POST",
                            self.API_HOST_LINK,
                            headers={"Authorization": f"Api-Key {self.MODEL_API_KEY}"},
                            timeout=urllib3.Timeout(self.API_TIMEOUT),
                            json=json,
                )

                resp_json = resp.json()

                if 'vicuna' in self.model_name:
                    if 'error' in resp_json:
                        print(self.API_ERROR_OUTPUT)
    
                    output = resp_json['output']
                    
                else:
                    output = resp_json
                    
                if type(output) == type([]):
                    output = output[0] 
                
                break
            except Exception as e:
                print('exception!', type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class APIModelLlama7B(APIModel): 
    API_HOST_LINK = LLAMA_API_LINK
    MODEL_API_KEY = os.getenv("LLAMA_API_KEY")

class APIModelVicuna13B(APIModel): 
    API_HOST_LINK = VICUNA_API_LINK 
    MODEL_API_KEY = os.getenv("VICUNA_API_KEY")

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 20
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try: 
                
                response = openai.ChatCompletion.create(
                            model = self.model_name,
                            messages = conv,
                            max_tokens = max_n_tokens,
                            temperature = temperature,
                            top_p = top_p,
                            request_timeout = self.API_TIMEOUT,
                            )
                output = response["choices"][0]["message"]["content"]

                # # debug
                if self.model_name == "glm-4.6":
                    print(conv)
                    print(output)
                    sys.exit()

                break
            except Exception as e: 
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
     
class PaLM():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")
    

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        genai.configure(api_key=self.API_KEY) 

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):            
            try:
                completion = genai.chat(
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.last
                
                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[:(max_n_tokens*4)] 
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


# class GeminiPro():
#     API_RETRY_SLEEP = 10
#     API_ERROR_OUTPUT = "$ERROR$"
#     API_QUERY_SLEEP = 1
#     API_MAX_RETRY = 5
#     API_TIMEOUT = 20
#     default_output = "I'm sorry, but I cannot assist with that request."
#     API_KEY = os.getenv("PALM_API_KEY")

#     def __init__(self, model_name) -> None:
#         self.model_name = model_name
#         genai.configure(api_key=self.API_KEY) 

#     def generate(self, conv: List, 
#                 max_n_tokens: int, 
#                 temperature: float,
#                 top_p: float):
#         '''
#         Args:
#             conv: List of dictionaries, 
#             max_n_tokens: int, max number of tokens to generate
#             temperature: float, temperature for sampling
#             top_p: float, top p for sampling
#         Returns:
#             str: generated response
#         '''
#         output = self.API_ERROR_OUTPUT
#         for _ in range(self.API_MAX_RETRY):            
#             try:
#                 model = genai.GenerativeModel(self.model_name)
#                 output = model.generate_content(
#                     contents = conv,
#                     generation_config = genai.GenerationConfig(
#                         candidate_count = 1,
#                         temperature = temperature,
#                         top_p = top_p,
#                         max_output_tokens=max_n_tokens,
#                     )
#                 )

#                 if output is None:
#                     # If PaLM refuses to output and returns None, we replace it with a default output
#                     output = self.default_output
#                 else:
#                     # Use this approximation since PaLM does not allow
#                     # to specify max_tokens. Each token is approximately 4 characters.
#                     output = output.text
#                 break
#             except Exception as e:
#                 print(type(e), e)
#                 time.sleep(self.API_RETRY_SLEEP)
        
#             time.sleep(self.API_QUERY_SLEEP)
#         return output
    
#     def batched_generate(self, 
#                         convs_list: List[List[Dict]],
#                         max_n_tokens: int, 
#                         temperature: float,
#                         top_p: float = 1.0,):
#         return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

# the latest Gemini
class Gemini():
    def __init__(self, model_name: str, system_prompt="You are a helpful assistant."):
        self.API_RETRY_SLEEP = 10
        self.API_ERROR_OUTPUT = "$ERROR$"
        self.API_QUERY_SLEEP = 1
        self.API_MAX_RETRY = 5
        self.API_TIMEOUT = 20
        self.default_output = "I'm sorry, but I cannot assist with that request."
        
        self.API_KEY = os.getenv("GEMINI_API_KEY")
        self.API_BASE = os.getenv("GEMINI_API_BASE")

        self.client = genai.Client(api_key=self.API_KEY,
                        http_options={"base_url": self.API_BASE})
        
        self.model_name = model_name
        self.system_prompt = system_prompt

    def generate(self, full_prompt: str, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            full_prompt: str, the full prompt to generate a response 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try: 

                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction = self.system_prompt,
                        max_output_tokens = max_n_tokens,
                        temperature = temperature,
                        thinking_config = types.ThinkingConfig(thinking_budget=0) # Disables thinking
                    ),
                    contents = full_prompt
                )

                output = response.text

                # # debug
                # print("input:")
                # print(full_prompt)
                # print("output:")
                # print(output)

                # sys.exit()

                break
            except Exception as e: 
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 

    def batched_generate(self, 
                            prompt_list: List[str],
                            max_n_tokens: int, 
                            temperature: float,
                            top_p: float = 1.0,):
            return [self.generate(prompt, max_n_tokens, temperature, top_p) for prompt in prompt_list]




class Claude(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 20

    api_key = os.getenv("CLAUDE_API_KEY")
    api_base = os.getenv("CLAUDE_API_BASE")

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Anthropic(
                api_key=self.api_key,  # Replace with your API key
                base_url=self.api_base
                )
    
    def generate(self, conv, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: Dict
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try: 
                print("Claude input test:")
               
                # no system information should be included in message
                system_message = conv[0]['content']
                conv = conv[1:]
                print(conv)
                # sys.exit()
                # disable the thinking mode
                response = self.client.messages.create(
                            model=self.model_name,
                            messages=conv,
                            max_tokens=max_n_tokens,
                            temperature=temperature,
                            # top_p=top_p,
                            timeout=self.API_TIMEOUT,
                            
                            system=system_message,
                            thinking={
                                        "type": "disabled"
                                    },
                            )
                output = response.content[0].text
                print("Claude output test:")
                print(output)

                sys.exit()
                
                break
            except Exception as e: 
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
     