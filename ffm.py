#%%
import os
import sys
import json
import re
import logging
import string
import requests
import pandas as pd
import tqdm
import opencc
#%%
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
file_handler = logging.FileHandler('modellog.txt', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
#%%
class FFMLLama2:
    def __init__(self, switch_zh_hans=False) -> None:
        self.api_key="98430d42-c60c-4612-adbb-eeb927c5c2be"
        self.model_name="ffm-llama2-70b-chat-exp"
        self.api_url="https://53955.afs.twcc.ai/text-generation"
        self.max_new_tokens=350
        self.temperature=0.5
        self.top_k=50
        self.top_p=1.0
        self.frequency_penalty=1.0
        self.choices=["A", "B", "C", "D"]
        self.converter=None
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")
        self.puncs=list(string.punctuation)

    def format_example(self, line):
        example=line["question"]
        for choice in self.choices:
            example+= f'\n{choice}.{line[f"{choice}"]}'
        
        example += "\n答案:"
        return [
            {"role":"human", "content": example},
        ]
    
    def extract_ans(self, response_str):
        pattern = [
            r"([A-D]). ",
            r"([A-D]).",
            r"^選([A-D])",
            r"^選項([A-D])",
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?選?項?\s?([A-D])",
            r"答案為\s?選?項?\s?([A-D])",
            r"答案應為\s?選?項?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案選\s?選?項?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?選?項?\s?([A-D])",
            r"答案應該是:\s?選?項?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正確的一項是\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案為:\s?選?項?\s?([A-D])",
            r"答案應為:\s?選?項?\s?([A-D])",
            r"答案:\s?選?項?\s?([A-D])",
            r"答案是：\s?選?項?\s?([A-D])",
            r"答案應該是：\s?選?項?\s?([A-D])",
            r"答案為：\s?選?項?\s?([A-D])",
            r"答案應為：\s?選?項?\s?([A-D])",
            r"答案：\s?選?項?\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
            r"正確答案是：\s?選?項?\s?([A-D])",
        ]
        ans_list = []
        if response_str[0] in ["A", "B", "C", "D"]:
            ans_list.append(response_str[0])
        for p in pattern:
            if self.converter:
                p = self.converter.convert(p)
            if len(ans_list) == 0:
                ans_list = re.findall(p, response_str)
            else:
                break
        return ans_list
    
    def normalize_answer(self, s):
        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(self.puncs)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self, pred, target):
        return self.normalize_answer(pred) == self.normalize_answer(target)
    
    def conversation(
            self,
            subject_name,
            test_df,
            save_result_dir=None
            ):
        headers = {
            "content-type": "application/json",
            "X-API-Key": self.api_key
        }
        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        
        system = [
            {
                "role":"system",
                "content": f"你是一位專業的中文AI助理，以下是關於{subject_name}考試單選題，請選出正確的答案。",
            }
        ]
        answers = list(test_df["answer"])
        for row_index, row in tqdm.tqdm(
            test_df.iterrows(), total=len(test_df), dynamic_ncols=True
        ):
            question = self.format_example(row)
            full_prompt = system + question
            if self.converter:
                converted = []
                for p in full_prompt:
                    p['content'] = self.converter.convert(p["content"])
                    converted.append(p)
                full_prompt = converted
            
            data = {
                "model":self.model_name,
                "message":full_prompt,
                "parameters": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty
                }
            }
            res = ""
            try:
                response = requests.post(self.api_url + "/api/models/conversation", json=data, headers=headers, verify=False)
                res = json.loads(response.text, strict=False)['generated_text']
                with open('answer_record.json', 'a', encoding='utf-8-sig') as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
            except Exception as er:
                print(response.status_code)
            if res != None:
                # response_str = res.choices[0].message.content
                response_str = res
                response_str = response_str.strip()
                if len(response_str) >0:
                    ans_list = self.extract_ans(response_str)
                    # logger.info(f'debug')
                    if len(ans_list) >0 and (ans_list[-1] == row['answer']):
                        # logger.info(f'debug')
                        correct_num += 1
                        correct = 1
                    else:
                        correct = 0
                else:
                    correct = 0
                if save_result_dir:
                    result.append(response_str)
                    score.append(correct)
            else:
                response_str = ""
        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(
                os.path.join(save_result_dir, f"{subject_name}_val.csv"),
                encoding="utf-8",
                index=False,
            )
        return correct_ratio
#%%