#%%
import os
import sys
import json
import logging
import pandas as pd
from datasets import load_dataset
from ffm import FFMLLama2
#%%
# Global
current_dir = os.path.dirname(os.path.abspath(__file__))

#%%
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
file_handler = logging.FileHandler('mainlog.txt', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
#%%
def exp_setting():

    categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
        ],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": [
        "politics",
        "culture",
        "economics",
        "geography",
        "psychology",
        "education",
        ],
    "other (business, health, misc.)": ["other", "business", "health"]
    }

    task_list = [
        "engineering_math",
        "dentistry",
        "traditional_chinese_medicine_clinical_medicine",
        "clinical_psychology",
        "technical",
        "culinary_skills",
        "mechanical",
        "logic_reasoning",
        "real_estate",
        "general_principles_of_law",
        "finance_banking",
        "anti_money_laundering",
        "ttqav2",
        "marketing_management",
        "business_management",
        "organic_chemistry",
        "advance_chemistry",
        "physics",
        "secondary_physics",
        "human_behavior",
        "national_protection",
        "jce_humanities",
        "politic_science",
        "agriculture",
        "official_document_management",
        "financial_analysis",
        "pharmacy",
        "educational_psychology",
        "statistics_and_machine_learning",
        "management_accounting",
        "introduction_to_law",
        "computer_science",
        "veterinary_pathology",
        "accounting",
        "fire_science",
        "optometry",
        "insurance_studies",
        "pharmacology",
        "taxation",
        "education_(profession_level)",
        "economics",
        "veterinary_pharmacology",
        "nautical_science",
        "occupational_therapy_for_psychological_disorders",
        "trust_practice",
        "geography_of_taiwan",
        "physical_education",
        "auditing",
        "administrative_law",
        "basic_medical_science",
        "macroeconomics",
        "trade",
        "chinese_language_and_literature",
        "tve_design",
        "junior_science_exam",
        "junior_math_exam",
        "junior_chinese_exam",
        "junior_social_studies",
        "tve_mathematics",
        "tve_chinese_language",
        "tve_natural_sciences",
        "junior_chemistry",
        "music",
        "education",
        "three_principles_of_people",
        "taiwanese_hokkien"]
    subject2name = {}
    subject2category = {}

    df = pd.read_csv(os.path.join(current_dir, "subject.tsv"), delimiter="\t")
    for _, row in df.iterrows():
        if row["subject"] in task_list:
            subject2category[row["subject"]] = row["category"]
            subject2name[row["subject"]] = row["name"]
    return task_list, subject2name, subject2category
#%%

def ieval():

    evaluator = FFMLLama2()
    data_path = r'D:\ellom\working\Llm_eval\tmmluplus\data'
    model_name = "ffm-llama2-70b-chat-exp"
    model_name_path = model_name.replace("-", "_")
    Save_result_dir = r'D:\ellom\working\Llm_eval\tmmluplus\eval_result'

    task_list, subject2name, subject2category = exp_setting()

    postfix_name = "-".join(model_name.split("-")[0:2])
    prefix_name = "tmmluplus"
    result_cache = f"{prefix_name}_{postfix_name}.tsv"

    if os.path.exists(result_cache):
        logging.info(f'Found previous cache {result_cache}, skipping executed subjects.')
        df = pd.read_csv(result_cache, delimiter="\t", header=None)
        df.columns = ["model_name", "subject", "score"]
        finished_subjects = df["subject"].to_list()
        task_list = [t for t in task_list if t not in finished_subjects]

    output_filename = ""

    for task in task_list:
        zh_name = subject2name[task]
        test = load_dataset(data_path, task)["test"]
        test_df = pd.DataFrame([dict(row) for row in test])
        
        accuracy = evaluator.conversation(
            subject_name=zh_name,
            test_df=test_df,
            save_result_dir=Save_result_dir
        )

        with open(result_cache, "a") as fout:
            fout.write("{}\t{}\t{:.5f}\n".format(model_name, task, accuracy))

    df = pd.read_csv(result_cache, delimiter="\t", header=None)
    df.columns = ["model_name", "subject", "score"]
    for model_name in df["model_name"].unique():
        print(model_name)
#%%
def main() -> None:
    ieval

if __name__ == "__main__":
    main()
#%%