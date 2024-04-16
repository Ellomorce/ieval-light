#%%
import os
import sys
import json
import logging
import pandas as pd
from datasets import load_dataset
from models import FFMLLama2
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
def exp_setting(chunk_no:str):

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

    chunks = {
        "0": ["jce_humanities.csv", "real_estate.csv"],
        "1": ["politic_science.csv", "tve_natural_sciences.csv"],
        "2": ["basic_medical_science.csv", "education_(profession_level).csv"],
        "3": ["optometry.csv", "trade.csv"], 
        "4": ["geography_of_taiwan.csv", "marketing_management.csv"],
        "5": ["insurance_studies.csv", "nautical_science.csv", "taiwanese_hokkien.csv"],
        "6": ["auditing.csv", "human_behavior.csv", "occupational_therapy_for_psychological_disorders.csv"],
        "7": ["macroeconomics.csv", "tve_chinese_language.csv", "veterinary_pharmacology.csv"],
        "8": ["administrative_law.csv", "finance_banking.csv", "technical.csv", "tve_design.csv"],
        "9": ["dentistry.csv", "economics.csv", "introduction_to_law.csv", "trust_practice.csv"],
        "10": ["culinary_skills.csv", "financial_analysis.csv", "pharmacy.csv", "taxation.csv"],
        "11": ["agriculture.csv", "music.csv", "official_document_management.csv", "statistics_and_machine_learning.csv",
               "traditional_chinese_medicine_clinical_medicine.csv", "veterinary_pathology.csv"],
        "12": ["accounting.csv", "chinese_language_and_literature.csv", "junior_chemistry.csv", "junior_science_exam.csv",
               "management_accounting.csv", "national_protection.csv", "physical_education.csv"],
        "13": ["anti_money_laundering.csv", "business_management.csv", "computer_science.csv", "educational_psychology.csv",
               "junior_chinese_exam.csv", "junior_math_exam.csv", "logic_reasoning.csv", "three_principles_of_people.csv",
               "tve_mathematics.csv"],
        "14": ["advance_chemistry.csv", "clinical_psychology.csv", "education.csv", "engineering_math.csv", "fire_science.csv",
               "general_principles_of_law.csv", "junior_social_studies.csv", "linear_algebra.csv", "mechanical.csv",
               "organic_chemistry.csv", "physics.csv", "secondary_physics.csv", "ttqav2.csv"]
    }
    task_list = chunks[chunk_no]
    task_name_list = [task.replace(".csv", "") for task in task_list]
    subject2name = {}
    subject2category = {}

    df = pd.read_csv(os.path.join(current_dir, "subject.tsv"), delimiter="\t")
    for _, row in df.iterrows():
        if row["subject"] in task_name_list:
            subject2category[row["subject"]] = row["category"]
            subject2name[row["subject"]] = row["name"]
    return task_list, subject2name, subject2category
#%%

def ieval(chunk_no:str):

    evaluator = FFMLLama2()
    data_path = r'D:\ellom\working\Llm_eval\tmmluplus\data_test'
    model_name = "ffm-llama2-70b-chat-exp"
    model_name_path = model_name.replace("-", "_")
    Save_result_dir = r'D:\ellom\working\Llm_eval\tmmluplus\eval_result'

    task_list, subject2name, subject2category = exp_setting(chunk_no=chunk_no)

    postfix_name = "-".join(model_name.split("-")[0:2])
    prefix_name = "tmmluplus"
    result_cache = f"{prefix_name}_{postfix_name}.tsv"

    if os.path.exists(result_cache):
        logging.info(f'Found previous cache {result_cache}, skipping executed subjects.')
        df = pd.read_csv(result_cache, delimiter="\t", header=None)
        df.columns = ["model_name", "subject", "score"]
        finished_subjects = df["subject"].to_list()
        task_list = [t for t in task_list if t not in finished_subjects]

    # output_filename = ""

    for task in task_list:
        task_name = task.replace(".csv", "")
        zh_name = subject2name[task_name]
        # test = load_dataset(data_path, task)["test"]
        # test_df = pd.DataFrame([dict(row) for row in test])
        test = load_dataset(path=data_path, name='csv', data_files={'test':task})
        test_df = test["test"].to_pandas()
        
        accuracy, res_df = evaluator.conversation(
            subject_name=zh_name,
            test_df=test_df,
            save_result_dir=Save_result_dir)

        with open(result_cache, "a") as fout:
            fout.write("{}\t{}\t{:.5f}\n".format(model_name, task, accuracy))

        res_df.to_csv(
            os.path.join(Save_result_dir, f"chunk{chunk_no}_val.csv"),
            encoding='utf-8',
            index = False)

    df = pd.read_csv(result_cache, delimiter="\t", header=None)
    df.columns = ["model_name", "subject", "score"]
    for model_name in df["model_name"].unique():
        print(model_name)
#%%
def main() -> None:
    chunk_no = "4"
    ieval(chunk_no)

if __name__ == "__main__":
    main()
#%%