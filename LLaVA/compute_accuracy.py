import pandas as pd
import numpy as np
import os


word_mapping_dict={
                "dishcloth":["dishrag"],
                "weight (exercise)": ["dumbbell", "weight"],
                "nail (fastener)":["nail"],
                "tv": ["television"],
                "iron (for clothes)" :["iron"],
                "t shirt" :["t-shirt", 'shirt'],
                "laptop (open)" :["laptop" ],
                "dress shoe (men)" :["dress shoe"],
                "binder (closed)" :["binder"],
                "portable heater" :["heater" ],
                "computer mouse" :["mouse"],
                "still camera" :["camera"],
                "lampshade" :["lamp shade"],
                "trash bin" :["trash can"],
                "winter glove" :["glove" ],
                "band aid" :["band-aid", "bandaid"],
                "toilet paper roll" :["roll of toilet paper"],
                "pill bottle" :["pill"],
                "soup bowl" :["bowl of soup"],
                "water bottle" :["bottle of water"],
                "pop can" :["can of"],
                "weight scale" :["scale"],
                "sandal" :["flip flop"],
                "swimming trunks" :["swimming trunk"],
                "bread loaf" :["bread"],
                "coffee or french press" :["coffee press" , "french press" , "coffee machine", "coffee maker"],
                "mixing or salad bowl" :["mixing bowl" , "salad bowl"],
                "alarm clock": ["clock"],
                "wheel": ["bicycle wheel"],
                "ladle": ["spoon"],
                "matchstick": ["match"],
                "cellphone": ["cell phone", 'mobile phone']
        }

def test_word_mapping(word, answer, word_mapping_dict):
    if word in answer:
        return True
    if word in word_mapping_dict.keys():
        for map_word in word_mapping_dict[word]:
            if map_word in answer:
                return True
    return False


orginal_txt_root = '/media/philipp/ssd1_2tb/zcs_projects/homework/imagenet_d/LLaVA/results_acceptance/'
for root, _, txt_list in os.walk(orginal_txt_root):
    for csv_name in txt_list:
        data = []
        if not csv_name.endswith('LLava_predictions.csv'):
            continue 

        df_ori = pd.read_csv(f"{root}/{csv_name}", sep='\t')

        for index, row in df_ori.iterrows():

            gt_category = row['gt_category']
            pred_category = row['pred_category']
            question = row['question']
            answer = row['answer']
            image_path = row['image_dir']

            answer_test = answer.lower()

            test_gt = test_word_mapping(gt_category, answer_test, word_mapping_dict)
            test_pred = test_word_mapping(pred_category, answer_test, word_mapping_dict) 

            if gt_category =="pen" and pred_category == "can opener" or gt_category == "plate" and pred_category == "drying rack for plates":
                if test_pred:
                    correct_flag = "LLaVa FALSE"
                elif test_gt:
                    correct_flag = "LLaVa TRUE"
                else:
                    correct_flag = "LLaVa TODO"
            elif gt_category =="can opener" and pred_category == "pen" or gt_category == "drying rack for plates" and pred_category == "plate":
                if test_gt:
                    correct_flag = "LLaVa TRUE"
                elif test_pred:
                    correct_flag = "LLaVa FALSE" 
                else:
                    correct_flag = "LLaVa TODO"
            elif test_gt and test_pred:
                correct_flag = "LLaVa DOUBLE" 
            elif test_gt:
                    correct_flag = "LLaVa TRUE"
            elif test_pred:
                    correct_flag = "LLaVa FALSE"
            else:
                correct_flag = "LLaVa TODO" 

            data.append([correct_flag, gt_category, pred_category, answer,  question, image_path])

        df = pd.DataFrame(data, columns=["LLaVa_correct", "gt_category", "pred_category", "answer",  "question",  "image_dir"])

        save_csv_path = os.path.join(root, csv_name.replace(".csv", "_formated.csv"))

        df.to_csv(save_csv_path, index=False, sep="\t")

        true_num = df['LLaVa_correct'].str.contains('TRUE', case=False).sum()
        false_num = df['LLaVa_correct'].str.contains('FALSE', case=False).sum()
        acc = true_num / (true_num + false_num)

        print(f"Dataset: {root.split('/')[-2]} model: {root.split('/')[-1]} True: {true_num}  False: {false_num}  Acc: {acc * 100} ")