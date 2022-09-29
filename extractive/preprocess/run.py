import os

# raw_path = "/remote-home/cxan/summary/Datasets/NSum/"
# datasets = ["reddit", "XSum", "SSN", "pubMed"]
# for dataset in datasets:
#     os.system(f"python preprocess_to_Bart.py --dataset {dataset} --raw_path {raw_path}{dataset}")
raw_path = "/remote-home/cxan/summary/Datasets/NSum/"
os.system(f"python preprocess_to_Bart.py --dataset pubMed_long --raw_path {raw_path}/pubMed")