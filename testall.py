import argparse
import os
from pathlib import Path

import gdown
import pytest
from torch_geometric.data import extract_zip

from GOOD import config_summoner, args_parser
from GOOD.definitions import ROOT_DIR, STORAGE_DIR
from GOOD.kernel.pipeline import initialize_model_dataset, load_ood_alg, load_logger, config_model
from GOOD.kernel.evaluation import evaluate

parser = argparse.ArgumentParser(description='GOOD')
parser.add_argument('--alg', type=str,default="ciga")
parser.add_argument('--usr', action="store_true")  # if true, use user priviledge to run the project
parser.add_argument('--exp_round', type=int,default=1)  
parser.add_argument('--dataset', type=str,default="")  
args = parser.parse_args()


config_paths = []
config_root = Path(ROOT_DIR, 'configs', 'GOOD_configs')
for dataset_path in config_root.iterdir():
    if not dataset_path.is_dir():
        continue
    for domain_path in dataset_path.iterdir():
        if not domain_path.is_dir():
            continue
        for shift_path in domain_path.iterdir():
            if not shift_path.is_dir():
                continue
            for ood_config_path in shift_path.iterdir():
                if 'base' in ood_config_path.name or args.alg.lower() not in ood_config_path.name.lower():
                    continue
                if len(args.dataset)>0 and args.dataset.lower() not in ood_config_path.lower():
                    continue
                config_paths.append(str(ood_config_path))
                print(f"testing with",ood_config_path)
                if args.usr:
                    cmd_str = f"goodtg --config_path {ood_config_path} --exp_round {args.exp_round}"
                else:
                    cmd_str = f"sudo goodtg --config_path {ood_config_path} --exp_round {args.exp_round}"
                print(cmd_str)
                os.system(cmd_str)
