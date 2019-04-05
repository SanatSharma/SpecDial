# Main file for extracting Visdial features and predicting specificity

import argparse
import collections
import logging
import json
import itertools
from tensorboardX import SummaryWriter
import torch
import yaml

from data.dataset import VisDialDataset
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml", default="configs/lf_disc_faster_rcnn_x101.yml",
    help="Path to a config file listing reader, model and solver parameters."
)
parser.add_argument(
    "--train-json", default="data/visdial_1.0_train.json",
    help="Path to json file containing VisDial v1.0 training data."
)
parser.add_argument(
    "--val-json", default="data/visdial_1.0_val.json",
    help="Path to json file containing VisDial v1.0 validation data."
)
parser.add_argument(
    "--val-dense-json", default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to json file containing VisDial v1.0 validation dense ground truth annotations."
)


parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=0,
    help="List of ids of GPUs to use."
)
parser.add_argument(
    "--cpu-workers", type=int, default=4,
    help="Number of CPU workers for dataloader."
)
parser.add_argument(
    "--overfit", action="store_true",
    help="Overfit model on 5 examples, meant for debugging."
)
parser.add_argument(
    "--validate", action="store_true",
    help="Whether to validate on val split after every epoch."
)
parser.add_argument(
    "--in-memory", action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. Use only in "
         "presence of large RAM, atleast few tens of GBs."
)


parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath", default="checkpoints/",
    help="Path of directory to create checkpoint directory and save checkpoints."
)
parser.add_argument(
    "--load-pthpath", default="",
    help="To continue training, path to .pth file of saved checkpoint."
)

# Retrieve example information to feed into BERT
def extract_visdial_features(config):
    train_dataset = VisDialDataset(
    config["dataset"], args.train_json, overfit=args.overfit, in_memory=args.in_memory
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["solver"]["batch_size"], num_workers=args.cpu_workers, shuffle=True
    )

    val_dataset = VisDialDataset(
        config["dataset"], args.val_json, args.val_dense_json, overfit=args.overfit, in_memory=args.in_memory
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["solver"]["batch_size"], num_workers=args.cpu_workers
    )


if __name__ == "__main__":
    args = parser.parse_args()

    # keys: {"dataset", "model", "solver"}
    config = yaml.load(open(args.config_yml))
    print(yaml.dump(config, default_flow_style=False))
    extract_visdial_features(config)