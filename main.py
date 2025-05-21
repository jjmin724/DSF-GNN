#!/usr/bin/env python3
import argparse
import os
from core.config import cfg
from data.collector import Collector
from data.processor import Preprocessor
from data.graph_builder import GraphBuilder
from tasks.pretrain import Pretrainer
from models.trainer import Trainer
from data.symbol_updater import SymbolUpdater
from core.utils import ensure_dir

def set_cuda():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cfg["gpus"])

def run_update_symbols():
    SymbolUpdater(cfg).run()

def run_collect():
    Collector(cfg).run()

def run_preprocess():
    Preprocessor(cfg).run()
    GraphBuilder(cfg).run()

def run_pretrain():
    Pretrainer(cfg).run()

def run_train():
    Trainer(cfg, mode="train").run()

def run_predict():
    Trainer(cfg, mode="predict").run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",
        choices=["update_symbols", "collect", "preprocess", "pretrain", "train", "predict"])
    args = parser.parse_args()

    ensure_dir(cfg["workspace"])
    set_cuda()

    {
      "update_symbols": run_update_symbols,
      "collect": run_collect,
      "preprocess": run_preprocess,
      "pretrain": run_pretrain,
      "train": run_train,
      "predict": run_predict
    }[args.mode]()
