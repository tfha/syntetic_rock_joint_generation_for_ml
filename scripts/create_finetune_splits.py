"""
Create dataset splits for Wachter et al. finetuning experiments.

This script generates train/test splits following the experimental design:
- BOX experiments: varying ratios of synthetic/real for training
- SLOPE experiments: varying ratios of synthetic/real for training
- Test set: always 100% real data (fixed size)

Usage:
    python scripts/create_finetune_splits.py
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from ml_segmentation.data_loading import get_data_files, get_datasets_prefixes

console = Console()


# Wachter et al. (2026) Experimental design from Table 1
# Simple Mixed (SM) strategy - train on shuffled synthetic+real
SIMPLEMIXED_EXPERIMENTS_BOX = [
    {
        "name": "simplemixed_box_0",
        "percent_real": 0,
        "train_synthetic": 180,
        "train_real": 0,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_10",
        "percent_real": 10,
        "train_synthetic": 162,
        "train_real": 18,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_30",
        "percent_real": 30,
        "train_synthetic": 126,
        "train_real": 54,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_50",
        "percent_real": 50,
        "train_synthetic": 90,
        "train_real": 90,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_70",
        "percent_real": 70,
        "train_synthetic": 54,
        "train_real": 126,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_90",
        "percent_real": 90,
        "train_synthetic": 18,
        "train_real": 162,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 180,
        "test_real": 20,
    },
]

SIMPLEMIXED_EXPERIMENTS_SLOPE = [
    {
        "name": "simplemixed_slope_0",
        "percent_real": 0,
        "train_synthetic": 2700,
        "train_real": 0,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_10",
        "percent_real": 10,
        "train_synthetic": 2430,
        "train_real": 270,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_30",
        "percent_real": 30,
        "train_synthetic": 1890,
        "train_real": 810,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_50",
        "percent_real": 50,
        "train_synthetic": 1350,
        "train_real": 1350,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_70",
        "percent_real": 70,
        "train_synthetic": 810,
        "train_real": 1890,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_90",
        "percent_real": 90,
        "train_synthetic": 270,
        "train_real": 2430,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 2700,
        "test_real": 300,
    },
]

# Fine-Tuned (FT) strategy - pretrain on synthetic, finetune on real
FINETUNE_EXPERIMENTS_BOX = [
    {
        "name": "finetune_box_0",
        "percent_real": 0,
        "train_synthetic": 180,
        "train_real": 0,
        "test_real": 20,
    },
    {
        "name": "finetune_box_10",
        "percent_real": 10,
        "train_synthetic": 162,
        "train_real": 18,
        "test_real": 20,
    },
    {
        "name": "finetune_box_30",
        "percent_real": 30,
        "train_synthetic": 126,
        "train_real": 54,
        "test_real": 20,
    },
    {
        "name": "finetune_box_50",
        "percent_real": 50,
        "train_synthetic": 90,
        "train_real": 90,
        "test_real": 20,
    },
    {
        "name": "finetune_box_70",
        "percent_real": 70,
        "train_synthetic": 54,
        "train_real": 126,
        "test_real": 20,
    },
    {
        "name": "finetune_box_90",
        "percent_real": 90,
        "train_synthetic": 18,
        "train_real": 162,
        "test_real": 20,
    },
    {
        "name": "finetune_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 180,
        "test_real": 20,
    },
]

FINETUNE_EXPERIMENTS_SLOPE = [
    {
        "name": "finetune_slope_0",
        "percent_real": 0,
        "train_synthetic": 2700,
        "train_real": 0,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_10",
        "percent_real": 10,
        "train_synthetic": 2430,
        "train_real": 270,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_30",
        "percent_real": 30,
        "train_synthetic": 1890,
        "train_real": 810,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_50",
        "percent_real": 50,
        "train_synthetic": 1350,
        "train_real": 1350,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_70",
        "percent_real": 70,
        "train_synthetic": 810,
        "train_real": 1890,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_90",
        "percent_real": 90,
        "train_synthetic": 270,
        "train_real": 2430,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 2700,
        "test_real": 300,
    },
]

# Generalisation Pattern BOX experiments - train on cardboard, test on pattern
FINETUNE_EXPERIMENTS_GEN_PATTERN_BOX = [
    {
        "name": "finetune_generalisation_pattern_box_0",
        "percent_real": 0,
        "train_synthetic": 90,
        "train_real": 0,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_pattern_box_10",
        "percent_real": 10,
        "train_synthetic": 81,
        "train_real": 9,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_pattern_box_30",
        "percent_real": 30,
        "train_synthetic": 63,
        "train_real": 27,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_pattern_box_50",
        "percent_real": 50,
        "train_synthetic": 45,
        "train_real": 45,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_pattern_box_70",
        "percent_real": 70,
        "train_synthetic": 27,
        "train_real": 63,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_pattern_box_90",
        "percent_real": 90,
        "train_synthetic": 9,
        "train_real": 81,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_pattern_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 90,
        "test_real": 10,
    },
]

SIMPLEMIXED_EXPERIMENTS_GEN_PATTERN_BOX = [
    {
        "name": "simplemixed_generalisation_pattern_box_0",
        "percent_real": 0,
        "train_synthetic": 90,
        "train_real": 0,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_pattern_box_10",
        "percent_real": 10,
        "train_synthetic": 81,
        "train_real": 9,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_pattern_box_30",
        "percent_real": 30,
        "train_synthetic": 63,
        "train_real": 27,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_pattern_box_50",
        "percent_real": 50,
        "train_synthetic": 45,
        "train_real": 45,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_pattern_box_70",
        "percent_real": 70,
        "train_synthetic": 27,
        "train_real": 63,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_pattern_box_90",
        "percent_real": 90,
        "train_synthetic": 9,
        "train_real": 81,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_pattern_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 90,
        "test_real": 10,
    },
]

# Pattern BOX experiments - Stage 2 and Test both use pattern images (separate splits)
FINETUNE_EXPERIMENTS_PATTERN_BOX = [
    {
        "name": "finetune_pattern_box_0",
        "percent_real": 0,
        "train_synthetic": 90,
        "train_real": 0,
        "test_real": 10,
    },
    {
        "name": "finetune_pattern_box_10",
        "percent_real": 10,
        "train_synthetic": 81,
        "train_real": 9,
        "test_real": 10,
    },
    {
        "name": "finetune_pattern_box_30",
        "percent_real": 30,
        "train_synthetic": 63,
        "train_real": 27,
        "test_real": 10,
    },
    {
        "name": "finetune_pattern_box_50",
        "percent_real": 50,
        "train_synthetic": 45,
        "train_real": 45,
        "test_real": 10,
    },
    {
        "name": "finetune_pattern_box_70",
        "percent_real": 70,
        "train_synthetic": 27,
        "train_real": 63,
        "test_real": 10,
    },
    {
        "name": "finetune_pattern_box_90",
        "percent_real": 90,
        "train_synthetic": 9,
        "train_real": 81,
        "test_real": 10,
    },
    {
        "name": "finetune_pattern_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 90,
        "test_real": 10,
    },
]

SIMPLEMIXED_EXPERIMENTS_PATTERN_BOX = [
    {
        "name": "simplemixed_pattern_box_0",
        "percent_real": 0,
        "train_synthetic": 90,
        "train_real": 0,
        "test_real": 10,
    },
    {
        "name": "simplemixed_pattern_box_10",
        "percent_real": 10,
        "train_synthetic": 81,
        "train_real": 9,
        "test_real": 10,
    },
    {
        "name": "simplemixed_pattern_box_30",
        "percent_real": 30,
        "train_synthetic": 63,
        "train_real": 27,
        "test_real": 10,
    },
    {
        "name": "simplemixed_pattern_box_50",
        "percent_real": 50,
        "train_synthetic": 45,
        "train_real": 45,
        "test_real": 10,
    },
    {
        "name": "simplemixed_pattern_box_70",
        "percent_real": 70,
        "train_synthetic": 27,
        "train_real": 63,
        "test_real": 10,
    },
    {
        "name": "simplemixed_pattern_box_90",
        "percent_real": 90,
        "train_synthetic": 9,
        "train_real": 81,
        "test_real": 10,
    },
    {
        "name": "simplemixed_pattern_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 90,
        "test_real": 10,
    },
]

# Generalisation Cardboard BOX experiments - Stage 2 uses cardbox, Test uses cardboard (same source)
FINETUNE_EXPERIMENTS_GEN_CARDBOARD_BOX = [
    {
        "name": "finetune_generalisation_cardboard_box_0",
        "percent_real": 0,
        "train_synthetic": 90,
        "train_real": 0,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_cardboard_box_10",
        "percent_real": 10,
        "train_synthetic": 81,
        "train_real": 9,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_cardboard_box_30",
        "percent_real": 30,
        "train_synthetic": 63,
        "train_real": 27,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_cardboard_box_50",
        "percent_real": 50,
        "train_synthetic": 45,
        "train_real": 45,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_cardboard_box_70",
        "percent_real": 70,
        "train_synthetic": 27,
        "train_real": 63,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_cardboard_box_90",
        "percent_real": 90,
        "train_synthetic": 9,
        "train_real": 81,
        "test_real": 10,
    },
    {
        "name": "finetune_generalisation_cardboard_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 90,
        "test_real": 10,
    },
]

SIMPLEMIXED_EXPERIMENTS_GEN_CARDBOARD_BOX = [
    {
        "name": "simplemixed_generalisation_cardboard_box_0",
        "percent_real": 0,
        "train_synthetic": 90,
        "train_real": 0,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_cardboard_box_10",
        "percent_real": 10,
        "train_synthetic": 81,
        "train_real": 9,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_cardboard_box_30",
        "percent_real": 30,
        "train_synthetic": 63,
        "train_real": 27,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_cardboard_box_50",
        "percent_real": 50,
        "train_synthetic": 45,
        "train_real": 45,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_cardboard_box_70",
        "percent_real": 70,
        "train_synthetic": 27,
        "train_real": 63,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_cardboard_box_90",
        "percent_real": 90,
        "train_synthetic": 9,
        "train_real": 81,
        "test_real": 10,
    },
    {
        "name": "simplemixed_generalisation_cardboard_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 90,
        "test_real": 10,
    },
]

# Cardboard BOX experiments - Stage 2 uses pattern, Test uses cardboard
FINETUNE_EXPERIMENTS_CARDBOARD_BOX = [
    {
        "name": "finetune_cardboard_box_0",
        "percent_real": 0,
        "train_synthetic": 90,
        "train_real": 0,
        "test_real": 10,
    },
    {
        "name": "finetune_cardboard_box_10",
        "percent_real": 10,
        "train_synthetic": 81,
        "train_real": 9,
        "test_real": 10,
    },
    {
        "name": "finetune_cardboard_box_30",
        "percent_real": 30,
        "train_synthetic": 63,
        "train_real": 27,
        "test_real": 10,
    },
    {
        "name": "finetune_cardboard_box_50",
        "percent_real": 50,
        "train_synthetic": 45,
        "train_real": 45,
        "test_real": 10,
    },
    {
        "name": "finetune_cardboard_box_70",
        "percent_real": 70,
        "train_synthetic": 27,
        "train_real": 63,
        "test_real": 10,
    },
    {
        "name": "finetune_cardboard_box_90",
        "percent_real": 90,
        "train_synthetic": 9,
        "train_real": 81,
        "test_real": 10,
    },
    {
        "name": "finetune_cardboard_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 90,
        "test_real": 10,
    },
]

SIMPLEMIXED_EXPERIMENTS_CARDBOARD_BOX = [
    {
        "name": "simplemixed_cardboard_box_0",
        "percent_real": 0,
        "train_synthetic": 90,
        "train_real": 0,
        "test_real": 10,
    },
    {
        "name": "simplemixed_cardboard_box_10",
        "percent_real": 10,
        "train_synthetic": 81,
        "train_real": 9,
        "test_real": 10,
    },
    {
        "name": "simplemixed_cardboard_box_30",
        "percent_real": 30,
        "train_synthetic": 63,
        "train_real": 27,
        "test_real": 10,
    },
    {
        "name": "simplemixed_cardboard_box_50",
        "percent_real": 50,
        "train_synthetic": 45,
        "train_real": 45,
        "test_real": 10,
    },
    {
        "name": "simplemixed_cardboard_box_70",
        "percent_real": 70,
        "train_synthetic": 27,
        "train_real": 63,
        "test_real": 10,
    },
    {
        "name": "simplemixed_cardboard_box_90",
        "percent_real": 90,
        "train_synthetic": 9,
        "train_real": 81,
        "test_real": 10,
    },
    {
        "name": "simplemixed_cardboard_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 90,
        "test_real": 10,
    },
]

# Generalisation LARVIK experiments - Stage 2 uses Rv4, Test uses Larvik
FINETUNE_EXPERIMENTS_GEN_LARVIK = [
    {
        "name": "finetune_generalisation_larvik_0",
        "percent_real": 0,
        "train_synthetic": 1350,
        "train_real": 0,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_larvik_10",
        "percent_real": 10,
        "train_synthetic": 1215,
        "train_real": 135,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_larvik_30",
        "percent_real": 30,
        "train_synthetic": 945,
        "train_real": 405,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_larvik_50",
        "percent_real": 50,
        "train_synthetic": 675,
        "train_real": 675,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_larvik_70",
        "percent_real": 70,
        "train_synthetic": 405,
        "train_real": 945,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_larvik_90",
        "percent_real": 90,
        "train_synthetic": 135,
        "train_real": 1215,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_larvik_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 1350,
        "test_real": 150,
    },
]

SIMPLEMIXED_EXPERIMENTS_GEN_LARVIK = [
    {
        "name": "simplemixed_generalisation_larvik_0",
        "percent_real": 0,
        "train_synthetic": 1350,
        "train_real": 0,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_larvik_10",
        "percent_real": 10,
        "train_synthetic": 1215,
        "train_real": 135,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_larvik_30",
        "percent_real": 30,
        "train_synthetic": 945,
        "train_real": 405,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_larvik_50",
        "percent_real": 50,
        "train_synthetic": 675,
        "train_real": 675,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_larvik_70",
        "percent_real": 70,
        "train_synthetic": 405,
        "train_real": 945,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_larvik_90",
        "percent_real": 90,
        "train_synthetic": 135,
        "train_real": 1215,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_larvik_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 1350,
        "test_real": 150,
    },
]

# LARVIK experiments - Stage 2 uses Larvik, Test uses Larvik (same source)
FINETUNE_EXPERIMENTS_LARVIK = [
    {
        "name": "finetune_larvik_0",
        "percent_real": 0,
        "train_synthetic": 1350,
        "train_real": 0,
        "test_real": 150,
    },
    {
        "name": "finetune_larvik_10",
        "percent_real": 10,
        "train_synthetic": 1215,
        "train_real": 135,
        "test_real": 150,
    },
    {
        "name": "finetune_larvik_30",
        "percent_real": 30,
        "train_synthetic": 945,
        "train_real": 405,
        "test_real": 150,
    },
    {
        "name": "finetune_larvik_50",
        "percent_real": 50,
        "train_synthetic": 675,
        "train_real": 675,
        "test_real": 150,
    },
    {
        "name": "finetune_larvik_70",
        "percent_real": 70,
        "train_synthetic": 405,
        "train_real": 945,
        "test_real": 150,
    },
    {
        "name": "finetune_larvik_90",
        "percent_real": 90,
        "train_synthetic": 135,
        "train_real": 1215,
        "test_real": 150,
    },
    {
        "name": "finetune_larvik_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 1350,
        "test_real": 150,
    },
]

SIMPLEMIXED_EXPERIMENTS_LARVIK = [
    {
        "name": "simplemixed_larvik_0",
        "percent_real": 0,
        "train_synthetic": 1350,
        "train_real": 0,
        "test_real": 150,
    },
    {
        "name": "simplemixed_larvik_10",
        "percent_real": 10,
        "train_synthetic": 1215,
        "train_real": 135,
        "test_real": 150,
    },
    {
        "name": "simplemixed_larvik_30",
        "percent_real": 30,
        "train_synthetic": 945,
        "train_real": 405,
        "test_real": 150,
    },
    {
        "name": "simplemixed_larvik_50",
        "percent_real": 50,
        "train_synthetic": 675,
        "train_real": 675,
        "test_real": 150,
    },
    {
        "name": "simplemixed_larvik_70",
        "percent_real": 70,
        "train_synthetic": 405,
        "train_real": 945,
        "test_real": 150,
    },
    {
        "name": "simplemixed_larvik_90",
        "percent_real": 90,
        "train_synthetic": 135,
        "train_real": 1215,
        "test_real": 150,
    },
    {
        "name": "simplemixed_larvik_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 1350,
        "test_real": 150,
    },
]

# Generalisation RV4 experiments - Stage 2 uses Larvik, Test uses Rv4
FINETUNE_EXPERIMENTS_GEN_RV4 = [
    {
        "name": "finetune_generalisation_rv4_0",
        "percent_real": 0,
        "train_synthetic": 1350,
        "train_real": 0,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_rv4_10",
        "percent_real": 10,
        "train_synthetic": 1215,
        "train_real": 135,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_rv4_30",
        "percent_real": 30,
        "train_synthetic": 945,
        "train_real": 405,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_rv4_50",
        "percent_real": 50,
        "train_synthetic": 675,
        "train_real": 675,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_rv4_70",
        "percent_real": 70,
        "train_synthetic": 405,
        "train_real": 945,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_rv4_90",
        "percent_real": 90,
        "train_synthetic": 135,
        "train_real": 1215,
        "test_real": 150,
    },
    {
        "name": "finetune_generalisation_rv4_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 1350,
        "test_real": 150,
    },
]

SIMPLEMIXED_EXPERIMENTS_GEN_RV4 = [
    {
        "name": "simplemixed_generalisation_rv4_0",
        "percent_real": 0,
        "train_synthetic": 1350,
        "train_real": 0,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_rv4_10",
        "percent_real": 10,
        "train_synthetic": 1215,
        "train_real": 135,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_rv4_30",
        "percent_real": 30,
        "train_synthetic": 945,
        "train_real": 405,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_rv4_50",
        "percent_real": 50,
        "train_synthetic": 675,
        "train_real": 675,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_rv4_70",
        "percent_real": 70,
        "train_synthetic": 405,
        "train_real": 945,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_rv4_90",
        "percent_real": 90,
        "train_synthetic": 135,
        "train_real": 1215,
        "test_real": 150,
    },
    {
        "name": "simplemixed_generalisation_rv4_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 1350,
        "test_real": 150,
    },
]

# RV4 experiments - Stage 2 uses Rv4, Test uses Rv4 (same source)
FINETUNE_EXPERIMENTS_RV4 = [
    {
        "name": "finetune_rv4_0",
        "percent_real": 0,
        "train_synthetic": 1350,
        "train_real": 0,
        "test_real": 150,
    },
    {
        "name": "finetune_rv4_10",
        "percent_real": 10,
        "train_synthetic": 1215,
        "train_real": 135,
        "test_real": 150,
    },
    {
        "name": "finetune_rv4_30",
        "percent_real": 30,
        "train_synthetic": 945,
        "train_real": 405,
        "test_real": 150,
    },
    {
        "name": "finetune_rv4_50",
        "percent_real": 50,
        "train_synthetic": 675,
        "train_real": 675,
        "test_real": 150,
    },
    {
        "name": "finetune_rv4_70",
        "percent_real": 70,
        "train_synthetic": 405,
        "train_real": 945,
        "test_real": 150,
    },
    {
        "name": "finetune_rv4_90",
        "percent_real": 90,
        "train_synthetic": 135,
        "train_real": 1215,
        "test_real": 150,
    },
    {
        "name": "finetune_rv4_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 1350,
        "test_real": 150,
    },
]

SIMPLEMIXED_EXPERIMENTS_RV4 = [
    {
        "name": "simplemixed_rv4_0",
        "percent_real": 0,
        "train_synthetic": 1350,
        "train_real": 0,
        "test_real": 150,
    },
    {
        "name": "simplemixed_rv4_10",
        "percent_real": 10,
        "train_synthetic": 1215,
        "train_real": 135,
        "test_real": 150,
    },
    {
        "name": "simplemixed_rv4_30",
        "percent_real": 30,
        "train_synthetic": 945,
        "train_real": 405,
        "test_real": 150,
    },
    {
        "name": "simplemixed_rv4_50",
        "percent_real": 50,
        "train_synthetic": 675,
        "train_real": 675,
        "test_real": 150,
    },
    {
        "name": "simplemixed_rv4_70",
        "percent_real": 70,
        "train_synthetic": 405,
        "train_real": 945,
        "test_real": 150,
    },
    {
        "name": "simplemixed_rv4_90",
        "percent_real": 90,
        "train_synthetic": 135,
        "train_real": 1215,
        "test_real": 150,
    },
    {
        "name": "simplemixed_rv4_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 1350,
        "test_real": 150,
    },
]


def create_splits_for_experiment(
    experiment: dict,
    synthetic_files: list[str],
    real_files: list[str],
    output_dir: Path,
    seed: int = 42,
    test_files: list[str] | None = None,
    fixed_test_set: list[str] | None = None,
) -> None:
    """Create train/test splits for a single experiment.

    Args:
        experiment: Experiment configuration dict
        synthetic_files: List of synthetic file paths
        real_files: List of real file paths for training
        output_dir: Output directory for splits
        seed: Random seed for reproducibility
        test_files: Optional separate test files (e.g., for generalization experiments)
        fixed_test_set: Pre-selected test set to ensure consistency across experiments
    """
    import random

    random.seed(seed)

    # Shuffle files for randomness
    synthetic_shuffled = synthetic_files.copy()
    real_shuffled = real_files.copy()
    random.shuffle(synthetic_shuffled)
    random.shuffle(real_shuffled)

    # Use fixed test set if provided (ensures same test set across all experiments)
    if fixed_test_set is not None:
        test_shuffled = fixed_test_set
    elif test_files is not None:
        # Use separate test files if provided (for generalization experiments)
        test_shuffled = test_files.copy()
        random.shuffle(test_shuffled)
    else:
        test_shuffled = real_shuffled  # Use real_files for both train and test

    # Extract counts
    n_train_synth = experiment["train_synthetic"]
    n_train_real = experiment["train_real"]
    n_test_real = experiment["test_real"]

    # Validate we have enough data
    if n_train_synth > len(synthetic_shuffled):
        raise ValueError(
            f"Not enough synthetic data: need {n_train_synth}, "
            f"have {len(synthetic_shuffled)}"
        )

    # Check real training data availability
    if n_train_real > len(real_shuffled):
        raise ValueError(
            f"Not enough real training data: need {n_train_real}, "
            f"have {len(real_shuffled)}"
        )

    # Check test data availability
    if test_files is not None:
        # Using separate test files (generalization experiments)
        if n_test_real > len(test_shuffled):
            raise ValueError(
                f"Not enough test data: need {n_test_real}, have {len(test_shuffled)}"
            )
    elif fixed_test_set is not None:
        # Using pre-allocated fixed test set (no validation needed, already split)
        if n_test_real > len(fixed_test_set):
            raise ValueError(
                f"Not enough fixed test data: need {n_test_real}, have {len(fixed_test_set)}"
            )
    else:
        # Using real_files for both train and test
        if (n_train_real + n_test_real) > len(real_shuffled):
            raise ValueError(
                f"Not enough real data: need {n_train_real + n_test_real}, "
                f"have {len(real_shuffled)}"
            )

    # Split synthetic data
    train_synthetic = synthetic_shuffled[:n_train_synth]

    # Split real data for training (from real_files)
    train_real = real_shuffled[:n_train_real]

    # Split test data (use fixed test set if provided, otherwise use test_files or real_files)
    if fixed_test_set is not None:
        test_real = fixed_test_set[:n_test_real]
    elif test_files is not None:
        test_real = test_shuffled[:n_test_real]
    else:
        # Test comes after training data in real_shuffled
        test_real = real_shuffled[n_train_real : n_train_real + n_test_real]

    # Combine training data
    train_all = train_synthetic + train_real
    random.shuffle(train_all)  # Shuffle combined training set

    # For FT strategy: also save separate synthetic and real training lists
    train_synthetic_only = train_synthetic.copy()
    train_real_only = train_real.copy()

    # Create output directory
    exp_dir = output_dir / experiment["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Determine strategy type from experiment name
    is_simplemixed = experiment["name"].startswith("simplemixed_")

    # Save splits
    splits = {
        "train.json": train_all,  # Azure expects train.json
        "train_all.json": train_all,  # For SM strategy (kept for reference)
        "train_synthetic.json": train_synthetic_only,  # For FT stage 1
        "train_real.json": train_real_only,  # For FT stage 2
        "test.json": test_real,  # Always real data
        "val.json": test_real,  # Use same as test (following Wachter)
    }

    for filename, data in splits.items():
        with open(exp_dir / filename, "w") as f:
            json.dump(data, f, indent=2)

    # Save metadata
    metadata = {
        "experiment_name": experiment["name"],
        "strategy_type": "simplemixed" if is_simplemixed else "finetune",
        "percent_real": experiment["percent_real"],
        "train_synthetic_count": len(train_synthetic),
        "train_real_count": len(train_real),
        "train_total_count": len(train_all),
        "test_count": len(test_real),
        "val_count": len(test_real),
        "synth_real_ratio": (
            f"{100 - experiment['percent_real']}/{experiment['percent_real']}"
        ),
        "note": (
            "train.json contains shuffled synthetic+real for Azure compatibility. "
            "SM uses train.json directly. FT will use train_synthetic.json "
            "then train_real.json in two-stage training (not yet implemented)."
        ),
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(
        f"✓ Created splits for [cyan]{experiment['name']}[/cyan]: "
        f"{len(train_synthetic)} synth + {len(train_real)} real "
        f"→ {len(train_all)} train, {len(test_real)} test"
    )


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Generate all Simple Mixed and Fine-Tuned experiment splits."""
    console.print(
        "[bold green]Creating Wachter et al. (2026) Experiment Splits[/bold green]"
    )
    console.print("Generating both SM and FT strategy splits\n")

    # Load configuration
    from ml_segmentation.schema_config import ConfigSchema

    pcfg = ConfigSchema(**OmegaConf.to_container(cfg, resolve=True))

    # Output directory
    output_base = Path("data/model_ready/wachter_splits")
    output_base.mkdir(parents=True, exist_ok=True)

    console.print(f"Output directory: {output_base}\n")

    # === LOAD DATA FILES ===
    # Get synthetic box files (Benchmark prefix)
    synthetic_box_prefixes = get_datasets_prefixes(
        experiment_strategy="verification_box",
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    synthetic_box_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=synthetic_box_prefixes["train_prefixes"],
        test_prefixes=[],  # We want all files
    )

    # Get real box files (Box prefix)
    real_box_prefixes = get_datasets_prefixes(
        experiment_strategy="verification_real_box",
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    real_box_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=real_box_prefixes["train_prefixes"],
        test_prefixes=[],
    )

    # Get synthetic slope files (FracMan prefix)
    synthetic_slope_prefixes = get_datasets_prefixes(
        experiment_strategy="verification_dfn",
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    synthetic_slope_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=synthetic_slope_prefixes["train_prefixes"],
        test_prefixes=[],
    )

    # Get real slope files (Larvik, Rv4 prefixes)
    real_slope_prefixes = get_datasets_prefixes(
        experiment_strategy="verification_real_rock_slope",
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    real_slope_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=real_slope_prefixes["train_prefixes"],
        test_prefixes=[],
    )

    # Get real pattern files for generalisation experiments
    # Using Box-drone-pattern and Box-camera-pattern
    pattern_prefixes = ["Box-drone-pattern", "Box-camera-pattern"]
    pattern_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=pattern_prefixes,
        test_prefixes=[],
    )

    # Get cardboard-only files (exclude pattern files) for generalisation experiments
    # real_box_files includes both cardboard and pattern, so filter them
    cardboard_prefixes = ["Box-drone-cardboard", "Box-camera-cardboard"]
    cardboard_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=cardboard_prefixes,
        test_prefixes=[],
    )

    # Get FracMan synthetic files for Larvik/Rv4 experiments
    fracman_prefixes = ["FracMan"]
    fracman_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=fracman_prefixes,
        test_prefixes=[],
    )

    # Get Larvik files
    larvik_prefixes = ["Larvik"]
    larvik_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=larvik_prefixes,
        test_prefixes=[],
    )

    # Get Rv4 files (matches both "Rv4" and "Rv4_P" prefixes)
    rv4_prefixes = ["Rv4"]
    rv4_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=rv4_prefixes,
        test_prefixes=[],
    )

    # === PRE-SELECT TEST SETS (Fixed across all experiments of same type) ===
    # This ensures consistent test sets across all experiments of the same type
    # and prevents data leakage by separating train and test pools BEFORE experiments
    import random

    # Determine test set sizes from experiment definitions (max test_real per type)
    box_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_BOX + FINETUNE_EXPERIMENTS_BOX
    )
    slope_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_SLOPE + FINETUNE_EXPERIMENTS_SLOPE
    )
    pattern_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_PATTERN_BOX
        + FINETUNE_EXPERIMENTS_PATTERN_BOX
    )
    cardboard_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_CARDBOARD_BOX
        + FINETUNE_EXPERIMENTS_CARDBOARD_BOX
    )
    larvik_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_LARVIK + FINETUNE_EXPERIMENTS_LARVIK
    )
    rv4_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_RV4 + FINETUNE_EXPERIMENTS_RV4
    )

    # Pre-select BOX test set (same for all box experiments)
    random.seed(pcfg.experiment.seed)
    box_test_shuffled = real_box_files.copy()
    random.shuffle(box_test_shuffled)
    box_fixed_test = box_test_shuffled[-box_test_size:]  # type: ignore[operator]
    box_train_pool = box_test_shuffled[:-box_test_size]  # type: ignore[operator]

    # Pre-select SLOPE test set (same for all slope experiments)
    random.seed(pcfg.experiment.seed)
    slope_test_shuffled = real_slope_files.copy()
    random.shuffle(slope_test_shuffled)
    slope_fixed_test = slope_test_shuffled[-slope_test_size:]  # type: ignore[operator]
    slope_train_pool = slope_test_shuffled[:-slope_test_size]  # type: ignore[operator]

    # Pre-select PATTERN test set (same for all pattern_box experiments)
    random.seed(pcfg.experiment.seed)
    pattern_test_shuffled = pattern_files.copy()
    random.shuffle(pattern_test_shuffled)
    pattern_fixed_test = pattern_test_shuffled[-pattern_test_size:]  # type: ignore[operator]
    pattern_train_pool = pattern_test_shuffled[:-pattern_test_size]  # type: ignore[operator]

    # Pre-select CARDBOARD BOX test set (same for all cardboard_box experiments)
    random.seed(pcfg.experiment.seed)
    cardboard_test_shuffled = cardboard_files.copy()
    random.shuffle(cardboard_test_shuffled)
    cardboard_fixed_test = cardboard_test_shuffled[-cardboard_test_size:]  # type: ignore[operator]
    cardboard_train_pool = cardboard_test_shuffled[:-cardboard_test_size]  # type: ignore[operator]

    # Pre-select LARVIK test set (same for all larvik experiments)
    random.seed(pcfg.experiment.seed)
    larvik_test_shuffled = larvik_files.copy()
    random.shuffle(larvik_test_shuffled)
    larvik_fixed_test = larvik_test_shuffled[-larvik_test_size:]  # type: ignore[operator]
    larvik_train_pool = larvik_test_shuffled[:-larvik_test_size]  # type: ignore[operator]

    # Pre-select RV4 test set (same for all rv4 experiments)
    random.seed(pcfg.experiment.seed)
    rv4_test_shuffled = rv4_files.copy()
    random.shuffle(rv4_test_shuffled)
    rv4_fixed_test = rv4_test_shuffled[-rv4_test_size:]  # type: ignore[operator]
    rv4_train_pool = rv4_test_shuffled[:-rv4_test_size]  # type: ignore[operator]

    # === PRE-SELECT TEST SETS FOR GENERALIZATION EXPERIMENTS ===
    # These ensure train and test domains are properly separated

    # Compute test set sizes for generalization experiments
    gen_pattern_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_GEN_PATTERN_BOX
        + FINETUNE_EXPERIMENTS_GEN_PATTERN_BOX
    )
    gen_cardboard_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_GEN_CARDBOARD_BOX
        + FINETUNE_EXPERIMENTS_GEN_CARDBOARD_BOX
    )
    gen_larvik_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_GEN_LARVIK + FINETUNE_EXPERIMENTS_GEN_LARVIK
    )
    gen_rv4_test_size = max(  # type: ignore[type-var]
        exp["test_real"]
        for exp in SIMPLEMIXED_EXPERIMENTS_GEN_RV4 + FINETUNE_EXPERIMENTS_GEN_RV4
    )

    # Gen Pattern Box: train on cardboard, test on pattern
    random.seed(pcfg.experiment.seed)
    gen_pattern_test_shuffled = pattern_files.copy()
    random.shuffle(gen_pattern_test_shuffled)
    gen_pattern_fixed_test = gen_pattern_test_shuffled[-gen_pattern_test_size:]  # type: ignore[operator]

    # Cardboard pool for training in generalization experiments (completely separate from pattern)
    cardboard_train_shuffled = cardboard_files.copy()
    random.shuffle(cardboard_train_shuffled)
    cardboard_train_pool_for_gen = (
        cardboard_train_shuffled  # All available for gen training
    )

    # Gen Cardboard Box: train on pattern, test on cardboard
    random.seed(pcfg.experiment.seed)
    gen_cardboard_test_shuffled = cardboard_files.copy()
    random.shuffle(gen_cardboard_test_shuffled)
    gen_cardboard_fixed_test = gen_cardboard_test_shuffled[-gen_cardboard_test_size:]  # type: ignore[operator]

    # Pattern pool for training (completely separate from cardboard test)
    pattern_train_shuffled = pattern_files.copy()
    random.shuffle(pattern_train_shuffled)
    pattern_train_pool_for_gen = pattern_train_shuffled  # All available for training

    # Gen Larvik: train on RV4, test on Larvik
    random.seed(pcfg.experiment.seed)
    gen_larvik_test_shuffled = larvik_files.copy()
    random.shuffle(gen_larvik_test_shuffled)
    gen_larvik_fixed_test = gen_larvik_test_shuffled[-gen_larvik_test_size:]  # type: ignore[operator]

    # RV4 pool for training (separate from Larvik test)
    rv4_train_pool_for_gen = rv4_files.copy()
    random.shuffle(rv4_train_pool_for_gen)

    # Gen RV4: train on Larvik, test on RV4
    random.seed(pcfg.experiment.seed)
    gen_rv4_test_shuffled = rv4_files.copy()
    random.shuffle(gen_rv4_test_shuffled)
    gen_rv4_fixed_test = gen_rv4_test_shuffled[-gen_rv4_test_size:]  # type: ignore[operator]

    # Larvik pool for training (separate from RV4 test)
    larvik_train_pool_for_gen = larvik_files.copy()
    random.shuffle(larvik_train_pool_for_gen)

    # === SIMPLE MIXED (SM) EXPERIMENTS ===
    console.print("[bold yellow]Simple Mixed (SM) Strategy - BOX[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_box_files)} synth, {len(box_train_pool)} real (train), {len(box_fixed_test)} real (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=box_train_pool,
            output_dir=output_base / "sm_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=box_fixed_test,
        )
    console.print()

    console.print("[bold yellow]Simple Mixed (SM) Strategy - SLOPE[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_slope_files)} synth, {len(slope_train_pool)} real (train), {len(slope_fixed_test)} real (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_SLOPE:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_slope_files,
            real_files=slope_train_pool,
            output_dir=output_base / "sm_slope",
            seed=pcfg.experiment.seed,
            fixed_test_set=slope_fixed_test,
        )

    console.print()
    console.print(
        "[bold yellow]Simple Mixed (SM) Strategy - GENERALISATION PATTERN BOX[/bold yellow]"
    )
    console.print(
        f"Available: {len(synthetic_box_files)} synth cardboard, "
        f"{len(cardboard_train_pool_for_gen)} real cardboard (train), {len(gen_pattern_fixed_test)} real pattern (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_GEN_PATTERN_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=cardboard_train_pool_for_gen,  # Train on cardboard ONLY
            output_dir=output_base / "sm_gen_pattern_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=gen_pattern_fixed_test,  # Test on pattern (pre-selected)
        )
    console.print()

    console.print("[bold yellow]Simple Mixed (SM) Strategy - PATTERN BOX[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_box_files)} synth cardboard, "
        f"{len(pattern_train_pool)} real pattern (train), {len(pattern_fixed_test)} real pattern (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_PATTERN_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=pattern_train_pool,  # Use training pool only
            output_dir=output_base / "sm_pattern_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=pattern_fixed_test,  # Use pre-selected fixed test set
        )
    console.print()

    console.print(
        "[bold yellow]Simple Mixed (SM) Strategy - GENERALISATION CARDBOARD BOX[/bold yellow]"
    )
    console.print(
        f"Available: {len(synthetic_box_files)} synth cardboard, "
        f"{len(pattern_train_pool_for_gen)} real pattern (train), {len(gen_cardboard_fixed_test)} real cardboard (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_GEN_CARDBOARD_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=pattern_train_pool_for_gen,  # Train on pattern ONLY
            output_dir=output_base / "sm_gen_cardboard_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=gen_cardboard_fixed_test,  # Test on cardboard (pre-selected)
        )
    console.print()

    console.print(
        "[bold yellow]Simple Mixed (SM) Strategy - CARDBOARD BOX[/bold yellow]"
    )
    console.print(
        f"Available: {len(synthetic_box_files)} synth cardboard, "
        f"{len(cardboard_train_pool)} real cardbox (train), {len(cardboard_fixed_test)} real cardbox (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_CARDBOARD_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=cardboard_train_pool,  # Use training pool only
            output_dir=output_base / "sm_cardboard_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=cardboard_fixed_test,  # Use pre-selected fixed test set
        )
    console.print()

    console.print(
        "[bold yellow]Simple Mixed (SM) Strategy - GENERALISATION LARVIK[/bold yellow]"
    )
    console.print(
        f"Available: {len(fracman_files)} synth FracMan, "
        f"{len(rv4_train_pool_for_gen)} real rv4 (train), {len(gen_larvik_fixed_test)} real larvik (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_GEN_LARVIK:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=fracman_files,
            real_files=rv4_train_pool_for_gen,  # Train on RV4 ONLY
            output_dir=output_base / "sm_gen_larvik",
            seed=pcfg.experiment.seed,
            fixed_test_set=gen_larvik_fixed_test,  # Test on Larvik (pre-selected)
        )
    console.print()

    console.print("[bold yellow]Simple Mixed (SM) Strategy - LARVIK[/bold yellow]")
    console.print(
        f"Available: {len(fracman_files)} synth FracMan, "
        f"{len(larvik_train_pool)} real Larvik (train), {len(larvik_fixed_test)} real Larvik (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_LARVIK:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=fracman_files,
            real_files=larvik_train_pool,  # Use training pool only
            output_dir=output_base / "sm_larvik",
            seed=pcfg.experiment.seed,
            fixed_test_set=larvik_fixed_test,  # Use pre-selected fixed test set
        )
    console.print()

    console.print(
        "[bold yellow]Simple Mixed (SM) Strategy - GENERALISATION RV4[/bold yellow]"
    )
    console.print(
        f"Available: {len(fracman_files)} synth FracMan, "
        f"{len(larvik_train_pool_for_gen)} real larvik (train), {len(gen_rv4_fixed_test)} real rv4 (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_GEN_RV4:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=fracman_files,
            real_files=larvik_train_pool_for_gen,  # Train on Larvik ONLY
            output_dir=output_base / "sm_gen_rv4",
            seed=pcfg.experiment.seed,
            fixed_test_set=gen_rv4_fixed_test,  # Test on RV4 (pre-selected)
        )
    console.print()

    console.print("[bold yellow]Simple Mixed (SM) Strategy - RV4[/bold yellow]")
    console.print(
        f"Available: {len(fracman_files)} synth FracMan, "
        f"{len(rv4_train_pool)} real Rv4 (train), {len(rv4_fixed_test)} real Rv4 (test)"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_RV4:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=fracman_files,
            real_files=rv4_train_pool,  # Use training pool only
            output_dir=output_base / "sm_rv4",
            seed=pcfg.experiment.seed,
            fixed_test_set=rv4_fixed_test,  # Use pre-selected fixed test set
        )
    console.print()

    # === FINE-TUNED (FT) EXPERIMENTS ===
    # FT experiments use the same test sets as SM experiments
    console.print("[bold yellow]Fine-Tuned (FT) Strategy - BOX[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_box_files)} synth, {len(box_train_pool)} real (train), {len(box_fixed_test)} real (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=box_train_pool,
            output_dir=output_base / "ft_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=box_fixed_test,
        )
    console.print()

    console.print("[bold yellow]Fine-Tuned (FT) Strategy - SLOPE[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_slope_files)} synth, {len(slope_train_pool)} real (train), {len(slope_fixed_test)} real (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_SLOPE:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_slope_files,
            real_files=slope_train_pool,
            output_dir=output_base / "ft_slope",
            seed=pcfg.experiment.seed,
            fixed_test_set=slope_fixed_test,
        )

    console.print()
    console.print(
        "[bold yellow]Finetune (FT) Strategy - GENERALISATION PATTERN BOX[/bold yellow]"
    )
    console.print(
        f"Available: {len(synthetic_box_files)} synth cardboard (Stage 1), "
        f"{len(cardboard_train_pool_for_gen)} real cardboard (Stage 2), {len(gen_pattern_fixed_test)} real pattern (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_GEN_PATTERN_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=cardboard_train_pool_for_gen,  # Train on cardboard ONLY
            output_dir=output_base / "ft_gen_pattern_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=gen_pattern_fixed_test,  # Test on pattern (pre-selected)
        )
    console.print()
    console.print("[bold yellow]Fine-Tuned (FT) Strategy - PATTERN BOX[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_box_files)} synth cardboard, "
        f"{len(pattern_train_pool)} real pattern (train), {len(pattern_fixed_test)} real pattern (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_PATTERN_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=pattern_train_pool,  # Use training pool only
            output_dir=output_base / "ft_pattern_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=pattern_fixed_test,  # Use pre-selected fixed test set
        )

    console.print()
    console.print(
        "[bold yellow]Finetune (FT) Strategy - GENERALISATION CARDBOARD BOX[/bold yellow]"
    )
    console.print(
        f"Available: {len(synthetic_box_files)} synth cardboard (Stage 1), "
        f"{len(pattern_train_pool_for_gen)} real pattern (Stage 2), {len(gen_cardboard_fixed_test)} real cardboard (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_GEN_CARDBOARD_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=pattern_train_pool_for_gen,  # Train on pattern ONLY
            output_dir=output_base / "ft_gen_cardboard_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=gen_cardboard_fixed_test,  # Test on cardboard (pre-selected)
        )

    console.print()
    console.print("[bold yellow]Fine-Tuned (FT) Strategy - CARDBOARD BOX[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_box_files)} synth cardboard, "
        f"{len(cardboard_train_pool)} real cardbox (train), {len(cardboard_fixed_test)} real cardbox (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_CARDBOARD_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=cardboard_train_pool,  # Use training pool only
            output_dir=output_base / "ft_cardboard_box",
            seed=pcfg.experiment.seed,
            fixed_test_set=cardboard_fixed_test,  # Use pre-selected fixed test set
        )

    console.print()
    console.print(
        "[bold yellow]Finetune (FT) Strategy - GENERALISATION LARVIK[/bold yellow]"
    )
    console.print(
        f"Available: {len(fracman_files)} synth FracMan (Stage 1), "
        f"{len(rv4_train_pool_for_gen)} real rv4 (Stage 2), {len(gen_larvik_fixed_test)} real larvik (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_GEN_LARVIK:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=fracman_files,
            real_files=rv4_train_pool_for_gen,  # Train on RV4 ONLY
            output_dir=output_base / "ft_gen_larvik",
            seed=pcfg.experiment.seed,
            fixed_test_set=gen_larvik_fixed_test,  # Test on Larvik (pre-selected)
        )
    console.print()
    console.print("[bold yellow]Fine-Tuned (FT) Strategy - LARVIK[/bold yellow]")
    console.print(
        f"Available: {len(fracman_files)} synth FracMan, "
        f"{len(larvik_train_pool)} real Larvik (train), {len(larvik_fixed_test)} real Larvik (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_LARVIK:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=fracman_files,
            real_files=larvik_train_pool,  # Use training pool only
            output_dir=output_base / "ft_larvik",
            seed=pcfg.experiment.seed,
            fixed_test_set=larvik_fixed_test,  # Use pre-selected fixed test set
        )

    console.print()
    console.print(
        "[bold yellow]Finetune (FT) Strategy - GENERALISATION RV4[/bold yellow]"
    )
    console.print(
        f"Available: {len(fracman_files)} synth FracMan (Stage 1), "
        f"{len(larvik_train_pool_for_gen)} real larvik (Stage 2), {len(gen_rv4_fixed_test)} real rv4 (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_GEN_RV4:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=fracman_files,
            real_files=larvik_train_pool_for_gen,  # Train on Larvik ONLY
            output_dir=output_base / "ft_gen_rv4",
            seed=pcfg.experiment.seed,
            fixed_test_set=gen_rv4_fixed_test,  # Test on RV4 (pre-selected)
        )

    console.print()
    console.print("[bold yellow]Fine-Tuned (FT) Strategy - RV4[/bold yellow]")
    console.print(
        f"Available: {len(fracman_files)} synth FracMan, "
        f"{len(rv4_train_pool)} real Rv4 (train), {len(rv4_fixed_test)} real Rv4 (test)"
    )
    for exp in FINETUNE_EXPERIMENTS_RV4:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=fracman_files,
            real_files=rv4_train_pool,  # Use training pool only
            output_dir=output_base / "ft_rv4",
            seed=pcfg.experiment.seed,
            fixed_test_set=rv4_fixed_test,  # Use pre-selected fixed test set
        )

    console.print()
    console.print("[bold green]✓ All splits created successfully![/bold green]")
    console.print(f"\nSplits saved to: {output_base}")
    console.print(
        "\n[cyan]Generated:[/cyan]"
        "\n  • 14 Box experiments: sm_box/ and ft_box/"
        "\n  • 14 Slope experiments: sm_slope/ and ft_slope/"
        "\n  • 14 Generalisation Pattern Box: sm_gen_pattern_box/ and ft_gen_pattern_box/"
        "\n  • 14 Pattern Box: sm_pattern_box/ and ft_pattern_box/"
        "\n  • 14 Generalisation Cardboard Box: sm_gen_cardboard_box/ and ft_gen_cardboard_box/"
        "\n  • 14 Cardboard Box: sm_cardboard_box/ and ft_cardboard_box/"
        "\n  • 14 Generalisation Larvik: sm_gen_larvik/ and ft_gen_larvik/"
        "\n  • 14 Larvik: sm_larvik/ and ft_larvik/"
        "\n  • 14 Generalisation RV4: sm_gen_rv4/ and ft_gen_rv4/"
        "\n  • 14 RV4: sm_rv4/ and ft_rv4/"
        "\n  Total: 140 experiment configurations"
        "\n\n[cyan]Next steps:[/cyan]"
        "\n  1. Register splits in Azure ML (if deploying to cloud)"
        "\n  2. Run training using experiment_strategy parameter"
        "\n     - SM: simplemixed_box_10, simplemixed_slope_30, simplemixed_larvik_50, etc."
        "\n     - FT: finetune_box_10, finetune_slope_30, finetune_rv4_70, etc."
        "\n     - Gen: finetune_generalisation_pattern_box_10, finetune_generalisation_rv4_30, etc."
    )


if __name__ == "__main__":
    main()
