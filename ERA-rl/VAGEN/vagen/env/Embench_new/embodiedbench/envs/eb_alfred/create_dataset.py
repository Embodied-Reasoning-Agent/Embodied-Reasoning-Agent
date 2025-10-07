"""
Preprocess dataset for genereal tasks
"""

import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets
import multiprocessing as mp
from functools import partial
from collections import defaultdict
import numpy as np

from vagen.env.create_dataset import DatasetCreator
from vagen.env.sokoban.env import SokobanInterface
from vagen.env.sokoban.room_utils import get_shortest_action_path, plot_animation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--n_candidate', type=int, default=20000)
    parser.add_argument('--max_action_length', type=int, default=None)
    parser.add_argument('--force-gen', action='store_true', 
                        help='Force dataset generation even if files already exist')
    parser.add_argument('--data_dir', type=str, default='data/sokoban',)

    parser.add_argument('--dim_room', type=int, nargs=2, default=[6, 6],
                        help='Dimensions of the room [height, width]')
    parser.add_argument('--num_boxes', type=int, default=1,
                        help='Number of boxes in the environment')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of steps allowed')
    parser.add_argument('--search_depth', type=int, default=30,
                        help='Search depth that affects the starting position of the player')
    parser.add_argument('--visual_env', action='store_true',
                        help='Whether to use visual environment')
    
    parser.add_argument('--max_action_per_step', type=int, default=1,
                        help='Maximum number of actions per step')
    parser.add_argument('--max_action_penalty', type=float, default=-0.1,
                        help='Penalty for exceeding the maximum number of actions per step')
    parser.add_argument('--format_reward', type=float, default=0.5,
                        help='Reward for correct formatting')
    
    parser.add_argument(
        "--eval_set",
        type=str,
        default="base",
        help="Name of the evaluation set (default: base)"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Name of the experiment (default: empty string)"
    )
    parser.add_argument(
        "--down_sample_ratio",
        type=float,
        default=1.0,
        help="Down-sampling ratio for data (default: 1.0)"
    )
    parser.add_argument(
        "--selected_indexes",
        nargs="+",
        type=int,
        default=[],
        help="List of selected indexes (default: empty list)"
    )
    parser.add_argument(
        "--detection_box",
        action="store_true",
        default=False,
        help="Use detection box if specified (default: False)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=500,
        help="Image resolution (default: 500)"
    )
    
    import os
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '0'
        print("Set PYTHONHASHSEED to 0 for reproducibility")
    else:
        print(f"PYTHONHASHSEED already set to {os.environ['PYTHONHASHSEED']}")
    

    args = parser.parse_args()
    args.name = 'eb_alfred'
    args.env_config = {
        'eval_set': args.eval_set,
        'exp_name': args.exp_name,
        'down_sample_ratio': args.down_sample_ratio,
        'selected_indexes': args.selected_indexes,
        'detection_box': args.detection_box,
        'resolution': args.resolution
    }
    args.interface_config = {
        'max_action_per_step': args.max_action_per_step,
        'max_action_penalty': args.max_action_penalty,
        'format_reward': args.format_reward,
    }
    creator = DatasetCreator(config=vars(args))
    if args.max_action_length:
        creator.create_dataset(
            seed=args.start_seed,
            train_ratio=args.train_ratio,
            max_action_length=args.max_action_length,
            n_candidate=args.n_candidate,
            force_gen=args.force_gen
        )
    else:
        train_size = int(args.train_ratio * args.n_candidate)
        test_size = args.n_candidate - train_size
        creator.create_dataset(
            seed=args.start_seed,
            train_size=train_size,
            test_size=test_size,
            force_gen=args.force_gen)
