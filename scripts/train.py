# MIT License
#
# Copyright (c) 2019 seq2struct contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import functools
import collections
import datetime
import itertools
import json
import os
import shutil
import traceback
import pathlib
from typing import Type, List

import _jsonnet
import torch
import numpy as np

# noinspection PyUnresolvedReferences
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from duorat import datasets

# noinspection PyUnresolvedReferences
from duorat import preproc

# noinspection PyUnresolvedReferences
from duorat import models

# noinspection PyUnresolvedReferences
from duorat.asdl.lang.spider.spider_transition_system import SpiderTransitionSystem
from duorat.types import RATPreprocItem

from duorat.utils.determinism import set_torch_determinism, set_random_seed

# noinspection PyUnresolvedReferences
from duorat.utils import optimizers
from duorat.utils import registry, parallelizer
from duorat.utils import random_state
from duorat.utils import saver as saver_mod
from duorat.utils.evaluation import evaluate_default, load_from_lines
from third_party.spider.evaluation import LEVELS
from scripts.preprocess import Preprocessor


class Logger:
    def __init__(self, log_path=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, "a+")

    def log(self, msg):
        formatted = "[{}] {}".format(
            datetime.datetime.now().replace(microsecond=0).isoformat(), msg
        )
        # print(formatted)
        if self.log_file:
            self.log_file.write(formatted + "\n")
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, "a+")
            else:
                self.log_file.flush()


class Trainer:
    def __init__(self, logger, config):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.logger = logger
        self.config = config

        self.data_random = random_state.RandomContext(
            self.config["train"].get("data_seed", None)
        )
        self.model_random = random_state.RandomContext(
            self.config["train"].get("model_seed", None)
        )
        self.init_random = random_state.RandomContext(
            self.config["train"].get("init_seed", None)
        )
        other_seed = self.config["train"].get("other_seed", None)
        if other_seed:
            set_random_seed(seed=other_seed)

        with self.init_random:
            # 0. Construct preprocessors
            self.model_preproc = registry.construct(
                "preproc", self.config["model"]["preproc"],
            )

            if self.config["model"]["preproc"].get('pre_target_vocab', None) is not None:
                from shutil import copyfile
                copyfile(self.config["model"]["preproc"]['pre_target_vocab'], self.model_preproc.target_vocab_path)
            self.model_preproc.load()

            # 1. Construct model
            self.model = registry.construct(
                "model", self.config["model"], preproc=self.model_preproc,
            )
            self.model.to(device=device)

        if self.config["train"].get("deterministic", None):
            set_torch_determinism()

    def _log_loss(self, last_step, loss):
        self.logger.log("Step {}: loss={:.8f}".format(last_step, loss))
        # self.logger.log("Step {}: loss={:.4f}".format(last_step, loss))

    def _log_lr(self, last_step, lrs: List[dict]):
        for lr in lrs:
            self.logger.log(
                "Step {}: lr[{}]={:.8f}".format(last_step, lr["name"], lr["value"])
                # "Step {}: lr[{}]={:.4f}".format(last_step, lr["name"], lr["value"])
            )

    def _log_stats(self, last_step, eval_section, stats):
        self.logger.log(
            "Step {} stats, {}: {}".format(
                last_step,
                eval_section,
                ", ".join("{} = {}".format(k, v) for k, v in stats.items()),
            )
        )

    def train(self, modeldir):
        # Save the config info
        with open(
                os.path.join(
                    modeldir,
                    "config-{}.json".format(
                        datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")
                    ),
                ),
                "w",
        ) as f:
            json.dump(self.config, f, sort_keys=True, indent=4)

        # slight difference here vs. unrefactored train: The init_random starts over here. Could be fixed if it was important by saving random state at end of init
        with self.init_random:
            # We may be able to move optimizer and lr_scheduler to __init__ instead. Empirically it works fine. I think that's because saver.restore
            # resets the state by calling optimizer.load_state_dict.
            # But, if there is no saved file yet, I think this is not true, so might need to reset the optimizer manually?
            # For now, just creating it from scratch each time is safer and appears to be the same speed, but also means you have to pass in the config to train which is kind of ugly.
            optimizer = registry.construct(
                "optimizer",
                self.config["optimizer"],
                params=[
                    {
                        "name": "no-bert",
                        "params": (
                            parameters
                            for name, parameters in self.model.named_parameters()
                            if not name.startswith("initial_encoder.bert")
                        ),
                    },
                    {
                        "name": "bert",
                        "params": (
                            parameters
                            for name, parameters in self.model.named_parameters()
                            if name.startswith("initial_encoder.bert")
                        ),
                    },
                ],
            )
            lr_scheduler = registry.construct(
                "lr_scheduler",
                self.config.get("lr_scheduler", {"name": "noop"}),
                optimizer=optimizer,
            )

        # 1.5. Initialize Automatic Mixed Precision (AMP) training
        scaler = GradScaler(enabled=self.config["train"]["amp_enabled"])

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            self.model, optimizer
        )
        last_step, best_val_all_exact = saver.restore(model_dir=modeldir,
                                                      load_best=self.config["train"].get('load_best',
                                                                                         False))
        if last_step is 0 and self.config["train"].get("initialize_from", False):
            filters = self.config["train"]["initialize_from"].get('model_weight_filters', [])
            if len(filters) > 0:
                saver.restore_part(other_model_dir=self.config["train"]["initialize_from"].get('pretrained_model_path',
                                                                                               ''),
                                   filters=filters)
            else:
                pre_model_dir = self.config["train"]["initialize_from"].get('pretrained_model_path', '')
                if os.path.exists(pre_model_dir):
                    last_step, best_val_all_exact = saver.restore(model_dir=pre_model_dir,
                                                                  load_best=True)

                    self.logger.log(
                        "Model initialized from {}".format(pre_model_dir)
                    )
                    self.logger.log("Best step from previous train: "
                                    "{} with best accuracy on val {}".format(last_step, best_val_all_exact))
                    last_step = 0  # for continuous training with specified max_steps
                    if self.config["train"]["initialize_from"].get("reset_dev_accuracy", False):
                        best_val_all_exact = 0.0
                else:
                    raise ValueError(f"train.initialize_from.pretrained_model_path is not valid.")
        else:
            self.logger.log(f"Model restored, the last step is {last_step}, best val_all_exact is {best_val_all_exact}")

        self.logger.log(f"Number of parameters: {self.model.count_parameters()}")

        # 3. Get training data somewhere
        with self.data_random:
            data_splits = self.config["train"].get("data_splits", None)
            if data_splits is not None:
                self.logger.log(f"Using custom training data splits: {data_splits}.")
            else:
                # data_splits = [
                #     section for section, _ in self.config["data"].items() if "train" in section
                #                                                   and isinstance(self.config["data"][section], dict)
                # ]
                if isinstance(self.config["data"], list):
                    datasets = self.config["data"]
                else:
                    datasets = [self.config["data"]]
                data_splits = [f"{dataset['name']}_train" if 'name' in dataset else "train" for dataset in datasets]

            sampler = None
            if self.config["train"].get('batch_balancing', None) and len(data_splits) > 1:
                self.logger.log(f"Apply batch balancing for each batch from multiple datasets...")
                train_datasets = [self.model_preproc.dataset(split) for split in data_splits]
                dataset_example_count = [len(train_dataset) for train_dataset in train_datasets]
                dataset_weights = 1. / torch.Tensor(dataset_example_count)
                train_ds_ids = []
                for idx, train_dataset in enumerate(train_datasets):
                    train_ds_ids.extend([idx] * len(train_dataset))
                train_sample_weights = [dataset_weights[ds_id] for ds_id in train_ds_ids]
                sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sample_weights,
                                                                         len(train_ds_ids)
                                                                         )
                train_datasets = torch.utils.data.ConcatDataset(
                    datasets=train_datasets)
            else:
                train_datasets = list(itertools.chain.from_iterable(
                    self.model_preproc.dataset(split) for split in data_splits)
                )
            self.logger.log(f"There are {len(train_datasets)} training examples.")

            train_data_loader = self._yield_batches_from_epochs(
                DataLoader(
                    train_datasets,
                    batch_size=self.config["train"]["batch_size"],
                    shuffle=True if sampler is None else False,
                    drop_last=True,
                    collate_fn=lambda x: x,
                    pin_memory=self.config["train"].get('pin_memory', False),
                    num_workers=self.config["train"].get('num_workers', 0),
                    sampler=sampler,
                )
            )

        # loader for train data
        train_eval_data_loader = DataLoader(
            train_datasets,
            batch_size=self.config["train"]["eval_batch_size"],
            collate_fn=lambda x: x,
        )

        # loader for val data
        if isinstance(self.config["data"], list):
            val_datasets = [self.model_preproc.dataset(f"{dataset['name']}_val") for dataset in self.config["data"]]
            val_datasets = [val_dataset for val_dataset in val_datasets if val_dataset is not None]
            val_datasets = list(itertools.chain.from_iterable(val_datasets))
        else:
            dataset_name = self.config["data"].get("name", None)
            if dataset_name is not None:
                val_datasets = self.model_preproc.dataset(f"{dataset_name}_val")
            else:
                val_datasets = self.model_preproc.dataset("val")
        self.logger.log(f"There are {len(val_datasets)} validation examples.")
        val_data_loader = DataLoader(
            val_datasets,
            batch_size=self.config["train"]["eval_batch_size"],
            collate_fn=lambda x: x,
        )

        def _evaluate_model():
            # A model that is not evaluated (if eval_on_val=False, or if step < infer_min_n)
            # is given a performance of 0
            val_all_exact = 0
            if self.config["train"]["eval_on_train"]:
                self.logger.log("Evaluate on the training set")
                self._eval_model(modeldir, last_step, train_eval_data_loader, "train")
            if self.config["train"]["eval_on_val"]:
                self.logger.log("Evaluate on the validation set")
                val_all_exact = self._eval_model(modeldir, last_step, val_data_loader, "val")
            return val_all_exact

        val_all_exact = _evaluate_model()
        saver.save(
            modeldir,
            last_step,
            is_best=val_all_exact > best_val_all_exact,
            best_validation_metric=max(val_all_exact, best_val_all_exact)
        )
        best_val_all_exact = max(val_all_exact, best_val_all_exact)

        # Counter for grad aggregation
        grad_accumulation_counter = 0
        losses = []

        # 4. Start training loop
        with self.data_random:
            self.logger.log("Enter training loop")
            for batch in train_data_loader:
                # Quit if too long
                if last_step >= self.config["train"]["max_steps"]:
                    break

                # Compute and apply gradient
                with self.model_random:
                    with autocast(enabled=self.config["train"]["amp_enabled"]):
                        loss = self.model.compute_loss(batch, debug=self.config["train"].get("debug", False))
                        if torch.isnan(loss).any():
                            continue
                        loss /= self.config["train"]["n_grad_accumulation_steps"]

                scaler.scale(loss).backward()

                losses.append(
                    loss.item() * self.config["train"]["n_grad_accumulation_steps"]
                )

                grad_accumulation_counter += 1
                # Update params every `n_grad_accumulation_steps` times
                if (
                        grad_accumulation_counter
                        % self.config["train"]["n_grad_accumulation_steps"]
                        == 0
                ):
                    # may call unscale_ here to allow clipping unscaled gradients,
                    # see https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    lr_scheduler.update_lr(last_step)
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as e:
                        self.logger.log(
                            f"Caught error '{e}' while updating model. Continuing training ..."
                        )
                    optimizer.zero_grad()

                    last_step += 1
                    # Report metrics
                    cur_loss = np.mean(losses)
                    cur_lrs = [
                        {"name": param_group["name"], "value": param_group["lr"]}
                        for param_group in optimizer.param_groups
                    ]
                    if last_step % self.config["train"]["report_every_n"] == 0:
                        self._log_stats(
                            last_step,
                            "train",
                            {
                                "step": last_step,
                            },
                        )
                        self._log_loss(last_step, cur_loss)
                        self._log_lr(last_step, cur_lrs)
                    # Evaluate model
                    if last_step % self.config["train"]["eval_every_n"] == 0:
                        val_all_exact = _evaluate_model()
                        # Run saver
                        saver.save(
                            modeldir,
                            last_step,
                            is_best=val_all_exact > best_val_all_exact,
                            best_validation_metric=max(val_all_exact, best_val_all_exact)
                        )
                        if val_all_exact > best_val_all_exact:
                            self.logger.log(
                                "Step {}: got better model checkpoint with averaged val_all_exact {}. Saved.".format(
                                    last_step,
                                    val_all_exact
                                )
                            )
                        best_val_all_exact = max(val_all_exact, best_val_all_exact)

                    # Reset the list of losses
                    losses = []

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch

    def _eval_model(self, modeldir, last_step, eval_data_loader, eval_section):
        num_eval_items = self.config["train"]["num_eval_items"]
        stats = collections.defaultdict(float)
        self.model.eval()
        with torch.no_grad():
            for eval_batch in eval_data_loader:
                batch_res = self.model.eval_on_batch(eval_batch)
                for k, v in batch_res.items():
                    stats[k] += v
                if num_eval_items and stats["total"] > num_eval_items:
                    break
        self.model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != "total":
                stats[k] /= stats["total"]
        if "total" in stats:
            del stats["total"]

        all_exact = 0
        if last_step >= self.config["train"]["infer_min_n"]:
            eval_metrics = self._infer(modeldir, last_step, eval_section)
            if len(eval_metrics) == 1:
                metrics = eval_metrics[list(eval_metrics.keys())[0]]
                stats.update(
                    {
                        "{}_exact".format(level): metrics["total_scores"][level]["exact"]
                        for level in LEVELS
                    }
                )
                all_exact = metrics["total_scores"]["all"]["exact"]
            else:  # multiple val datasets
                all_exact = 0.
                for dataset_name, metrics in eval_metrics.items():
                    stats.update(
                        {
                            f"{dataset_name}_{level}_exact": metrics["total_scores"][level]["exact"]
                            for level in LEVELS
                        }
                    )
                    all_exact += metrics["total_scores"]["all"]["exact"]
                # @Vu Hoang: average score for all_exact, maybe better one?
                all_exact /= len(eval_metrics)
        self._log_stats(last_step, eval_section, stats)
        return all_exact

    def _infer(self, modeldir, last_step, eval_section):
        self.logger.log("Inferring...")

        if isinstance(self.config["data"], list):
            datasets = self.config["data"]
        else:
            datasets = [self.config["data"]]

        eval_results = {}
        for dataset in datasets:
            if 'name' in dataset:
                print(f"Inferring the {dataset['name']} dataset...")
                name = dataset['name']
            else:
                name = 'default'

            if eval_section not in dataset:
                continue

            # retrieve original data --> @Vu Hoang: can we do this step only once?
            orig_data = registry.construct("dataset", dataset[eval_section])
            orig_data.sample(sample_size=dataset.get(f'{eval_section}_sample_size', None))

            # get preprocessed data
            if name is 'default':
                preproc_data: List[RATPreprocItem] = self.model_preproc.dataset(eval_section)
            else:
                preproc_data: List[RATPreprocItem] = self.model_preproc.dataset(f"{name}_{eval_section}")

            if preproc_data is None:
                continue

            self.model.eval()
            with torch.no_grad():
                if isinstance(self.model.preproc.transition_system, SpiderTransitionSystem):
                    # Orig data only used with SpiderTransitionSystem
                    assert len(orig_data) == len(preproc_data)
                inferred_lines = list(
                    self._inner_infer(
                        model=self.model,
                        orig_data=orig_data,
                        preproc_data=preproc_data,
                        nproc=self.config["train"]["eval_nproc"],
                        beam_size=self.config["train"]["eval_beam_size"],
                        decode_max_time_step=self.config["train"][
                            "eval_decode_max_time_step"
                        ],
                    )
                )
            self.model.train()

            with open(f"{modeldir}/output-{last_step}-{name}", "w") as output_dst:
                for line in inferred_lines:
                    output_dst.write(line)

            if isinstance(self.model.preproc.transition_system, SpiderTransitionSystem):
                eval_results[name] = evaluate_default(orig_data, load_from_lines(inferred_lines))
            else:
                raise NotImplementedError

        return eval_results

    @classmethod
    def _inner_infer(
            cls, model, orig_data, preproc_data, nproc, beam_size, decode_max_time_step
    ):
        if torch.cuda.is_available():
            cp = parallelizer.CUDAParallelizer(nproc)
        else:
            cp = parallelizer.CPUParallelizer(nproc)
        return cp.parallel_map(
            [
                (
                    functools.partial(
                        cls._parse_single,
                        model,
                        beam_size=beam_size,
                        decode_max_time_step=decode_max_time_step,
                    ),
                    list(enumerate(zip(orig_data, preproc_data))),
                )
            ]
        )

    @staticmethod
    def _parse_single(model, item, beam_size, decode_max_time_step):
        index, (orig_item, preproc_item) = item
        batch = [preproc_item]

        try:
            beams = model.parse(batch, decode_max_time_step, beam_size)

            decoded = []
            for beam in beams:
                asdl_ast = beam.ast
                if isinstance(model.preproc.transition_system, SpiderTransitionSystem):
                    tree = model.preproc.transition_system.ast_to_surface_code(
                        asdl_ast=asdl_ast
                    )
                    inferred_code = model.preproc.transition_system.spider_grammar.unparse(
                        tree=tree, spider_schema=orig_item.spider_schema
                    )
                    inferred_code_readable = ""
                else:
                    raise NotImplementedError

                decoded.append(
                    {
                        "model_output": asdl_ast.pretty(),
                        "inferred_code": inferred_code,
                        "inferred_code_readable": inferred_code_readable,
                        "score": beam.score,
                    }
                )
            result = {
                "index": index,
                "beams": decoded,
            }
        except Exception as e:
            result = {"index": index, "error": str(e), "trace": traceback.format_exc()}
        return json.dumps(result) + "\n"


def main(
        args=None, logdir_suffix: List[str] = None, trainer_class: Type[Trainer] = Trainer
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    parser.add_argument(
        "--preprocess-only",
        default=False,
        action="store_true",
        help="If True, do preprocessing only.",
    )
    parser.add_argument(
        "--force-preprocess",
        default=False,
        action="store_true",
        help="If True, force doing preprocessing even if exists.",
    )
    parser.add_argument(
        "--force-train",
        default=False,
        action="store_true",
        help="If True, force doing training even if the model exists.",
    )
    parser.add_argument(
        "--train-sample-ratio",
        type=int,
        default=0,
        help="if specified, sample this ratio (in percentage) from the training data",
    )
    args = parser.parse_args(args)

    if args.config_args:
        config = json.loads(
            _jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args})
        )
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if logdir_suffix:
        args.logdir = os.path.join(args.logdir, *logdir_suffix)

    if "model_name" in config:
        args.logdir = os.path.join(args.logdir, config["model_name"])

    if args.force_train:
        try:
            logdir_path = pathlib.Path(args.logdir)
            shutil.rmtree(logdir_path, ignore_errors=True)
            os.makedirs(logdir_path, exist_ok=True)
        except OSError as e:
            print("Error: %s : %s" % (args.logdir, e.strerror))

    # Initialize the logger
    reopen_to_flush = config.get("log", {}).get("reopen_to_flush")
    logger = Logger(os.path.join(args.logdir, "log.txt"), reopen_to_flush)
    logger.log("Logging to {}".format(args.logdir))

    preproc_data_path = os.path.join(args.logdir, "data")
    logger.log(f"Overwriting preproc save_path with: {preproc_data_path}")
    config['model']['preproc']['save_path'] = preproc_data_path

    if args.train_sample_ratio:
        config['data']['train_sample_ratio'] = int(args.train_sample_ratio)

    if os.path.exists(preproc_data_path) and not args.force_preprocess:
        logger.log("Skip preprocessing..")
    else:
        logger.log("Run preprocessing...")
        # if isinstance(config["data"], list):
        #     dataset = config["data"][0]
        # else:
        #     dataset = config["data"]
        # sections = [key for key, v in dataset.items() if isinstance(v, dict)]
        # Here, we fixed the section list.
        sections = ['train', 'val', 'test']
        preprocessor = Preprocessor(config)
        preprocessor.preprocess(sections=sections,
                                keep_vocab=config['model']['preproc'].get('keep_vocab', False))

    if args.preprocess_only:
        print("Done preprocessing. No further training.")
    else:
        # Construct trainer and do training
        logger.log("Start training...")
        trainer = trainer_class(logger, config)
        trainer.train(modeldir=args.logdir)


if __name__ == "__main__":
    main()
