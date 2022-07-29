import argparse
import json
import _jsonnet
import tqdm

# noinspection PyUnresolvedReferences
from duorat import datasets

# noinspection PyUnresolvedReferences
from duorat.preproc import offline, utils

# noinspection PyUnresolvedReferences
from duorat.utils import schema_linker

# noinspection PyUnresolvedReferences
from duorat.asdl.lang import spider

from duorat.utils import registry


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preproc = registry.construct(
            "preproc", self.config["model"]["preproc"],
        )

    def preprocess(self, sections, keep_vocab):
        self.model_preproc.clear_items()

        if isinstance(self.config["data"], list):
            datasets = self.config["data"]
        else:
            datasets = [self.config["data"]]

        for dataset in datasets:
            if 'name' in dataset:
                print(f"Processing the {dataset['name']} dataset...")
            for section in sections:
                if section not in dataset:
                    continue

                print(f"Section: '{section}' in the {dataset['name'] if 'name' in dataset else 'given'} dataset")

                data = registry.construct("dataset",
                                          dataset[section])  # SpiderDataset/SparcDataset/CoSQLDataset

                sample_size = dataset.get(f'{section}_sample_size', None)
                if section in 'train_sample_ratio' and 'train_sample_ratio' in dataset:
                    sample_size = int(len(data) * float(dataset['train_sample_ratio'] / 100))
                data.sample(sample_size=sample_size)

                real_section = section
                if 'name' in dataset:
                    real_section = f"{dataset['name']}_{real_section}"

                for i, item in enumerate(
                        tqdm.tqdm(data, desc=real_section, dynamic_ncols=True)):  # SpiderItem/SparcItem

                    item.type = dataset.get('type', 'original')

                    to_add, validation_info = self.model_preproc.validate_item(
                        item, real_section
                    )
                    if to_add:
                        self.model_preproc.add_item(item, real_section, validation_info)
        if keep_vocab:
            self.model_preproc.save_examples()
        else:
            self.model_preproc.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    parser.add_argument("--sections", nargs='+', default=None,
                        help="Preprocess only the listed sections")
    parser.add_argument("--keep-vocab", action='store_true',
                        help="Keep existing vocabulary files")
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(
            _jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args})
        )
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    sections = args.sections if args.sections is not None else config["data"].keys()

    preprocessor = Preprocessor(config)
    preprocessor.preprocess(sections, args.keep_vocab)


if __name__ == "__main__":
    main()
