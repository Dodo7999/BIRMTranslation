# coding=utf-8
# Copyright 2023 The HuggingFace Datasets Authors and Ilya Gusev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""RuLM: a dataset for Russian language modeling"""

import os
import io

import zstandard
import jsonlines
import datasets

try:
    import simdjson
    parser = simdjson.Parser()
    def parse_json(x):
        try:
            return parser.parse(x).as_dict()
        except ValueError:
            return
except ImportError:
    import json
    def parse_json(x):
        return json.loads(x)



_TRAIN_SPLITS = 10
_DESCRIPTION = "Dataset for Russian language modeling"
_URLS = {
    "train": ["train/{}.jsonl.zst".format(str(i).zfill(2)) for i in range(_TRAIN_SPLITS)],
    "validation": "validation.jsonl.zst",
    "test": "test.jsonl.zst"
}
_TEXT = "text"


class RuLMDataset(datasets.GeneratorBasedBuilder):
    """RuLM Dataset"""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description=""),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "meta": {
                    "source": datasets.Value("string"),
                    "url": datasets.Value("string")
                }
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=(_TEXT,)
        )

    def _split_generators(self, dl_manager):
        print(_URLS)
        downloaded_files = dl_manager.download(_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"paths": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"paths": downloaded_files["test"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"paths": downloaded_files["validation"]}),
        ]

    def _generate_examples(self, paths):
        if not isinstance(paths, list):
            paths = [paths]
        for filename in paths:
            with open(filename, "rb") as f:
                cctx = zstandard.ZstdDecompressor()
                reader_stream = io.BufferedReader(cctx.stream_reader(f))
                reader = jsonlines.Reader(reader_stream, loads=parse_json)
                for id_, item in enumerate(reader):
                    yield id_, item
