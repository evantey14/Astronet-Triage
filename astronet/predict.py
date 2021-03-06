# Copyright 2018 The TensorFlow Authors.
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

"""Generates predictions using a trained model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import os
import sys

from absl import app
import numpy as np
import tensorflow as tf
import pandas as pd

from astronet.astro_cnn_model import input_ds
from astronet.util import config_util


parser = argparse.ArgumentParser()


parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Directory containing a model checkpoint.")

parser.add_argument(
    "--data_files",
    type=str,
    required=True,
    help="Comma-separated list of file patterns matching the TFRecord files.")

parser.add_argument(
    "--output_file",
    type=str,
    default='',
    help="Name of file in which predictions will be saved.")


def predict(model_dir, data_files, output_file=None, legacy=False):
    model = tf.keras.models.load_model(model_dir)
    config = config_util.load_config(model_dir)
    
    if legacy:
        for f in config.inputs.features.values():
            l = getattr(f, 'length', None)
            if l is None:
                f.shape = []
            else:
                f.shape = [l]

    ds = input_ds.build_dataset(
        file_pattern=data_files,
        input_config=config.inputs,
        batch_size=1,
        include_labels=False,
        shuffle_filenames=False,
        repeat=1,
        include_identifiers=True)
    
    label_index = {i:k for i, k in enumerate(config.inputs.label_columns)}
    print('Binary prediction threshold: {} (orientative)'.format(
        config.hparams.prediction_threshold))

    print('0 records', end='')
    series = []
    for features, identifiers in ds:
      preds = model(features)

      row = {}
      row['tic_id'] = identifiers.numpy().item()
      for i, p in enumerate(preds.numpy()[0]):
        row[label_index[i]] = p

      series.append(row)
      print('\r{} records'.format(len(series)), end='')

    results = pd.DataFrame.from_dict(series)
    
    if output_file:
      with tf.io.gfile.GFile(output_file, "w") as f:
        results.to_csv(f)
        
    return results, config

def load_ensemble(chkpt_root, nruns):
    checkpts = []
    for i in range(nruns):
        parent = os.path.join(chkpt_root, str(i + 1))
        if not os.path.exists(parent):
            break
        all_dirs = os.listdir(parent)
        if not all_dirs:
            break
        d, = all_dirs
        checkpts.append(os.path.join(parent, d))
    return checkpts

def batch_predict(models_dir, data_files, nruns, num_procs=1, **kwargs):
    model_dirs = load_ensemble(models_dir, nruns)
    ensemble_preds = []
    if num_procs == 1:
        for model_dir in model_dirs:
            preds, _ = predict(model_dir, data_files, **kwargs)
            ensemble_preds.append(preds)
    else:
        with multiprocessing.Pool(num_procs) as pool:
            ensemble_preds_cfgs = pool.starmap(
                predict,
                [(model_dir, data_files, kwargs) for model_dir in model_dirs]
            )
            ensemble_preds = [pred for pred, _ in ensemble_preds_cfgs]
    for i, pred in enumerate(ensemble_preds):
        pred["model_no"] = i

    return pd.concat(ensemble_preds, ignore_index=True)

def main(_):
    predict(FLAGS.model_dir, FLAGS.data_files, FLAGS.output_file)


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
