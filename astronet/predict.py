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
        model_dir = os.path.join(chkpt_root, str(i+1))
        if not os.path.exists(model_dir):
            break
        checkpts.append(model_dir)
    return checkpts

def batch_predict(models_dir, data_files, nruns, **kwargs):
    model_dirs = load_ensemble(models_dir, nruns)
    ensemble_preds = []
    for model_dir in model_dirs:
        preds, config = predict(model_dir, data_files, **kwargs)
        ensemble_preds.append(preds.set_index('tic_id'))

    agg_preds = {}
    for preds in ensemble_preds:
        for tic_id in preds.index:
            if tic_id not in agg_preds:
                agg_preds[tic_id] = []
            row = preds[preds.index == tic_id]
            pred_v = row.values[0]
            if pred_v[0] >= config.hparams.prediction_threshold:
                agg_preds[tic_id].append('disp_E')
            else:
                agg_preds[tic_id].append(preds.columns[np.argmax(pred_v)])

    labels = ['disp_E', 'disp_N', 'disp_J', 'disp_S', 'disp_B']
    final_preds = []
    for tic_id in list(agg_preds.keys()):
        counts = {l: 0 for l in labels}
        for e in agg_preds[tic_id]:
            counts[e] += 1
        maxcount = max(counts.values())
        counts.update({
            'tic_id': tic_id,
            'maxcount': maxcount,
        })
        final_preds.append(counts)
    final_preds = pd.DataFrame(final_preds).set_index('tic_id')
    return final_preds

def main(_):
    predict(FLAGS.model_dir, FLAGS.data_files, FLAGS.output_file)


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
