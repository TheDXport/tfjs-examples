/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

// This tiny example illustrates how little code is necessary build /
// train / predict from a model in TensorFlow.js.  Edit this code
// and refresh the index.html to quickly explore the API.

async function run() {
  // Create a simple model with non-linear capabilities.
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 10, activation: "relu", inputShape: [1] })
  );
  model.add(tf.layers.dense({ units: 1 }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  // Generate some synthetic data for training. (y = x^2)
  const xs = tf.tensor2d(
    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [19, 1]
  );
  const ys = tf.tensor2d(
    [
      4, 1, 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225,
      256,
    ],
    [19, 1]
  );

  // Train the model using the data.
  await model.fit(xs, ys, { epochs: 450 });

  document.getElementById("micro-out-div").innerText = model
    .predict(tf.tensor2d([14], [1, 1]))
    .dataSync();
}

run();
