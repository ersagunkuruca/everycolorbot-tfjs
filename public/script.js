function toGamma(u) {
  return u < 0.0031308 ? 12.92 * u : 1.055 * Math.pow(u, 1 / 2.4);
}

function toLinear(u) {
  return u < 0.04045 ? u / 12.92 : Math.pow((u + 0.055) / 1.055, 2.4);
}

function fromHexColor(s) {
  return toLinear(parseInt(s, 16) / 255);
}

function toRGBColor(u) {
  return toGamma(u) * 255;
}

function toRGBString(r, g, b) {
  return (
    "rgb(" + toRGBColor(r) + "," + toRGBColor(g) + "," + toRGBColor(b) + ")"
  );
}

async function getData() {
  const colorDataReq = await fetch("/colors");
  const colorData = await colorDataReq.json();
  const cleaned = colorData.map(c => ({
    r: fromHexColor(c.color.substring(2, 4)),
    g: fromHexColor(c.color.substring(4, 6)),
    b: fromHexColor(c.color.substring(6, 8)),
    favs: c.favs,
    rts: c.rts
  }));
  //.filter(car => car.mpg != null && car.horsepower != null);
  return cleaned;
}

async function run() {
  const data = await getData();
  const values = data.map(d => ({
    x: d.r,
    y: d.rts
  }));

  tfvis.render.scatterplot(
    { name: "Red vs RT Count" },
    { values },
    {
      xLabel: "Red",
      yLabel: "RT Count",
      height: 300
    }
  );

  // Create the model
  let model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, outputs } = tensorData;

  testModel(model, data, inputs, tensorData);
  try {
    model = await tf.loadLayersModel("localstorage://my-model");
  } catch (e) {}
  testModel(model, data, inputs, tensorData);
  createTester(model);

  // Train the model
  for (let i = 0; i < 100; i++) {
    await trainModel(model, inputs, outputs);

    testModel(model, data, inputs, tensorData);
  }
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single hidden layer
  model.add(
    tf.layers.dense({
      inputShape: [3],
      units: 50,
      useBias: true,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l1({ l1: 0.0001 })
    })
  );
  //model.add(tf.layers.batchNormalization());

  model.add(
    tf.layers.dense({
      units: 50,
      useBias: true,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l1({ l1: 0.0001 })
    })
  );
  //model.add(tf.layers.batchNormalization());
  // Add an output layer
  model.add(
    tf.layers.dense({
      units: 2,
      useBias: true,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l1({ l1: 0.0001 })
    })
  );

  return model;
}

function createTester(model) {
  document.body.style.backgroundColor = "black";
  const createTesterButton = document.createElement("button");
  document.body.appendChild(createTesterButton);
  createTesterButton.innerText = "Get Good Colors";
  createTesterButton.onclick = function() {
    const bestColors = createDivWithId("bestColors");
    bestColors.style.display = "flex";
    bestColors.style.flexWrap = "wrap";
    bestColors.style.width = "1000px";
    const randomColors = tf.tidy(() => {
      const input = tf.randomUniform([200, 3]).pow(2.2);
      const output = model.predict(input);

      return [input.arraySync(), output.arraySync()];
    });

    const sorted = randomColors[0]
      .map(function(e, i) {
        return [e, randomColors[1][i]];
      })
      .sort(function(a, b) {
        return b[1][0] - a[1][0];
      });
    for (let i = 0; i < sorted.length; i++) {
      let div = document.createElement("DIV");
      const inn = sorted[i][0];
      //const out = sorted[i][1];
      div.style.backgroundColor = toRGBString(inn[0], inn[1], inn[2]);
      div.style.height = "100px";
      div.style.width = "100px";
      div.style.borderRadius = "500px";
      div.title = sorted[i][1][0];
      bestColors.appendChild(div);
    }
  };
}

function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => [d.r, d.g, d.b]);
    const outputs = data.map(d => [d.favs, d.rts]);

    const inputTensor = tf.tensor2d(inputs);
    const outputTensor = tf.tensor2d(outputs).log1p();

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const outputMax = outputTensor.max();
    const outputMin = outputTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedOutputs = outputTensor
      .sub(outputMin)
      .div(outputMax.sub(outputMin));
    return {
      inputs: normalizedInputs,
      outputs: normalizedOutputs,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      outputMax,
      outputMin
    };
  });
}
let compiled = false;
async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  if (!compiled) {
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ["mse"]
    });
    compiled = true;
  }

  const batchSize = 16384;
  const epochs = 200;

  const a = await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    )
  });
  const b = await model.save("localstorage://my-model");
  return [a, b];
}

function testModel(model, data, inputData, normalizationData) {
  const { inputMax, inputMin, outputMin, outputMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    //const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(inputData);

    const unNormXs = inputData.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds
      .mul(outputMax.sub(outputMin))
      .add(outputMin)
      .exp()
      .sub(1);
    // Un-normalize the data
    return [unNormXs.arraySync(), unNormPreds.arraySync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val[0], y: preds[i][0] };
  });
  /*
  let examples = createDivWithId("examples");

  examples.style.display = "flex";
  examples.style.flexWrap = "wrap";

  for (var i = 0; i < 10; i++) {
    let j = Math.floor(Math.random() * xs.length);
    let { r, g, b, rts, favs } = data[j];
    let [predFavs, predRts] = preds[j];

    const color = document.createElement("DIV");
    color.style.backgroundColor = toRGBString(r, g, b);
    color.style.width = favs + "px";
    color.style.height = predFavs + "px";
    //color.innerText ="RTS: " +rts +" Predicted: " +      predRts +      " Favs: " +      favs +      " Predicted: " +      predFavs;
    examples.appendChild(color);
  }
*/
  const originalPoints = data.map(d => ({
    x: d.r,
    y: d.favs
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"]
    },
    {
      xLabel: "Red",
      yLabel: "Favs",
      height: 300
    }
  );
}

function createDivWithId(id, parent) {
  parent = parent || document.body;
  let div = document.getElementById(id);
  if (div == null) {
    div = document.createElement("DIV");
    div.id = id;

    parent.appendChild(div);
  } else {
    div.innerHTML = "";
  }
  return div;
}

document.addEventListener("DOMContentLoaded", run);
