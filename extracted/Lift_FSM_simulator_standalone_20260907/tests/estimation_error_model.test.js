"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {
  FittedEstimationErrorModel,
  generateExampleRows,
  parseCsv,
  rowsToCsv
} = require("../estimation_error_model.js");

const rows = generateExampleRows();
assert.equal(rows.length, 270);
const grouped = new Map();
rows.forEach((row) => {
  const key = `${row.variable}:${Number(row.condition)}`;
  grouped.set(key, (grouped.get(key) || 0) + 1);
});
assert.equal(grouped.size, 9);
grouped.forEach((count) => assert.equal(count, 30));

const csv = rowsToCsv(rows);
assert.equal(parseCsv(csv).length, 270);
const checkedInCsv = fs.readFileSync(path.join(__dirname, "..", "example_estimation_error_logs.csv"), "utf8");
assert.equal(parseCsv(checkedInCsv).length, 270);

const model = new FittedEstimationErrorModel(rows);
assert.equal(model.groups.length, 9);
assert.equal(model.selectionSummary().length, 3);
for (const variable of ["forward", "lateral", "yaw"]) {
  const groups = model.groups.filter((group) => group.variable === variable);
  assert.equal(groups.length, 3);
  assert.equal(new Set(groups.map((group) => group.selected.name)).size, 1,
    `${variable} must use one common distribution family`);
  const selection = model.variableSelections[variable];
  const lowest = Object.entries(selection.candidates).sort((a, b) => a[1] - b[1])[0][0];
  assert.equal(selection.selected, lowest);
}

const fixedRandom = () => 0.25;
const forwardSample = model.sample("forward", 1.9, fixedRandom);
assert.equal(forwardSample.lowerCondition, 1.8);
assert.equal(forwardSample.upperCondition, 2.0);
assert.ok(Math.abs(forwardSample.weightUpper - 0.5) < 1e-12);
const lateralSample = model.sample("lateral", 0.15, fixedRandom);
assert.equal(lateralSample.lowerCondition, 0);
assert.equal(lateralSample.upperCondition, 0.3);
assert.ok(Math.abs(lateralSample.weightUpper - 0.5) < 1e-12);
const yawSample = model.sample("yaw", 7.5, fixedRandom);
assert.equal(yawSample.lowerCondition, 0);
assert.equal(yawSample.upperCondition, 15);
assert.ok(Math.abs(yawSample.weightUpper - 0.5) < 1e-12);

console.log("PASS estimation-error logs, common-family BIC selection, and conditional interpolation");
