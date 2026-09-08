"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {
  FittedDriveErrorModel, createSeededRandom, generateExampleRows,
  parseCsv, rowsToCsv, distributionDensity
} = require("../drive_error_model.js");
const { CalibrationFSM, IdealPlant } = require("../original_fsm_core.js");

const csvPath = path.join(__dirname, "..", "example_drive_error_logs.csv");
const csvText = fs.readFileSync(csvPath, "utf8").trim();
const generatedRows = generateExampleRows();
assert.equal(generatedRows.length, 270);
assert.equal(rowsToCsv(generatedRows).trim(), csvText, "example CSV differs from deterministic generator");

const rows = parseCsv(csvText);
const groupedCounts = new Map();
rows.forEach((row) => {
  const key = `${row.kind}:${row.target}`;
  groupedCounts.set(key, (groupedCounts.get(key) || 0) + 1);
});
assert.equal(groupedCounts.size, 9);
groupedCounts.forEach((count) => assert.equal(count, 30));

const model = new FittedDriveErrorModel(rows);
const summary = model.summary();
assert.equal(summary.length, 9);
const selections = model.selectionSummary();
assert.equal(selections.length, 2);
selections.forEach((selection) => {
  const rowsForKind = summary.filter((row) => row.kind === selection.kind);
  assert(rowsForKind.every((row) => row.selected === selection.selected));
  const candidateNames = Object.keys(selection.candidates);
  candidateNames.forEach((name) => {
    const expectedAverage = rowsForKind.reduce((sum, row) => sum + row.candidates[name], 0) / rowsForKind.length;
    assert(Math.abs(selection.candidates[name] - expectedAverage) < 1e-10);
  });
  assert.equal(selection.averageBic, Math.min(...Object.values(selection.candidates)));
});
model.groups.forEach((group) => {
  assert.equal(group.candidates.length, 3);
  assert.equal(group.selected.name, model.kindSelections[group.kind].selected);
  group.candidates.forEach((candidate) => {
    const density = distributionDensity(candidate, group.errors[0]);
    assert(Number.isFinite(density) && density >= 0, `${candidate.name} density is invalid`);
  });
});

const randomA = createSeededRandom(1234);
const randomB = createSeededRandom(1234);
const interpolatedA = model.sample("forward", 1.9, randomA);
const interpolatedB = model.sample("forward", 1.9, randomB);
assert.deepEqual(interpolatedA, interpolatedB, "seeded drive-error sample is not reproducible");
assert.equal(interpolatedA.lowerTarget, 1.8);
assert.equal(interpolatedA.upperTarget, 2.0);
assert(Math.abs(interpolatedA.weightUpper - 0.5) < 1e-12);

const deterministicErrorModel = {
  sample(kind, target) {
    return {
      error: kind === "forward" ? 0.20 : (target > 0 ? 5 : -5),
      requestedTarget: target,
      lowerTarget: target,
      upperTarget: target,
      weightUpper: 0,
      sourceTarget: target,
      distribution: "test",
      bic: 0
    };
  }
};
const simulatorFsm = new CalibrationFSM({ config: { REL_YAW_TARGET_DEG: 90 } });
const errorPlant = new IdealPlant({
  distZ: 5,
  yaw: 0,
  offsetX: 0,
  driveErrorEnabled: true,
  driveErrorModel: deterministicErrorModel,
  driveRandom: createSeededRandom(1)
}, simulatorFsm.cfg);
errorPlant.advance("FWD", 2, {
  alignSub: "ALIGN_FWD_ADJUST", statePath: "ALIGN.ALIGN_FWD_ADJUST", now: 2
});
const motion = errorPlant.motionTelemetry();
assert(motion.driveGain > 1);
assert(motion.actualDistance > motion.distance, "forward drive error was not applied to physical travel");

const rotationPlant = new IdealPlant({
  distZ: 2.2,
  yaw: 0,
  offsetX: 0.5,
  rotateRate: 45,
  driveErrorEnabled: true,
  driveErrorModel: deterministicErrorModel,
  driveRandom: createSeededRandom(1)
}, simulatorFsm.cfg);
rotationPlant.advance("ROT_LEFT", 1, {
  alignSub: "ALIGN_ROTATE_LEFT", statePath: "ALIGN.ALIGN_ROTATE_LEFT", now: 1
});
assert(Math.abs(rotationPlant.headingDeg - 137.5) < 1e-9, "rotation drive gain did not apply 95/90 scaling");
assert.equal(rotationPlant.driveErrorTelemetry().activeRotation.distribution, "test");

console.log("PASS drive-error example logs, type-level mean-BIC selection, interpolation, deterministic sampling, plant injection");
