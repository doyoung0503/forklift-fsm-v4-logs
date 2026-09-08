"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const {
  CONFIG, SIM_CONFIG, CalibrationFSM, IdealPlant, fwdSecFromOffset,
  insertionSeconds, timeFromDistancePiecewise, rawMotionVelocityAtTime,
  distanceFromDurationPiecewise,
  computePalletFrontVisibility, forkPalletCollisionAtPose
} = require("../original_fsm_core.js");

const DT = 1 / 15;

function runIdeal(initial, maxFrames = 1200) {
  const fsm = new CalibrationFSM();
  const plant = new IdealPlant({ detectedLength: 1.10, ...initial }, fsm.cfg);
  const states = new Set();
  const commands = new Set();
  for (let frame = 0; frame < maxFrames && fsm.topState !== "DONE"; frame += 1) {
    const snapshot = fsm.step(plant.sensors(), DT);
    states.add(snapshot.statePath);
    commands.add(snapshot.command);
    plant.advance(snapshot.command, DT, fsm);
  }
  assert.equal(fsm.topState, "DONE", `scenario did not finish: ${fsm.statePath}`);
  return { fsm, plant, states, commands };
}

function assertSeen(result, expected) {
  expected.forEach((state) => assert(result.states.has(state), `missing state ${state}`));
}

assert.deepEqual(
  [CONFIG.ALIGN_DIST_M, CONFIG.ALIGN_BAND_M, CONFIG.YAW_TOL_DEG, CONFIG.OFF_TOL_M,
    CONFIG.CMD_STABLE_THR, CONFIG.REL_YAW_TARGET_DEG, CONFIG.STOP_SEC],
  [2.20, 0.30, 2.0, 0.12, 5, 85.0, 1.2]
);
assert.equal(CONFIG.WIDTH_MIN_FULL, 0.0);
assert.equal(CONFIG.PALLET_POCKET_M, 0.0);
assert.equal(SIM_CONFIG.PALLET_FRONT_WIDTH_M, 1.10);
assert.equal(SIM_CONFIG.PALLET_DEPTH_M, 1.30);
assert.equal(SIM_CONFIG.CAMERA_HFOV_DEG, 69.4);
assert.equal(SIM_CONFIG.DETECTION_VISIBLE_FRACTION, 0.50);
assert.equal(SIM_CONFIG.GRID_SIZE_M, 1.00);
assert.deepEqual(
  [SIM_CONFIG.PALLET_POCKET_WIDTH_M, SIM_CONFIG.PALLET_POCKET_INNER_GAP_M,
    SIM_CONFIG.FORK_OUTER_SPAN_M, SIM_CONFIG.FORK_TINE_WIDTH_M],
  [0.25, 0.30, 0.60, 0.10]
);
assert(Math.abs(fwdSecFromOffset(0.42) - 3.4057) < 0.001);
assert.equal(insertionSeconds(2.20), 8.8);

const profileDuration = timeFromDistancePiecewise(0.60);
assert(profileDuration > 0);

// The plant/graph must use the original fixed-effort duration→distance fit directly.
assert.equal(distanceFromDurationPiecewise(0), 0);
assert(distanceFromDurationPiecewise(4) < distanceFromDurationPiecewise(10));
assert(Math.abs(distanceFromDurationPiecewise(10) - 2.4128547369927) < 1e-9);
assert(Math.abs(rawMotionVelocityAtTime(10) - CONFIG.FWD_A * CONFIG.FWD_T1) < 1e-12);

const fixedEffortPlant = new IdealPlant({ distZ: 5, yaw: 0, offsetX: 0 });
fixedEffortPlant.advance("FWD", 2, {
  alignSub: "ALIGN_FWD_ADJUST", statePath: "ALIGN.ALIGN_FWD_ADJUST", now: 2
});
assert.equal(fixedEffortPlant.motionTelemetry().plannedDuration, null);
assert(Math.abs(fixedEffortPlant.motionTelemetry().distance - distanceFromDurationPiecewise(2)) < 1e-12);

const deterministicEstimationModel = {
  sample(variable, groundTruth) {
    const error = { forward: 0.10, lateral: -0.02, yaw: 1.25 }[variable];
    return {
      error, groundTruth, sourceCondition: groundTruth, distribution: "test",
      lowerCondition: groundTruth, upperCondition: groundTruth, weightUpper: 0
    };
  }
};
const estimatedPlant = new IdealPlant({
  distZ: 2.0, yaw: 5, offsetX: 0.3,
  estimationErrorEnabled: true,
  estimationErrorModel: deterministicEstimationModel
});
const estimatedSensors = estimatedPlant.sampleObservation();
assert(Math.abs(estimatedSensors.distZ - 2.10) < 1e-12);
assert(Math.abs(estimatedSensors.offsetX - 0.28) < 1e-12);
assert(Math.abs(estimatedSensors.yaw - 6.25) < 1e-12);
assert.equal(estimatedPlant.sensors().distZ, estimatedSensors.distZ, "sensor reads must reuse one frame sample");
assert.equal(estimatedPlant.estimationErrorTelemetry().sampleIndex, 1);
estimatedPlant.advance("STOP", DT, { alignSub: "IDLE", statePath: "SEARCH", now: DT });
estimatedPlant.sensors();
assert.equal(estimatedPlant.estimationErrorTelemetry().sampleIndex, 2, "plant advance must invalidate the observation");

const visibilityOptions = {
  frontStart: { x: -0.55, y: 0 }, frontEnd: { x: 0.55, y: 0 },
  outwardNormal: { x: 0, y: -1 }, rangeM: 3.5, fovDeg: 69.4
};
const fullyVisible = computePalletFrontVisibility({
  ...visibilityOptions, camera: { x: 0, y: -2, headingDeg: 90 }
});
assert(Math.abs(fullyVisible.fraction - 1) < 1e-12);
const halfVisible = computePalletFrontVisibility({
  ...visibilityOptions, camera: { x: 0, y: -2, headingDeg: 90 + 69.4 / 2 }
});
assert(Math.abs(halfVisible.fraction - 0.5) < 1e-9);
const outsideRadius = computePalletFrontVisibility({
  ...visibilityOptions, rangeM: 1.5, camera: { x: 0, y: -2, headingDeg: 90 }
});
assert.equal(outsideRadius.fraction, 0);
const behindFront = computePalletFrontVisibility({
  ...visibilityOptions, camera: { x: 0, y: 2, headingDeg: -90 }
});
assert.equal(behindFront.fraction, 0);

const safeFork = forkPalletCollisionAtPose({
  bodyX: 0, bodyY: 0.8, headingDeg: 90, palletFaceCenter: { x: 0, y: 2.2 }
});
assert.equal(safeFork.inserting, true);
assert.equal(safeFork.detected, false);
assert(Math.abs(safeFork.clearanceM - 0.05) < 1e-9);
const offsetForkCollision = forkPalletCollisionAtPose({
  bodyX: -0.10, bodyY: 0.8, headingDeg: 90, palletFaceCenter: { x: 0, y: 2.2 }
});
assert.equal(offsetForkCollision.detected, true);
assert(offsetForkCollision.details.some((detail) => detail.collision));
const yawForkCollision = forkPalletCollisionAtPose({
  bodyX: 0, bodyY: 0.8, headingDeg: 92, palletFaceCenter: { x: 0, y: 2.2 }
});
assert.equal(yawForkCollision.detected, true);
const approachingFork = forkPalletCollisionAtPose({
  bodyX: 0, bodyY: 0, headingDeg: 90, palletFaceCenter: { x: 0, y: 2.2 }
});
assert.equal(approachingFork.inserting, false);

const fixedForkliftPlant = new IdealPlant({
  fixedForkliftPose: true,
  bodyX: 0,
  bodyY: -1.15,
  headingDeg: 90,
  distZ: 2.0,
  offsetX: 0.30,
  yaw: 15
});
assert.deepEqual(
  [fixedForkliftPlant.bodyX, fixedForkliftPlant.bodyY, fixedForkliftPlant.headingDeg],
  [0, -1.15, 90]
);
assert(Math.abs(fixedForkliftPlant.distZ - 2.0) < 1e-12);
assert(Math.abs(fixedForkliftPlant.offsetX - 0.30) < 1e-12);
assert(Math.abs(fixedForkliftPlant.yaw - 15) < 1e-12);
const fixedFront = fixedForkliftPlant.palletFrontGeometry();
assert(Math.abs(Math.hypot(
  fixedFront.end.x - fixedFront.start.x,
  fixedFront.end.y - fixedFront.start.y
) - SIM_CONFIG.PALLET_FRONT_WIDTH_M) < 1e-12);

const rotatePoint = ({ x, y }, degrees) => {
  const radians = degrees * Math.PI / 180;
  return {
    x: x * Math.cos(radians) - y * Math.sin(radians),
    y: x * Math.sin(radians) + y * Math.cos(radians)
  };
};
const rotatedBody = rotatePoint({ x: 0, y: 0.8 }, 30);
const rotatedFace = rotatePoint({ x: 0, y: 2.2 }, 30);
const rotatedSafeFork = forkPalletCollisionAtPose({
  bodyX: rotatedBody.x,
  bodyY: rotatedBody.y,
  headingDeg: 120,
  palletFaceCenter: rotatedFace,
  palletHeadingDeg: 120
});
assert.equal(rotatedSafeFork.inserting, true);
assert.equal(rotatedSafeFork.detected, false);
assert(Math.abs(rotatedSafeFork.clearanceM - safeFork.clearanceM) < 1e-9);

const nominal = runIdeal({ distZ: 3.10, yaw: 12.0, offsetX: 0.42 });
assertSeen(nominal, [
  "ALIGN.ALIGN_FWD_ADJUST", "ALIGN.ROTATE_RIGHT_UNTIL_YAW_TOL",
  "ALIGN.ALIGN_ROTATE_LEFT", "ALIGN.FORWARD_AFTER_LEFT",
  "ALIGN.ALIGN_ROTATE_RIGHT_90", "ALIGN.INSERT_FORWARD", "DONE"
]);
assert(nominal.plant.distZ >= -0.02 && nominal.plant.distZ < 0.05, "nominal insertion endpoint must remain near the face");
assert(Math.abs(nominal.plant.offsetX) < 0.002);
assert.equal(nominal.plant.collisionTelemetry().detected, false);

const rightOffset = runIdeal({ distZ: 2.20, yaw: 0, offsetX: 0.55 });
assertSeen(rightOffset, [
  "ALIGN.ALIGN_ROTATE_RIGHT", "ALIGN.FORWARD_AFTER_RIGHT", "ALIGN.ALIGN_ROTATE_LEFT_90"
]);

const ninetyDegreeFsm = new CalibrationFSM({ config: { REL_YAW_TARGET_DEG: 90 } });
const ninetyDegreePlant = new IdealPlant(
  { distZ: 2.20, yaw: 0, offsetX: 0.55, rotateRate: 45 },
  ninetyDegreeFsm.cfg
);
let maximumRelativeYaw = 0;
for (let frame = 0; frame < 1200 && ninetyDegreeFsm.topState !== "DONE"; frame += 1) {
  const snapshot = ninetyDegreeFsm.step(ninetyDegreePlant.sensors(), DT);
  ninetyDegreePlant.advance(snapshot.command, DT, ninetyDegreeFsm);
  maximumRelativeYaw = Math.max(maximumRelativeYaw, Math.abs(ninetyDegreePlant.relYaw));
}
assert.equal(ninetyDegreeFsm.topState, "DONE", "90-degree simulator override did not finish");
assert(Math.abs(maximumRelativeYaw - 90) < 1e-9, `offset turn reached ${maximumRelativeYaw}° instead of 90°`);

const leftOffset = runIdeal({ distZ: 2.20, yaw: 0, offsetX: -0.55 });
assertSeen(leftOffset, [
  "ALIGN.ALIGN_ROTATE_LEFT", "ALIGN.FORWARD_AFTER_LEFT", "ALIGN.ALIGN_ROTATE_RIGHT_90"
]);

const near = runIdeal({ distZ: 1.20, yaw: 0, offsetX: 0 });
assertSeen(near, ["ALIGN.ALIGN_BWD_ADJUST", "ALIGN.INSERT_FORWARD"]);

const negativeYaw = runIdeal({ distZ: 2.20, yaw: -18, offsetX: 0 });
assertSeen(negativeYaw, ["ALIGN.ROTATE_LEFT_UNTIL_YAW_TOL"]);

const rotationPlant = new IdealPlant({ distZ: 2.2, yaw: 0, offsetX: 0, rotateRate: 30 });
rotationPlant.advance("ROT_RIGHT", 1.0, { alignSub: "ROTATE_RIGHT_UNTIL_YAW_TOL", statePath: "ALIGN.ROTATE_RIGHT_UNTIL_YAW_TOL", now: 1 });
assert(Math.abs(rotationPlant.yaw + 30) < 1e-9);
assert(Math.abs(rotationPlant.relYaw - 30) < 1e-9);

const clampedInsertPlant = new IdealPlant({ distZ: 2.6, yaw: 0, offsetX: 0 });
clampedInsertPlant.advance("FWD", 10, {
  alignSub: "INSERT_FORWARD", statePath: "ALIGN.INSERT_FORWARD",
  insertSecCached: 10, fwdSecCached: 0, now: 10
});
assert(Math.abs(clampedInsertPlant.motionTelemetry().distance - 2.5) < 1e-12);
assert(Math.abs(clampedInsertPlant.distZ - 0.1) < 1e-12);

const collisionFsm = new CalibrationFSM();
const collisionPlant = new IdealPlant({ distZ: 2.2, yaw: 0, offsetX: 0.10 }, collisionFsm.cfg);
for (let frame = 0; frame < 500 && !collisionPlant.collisionTelemetry().detected; frame += 1) {
  const snapshot = collisionFsm.step(collisionPlant.sensors(), DT);
  collisionPlant.advance(snapshot.command, DT, collisionFsm);
}
assert.equal(collisionPlant.collisionTelemetry().detected, true, "FSM-tolerated offset must trigger fork collision gate");
assert.equal(collisionFsm.alignSub, "INSERT_FORWARD");

function runEndpointYawValidation(firstRotationError = null) {
  let rotationSamples = 0;
  const errorModel = {
    sample(kind, target) {
      const error = kind === "rotation" && rotationSamples++ === 0 && firstRotationError !== null
        ? firstRotationError : 0;
      return {
        error, requestedTarget: target, sourceTarget: target, distribution: "endpoint-test",
        lowerTarget: target, upperTarget: target, weightUpper: 0, bic: 0
      };
    }
  };
  const fsm = new CalibrationFSM({
    config: { REL_YAW_TARGET_DEG: 90, ROTATION_ENDPOINT_VALIDATION: true }
  });
  const plant = new IdealPlant({
    distZ: 2.2, yaw: 15, offsetX: 0, rotateRate: 15,
    driveErrorEnabled: firstRotationError !== null,
    driveErrorModel: errorModel
  }, fsm.cfg);
  for (let frame = 0; frame < 600 && fsm.alignSub !== "OFFSET_CHECK"; frame += 1) {
    const snapshot = fsm.step(plant.sensors(), 1 / 30);
    plant.advance(snapshot.command, 1 / 30, fsm);
  }
  const rotationStarts = plant.driveErrorTelemetry().events.filter(
    (event) => event.kind === "rotation" && event.event === "start"
  );
  return { fsm, plant, rotationStarts };
}

const exactEndpoint = runEndpointYawValidation();
assert.equal(exactEndpoint.fsm.alignSub, "OFFSET_CHECK");
assert(Math.abs(exactEndpoint.plant.yaw) < 1e-9, "zero-error commanded rotation must finish at yaw=0°");
assert.equal(exactEndpoint.rotationStarts.length, 1);

const acceptedEndpointError = runEndpointYawValidation(1.5);
assert(Math.abs(acceptedEndpointError.plant.yaw - 1.5) < 1e-9);
assert.equal(acceptedEndpointError.rotationStarts.length, 1, "±2° residual should continue without realignment");

const rejectedEndpointError = runEndpointYawValidation(2.01);
assert(Math.abs(rejectedEndpointError.plant.yaw) < 1e-9);
assert.equal(rejectedEndpointError.rotationStarts.length, 2, "residual over ±2° must trigger another yaw command");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const simSource = fs.readFileSync(path.join(__dirname, "..", "sim.js"), "utf8");
assert.match(html, /<option value="1" selected>1×<\/option>/);
assert.match(html, /id="timeScale"/);
assert.match(html, /<option value="8">8×<\/option>/);
assert.doesNotMatch(html, /id="timeScale" disabled/);
assert.match(html, /카메라 갱신 주기[^<]*<input type="text" value="30 Hz" disabled>/);
assert.match(html, /테스트 초기 배치 · 팔레트만 이동/);
assert.match(html, /현재 상황/);
assert.doesNotMatch(html, /scenarioSelect|>Preset<|dist_z m|yaw deg|offset_x m/);
for (const candidateId of ["gridYawCandidates", "gridOffsetCandidates", "gridWidthCandidates",
  "gridDistanceCandidates", "gridBandCandidates", "gridStableCandidates"])
  assert.match(html, new RegExp(`id="${candidateId}"`));
assert.doesNotMatch(html, /gridStopCandidates|gridTurnCandidates|정지 대기 시간|좌우 이동용 회전각/);
assert.match(html, /id="rotateSpeed"[^>]*value="15"/);
assert.match(html, /id="fsmDiagram"/);
assert.match(html, /id="runBatchBtn"/);
assert.match(html, /id="batchResults"/);
assert.match(html, /id="trajectoryChart"/);
assert.match(html, /id="startVideoBtn"/);
assert.match(html, /id="driveErrorEnabled"[^>]*checked/);
assert.match(html, /id="driveDistributionChart"/);
assert.match(html, /id="estimationErrorEnabled"[^>]*checked/);
assert.match(html, /id="estimationDistributionChart"/);
assert.match(html, /id="applyConstantsBtn"/);
assert.doesNotMatch(html, /id="batchLog"/);
assert.match(simSource, /data-action="trajectory"/);
assert.match(simSource, /data-action="replay"/);
assert.match(simSource, /data-action="save"/);
assert.doesNotMatch(simSource, /data-action="log"/);
assert.match(html, /drive_error_model\.js\?v=/);
assert.match(html, /estimation_error_model\.js\?v=/);
assert.match(html, /hyperparameter_search\.js\?v=/);
assert.match(html, /id="runGridSearchBtn"/);
assert.match(html, /id="applyBestGridBtn"/);
assert.doesNotMatch(html, /class="canvas-toolbar"/);
assert.doesNotMatch(html, /class="source-badge"/);
assert.doesNotMatch(html, /canSummary|canFrames|검증 모드/i);
assert.doesNotMatch(html, /시뮬레이션 요구사항에 따라|그래프는 목표거리에 맞춘/);
assert.match(simSource, /BATCH_LATERALS_M = Object\.freeze\(\[-0\.30, 0\.00, 0\.30\]\)/);
assert.match(simSource, /BATCH_FORWARDS_M = Object\.freeze\(\[1\.80, 2\.00, 2\.20\]\)/);
assert.match(simSource, /BATCH_YAWS_DEG = Object\.freeze\(\[15, 0, -15\]\)/);
assert.match(simSource, /REL_YAW_TARGET_DEG: 90\.0/);
assert.match(simSource, /ROTATION_ENDPOINT_VALIDATION: true/);
assert.match(simSource, /function currentSimulationFsmConfig\(extra = \{\}\)/);
assert.match(simSource, /function trajectoryPoint\(/);
assert.match(simSource, /class="fsm-flowchart"/);
assert.match(simSource, /projectNormalized/);
assert.doesNotMatch(simSource, /renderCan\(|canSummary|canFrames/i);
assert.doesNotMatch(simSource, /drawMeasurements\(/);
assert.match(simSource, /drawDot\(ctx, worldToCanvas\(app\.plant\.cameraPose\(\)\), 6/);
assert.match(simSource, /new MediaRecorder\(/);
assert.match(simSource, /function batchInitial\(condition, seedOffset = condition\.id \* 1009\)/);
assert.match(simSource, /function runHyperparameterSearch\(/);
assert.match(simSource, /const CAMERA_RATE_HZ = 30/);
assert.match(simSource, /y: \{ x: 0, y: -height \* 0\.42 \}/);
assert.match(simSource, /const origin = \{ x: width \* 0\.19, y: height \* 0\.56 \}/);
assert.match(simSource, /x \+= SIM_CONFIG\.GRID_SIZE_M/);
assert.match(simSource, /y \+= SIM_CONFIG\.GRID_SIZE_M/);
assert.doesNotMatch(simSource, /[xy] \+= 0\.5/);

const recover = new CalibrationFSM();
recover.step({ detOk: true, detectedLength: null, distZ: 2.2, yaw: 0, offsetX: 0, relYaw: 0 }, DT);
recover.step({ detOk: true, detectedLength: null, distZ: 2.2, yaw: 0, offsetX: 0, relYaw: 0 }, DT);
assert.equal(recover.topState, "RECOVER");
for (let i = 0; i < CONFIG.CMD_STABLE_THR; i += 1) {
  recover.step({ detOk: true, detectedLength: null, distZ: 2.2, yaw: 0, offsetX: 0, relYaw: 0 }, DT);
}
assert.equal(recover.recoverSub, "RECOVER_ROTATE_LEFT");
recover.step({ detOk: true, detectedLength: 0, distZ: 2.2, yaw: 0, offsetX: 0, relYaw: 0 }, DT);
assert.equal(recover.topState, "CHECK");
for (let i = 0; i < CONFIG.CMD_STABLE_THR; i += 1) {
  recover.step({ detOk: true, detectedLength: 0, distZ: 2.2, yaw: 0, offsetX: 0, relYaw: 0 }, DT);
}
assert.equal(recover.topState, "DONE");

const loss = new CalibrationFSM();
loss.step({ detOk: true }, DT);
loss.step({ detOk: false }, DT);
assert.equal(loss.topState, "SEARCH");

const commandFsm = new CalibrationFSM();
commandFsm.step({ detOk: false }, DT);
const stopTx = commandFsm.lastTransaction;
assert.equal(stopTx.command, "STOP");
commandFsm.exec("FWD");
assert.equal(commandFsm.lastTransaction.command, "FWD");
const countBeforeDuplicate = commandFsm.commandTransactions.length;
commandFsm.exec("FWD");
assert.equal(commandFsm.commandTransactions.length, countBeforeDuplicate);
commandFsm.exec("ROT_RIGHT");
assert.equal(commandFsm.lastTransaction.command, "ROT_RIGHT");
assert.equal(commandFsm.executor.lastDirection, -1);

console.log("PASS standalone FSM core: FOV/scale/motion/fork collision, FSM branches, command sequencing");
