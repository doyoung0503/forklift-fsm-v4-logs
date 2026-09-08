"use strict";

const {
  CONFIG, SIM_CONFIG, CalibrationFSM, IdealPlant,
  clamp, computePalletFrontVisibility, distanceFromDurationPiecewise,
  rawMotionVelocityAtTime
} = window.OriginalFSM;
const {
  FittedDriveErrorModel, createSeededRandom, generateExampleRows,
  parseCsv: parseDriveErrorCsv, rowsToCsv: driveErrorRowsToCsv, distributionDensity
} = window.DriveError;
const {
  FittedEstimationErrorModel, generateExampleRows: generateEstimationExampleRows,
  parseCsv: parseEstimationErrorCsv, rowsToCsv: estimationErrorRowsToCsv
} = window.EstimationError;
const {
  parseGridSpec, cartesianProduct, rankEvaluations, formatParameters
} = window.HyperparameterSearch;

const CAMERA_RATE_HZ = 30;
const FRAME_DT = 1 / CAMERA_RATE_HZ;
const DEG = Math.PI / 180;
const TAU = Math.PI * 2;
const DEFAULT_SIMULATION_OVERRIDES = Object.freeze({
  REL_YAW_TARGET_DEG: 90.0,
  ROTATION_ENDPOINT_VALIDATION: true
});
let activeFsmOverrides = { ...DEFAULT_SIMULATION_OVERRIDES };
const BATCH_LATERALS_M = Object.freeze([-0.30, 0.00, 0.30]);
const BATCH_FORWARDS_M = Object.freeze([1.80, 2.00, 2.20]);
const BATCH_YAWS_DEG = Object.freeze([15, 0, -15]);
const BATCH_MAX_SIM_SEC = 120;
const GRID_SEARCH_MAX_COMBINATIONS = 500;
const GRID_SEARCH_MAX_RUNS = 250000;
const EXAMPLE_DRIVE_ERROR_ROWS = generateExampleRows();
const EXAMPLE_ESTIMATION_ERROR_ROWS = generateEstimationExampleRows();

function currentSimulationFsmConfig(extra = {}) {
  return { ...CONFIG, ...activeFsmOverrides, ...extra };
}

function createSimulationFSM(extra = {}) {
  return new CalibrationFSM({ config: currentSimulationFsmConfig(extra) });
}

const DEFAULT_INITIAL = Object.freeze({
  distZ: 3.10,
  yaw: 12.0,
  offsetX: 0.42,
  fixedForkliftPose: true,
  bodyX: 0,
  bodyY: -1.15,
  headingDeg: 90
});

const GRID_CANDIDATE_INPUTS = Object.freeze([
  ["YAW_TOL_DEG", "gridYawCandidates"],
  ["OFF_TOL_M", "gridOffsetCandidates"],
  ["WIDTH_MIN_FULL", "gridWidthCandidates"],
  ["ALIGN_DIST_M", "gridDistanceCandidates"],
  ["ALIGN_BAND_M", "gridBandCandidates"],
  ["CMD_STABLE_THR", "gridStableCandidates"]
]);

const FSM_CONSTANT_FIELDS = Object.freeze([
  { key: "ALIGN_DIST_M", label: "정렬 목표거리 (m)", min: 0.1, max: 10, step: 0.01 },
  { key: "ALIGN_BAND_M", label: "전방 허용범위 (m)", min: 0.001, max: 2, step: 0.01 },
  { key: "YAW_TOL_DEG", label: "각도 허용범위 (°)", min: 0.01, max: 45, step: 0.1 },
  { key: "OFF_TOL_M", label: "좌우 허용범위 (m)", min: 0.001, max: 2, step: 0.01 },
  { key: "WIDTH_MIN_FULL", label: "전면 폭 기준 (m)", min: 0, max: 2, step: 0.01 },
  { key: "CMD_STABLE_THR", label: "연속 판정 횟수", min: 1, max: 300, step: 1, integer: true },
  { key: "REL_YAW_TARGET_DEG", label: "상대 회전 목표 (°)", min: 0, max: 180, step: 1 },
  { key: "STOP_SEC", label: "정지 대기시간 (s)", min: 0, max: 30, step: 0.1 },
  { key: "INSERT_FWD_MPS", label: "삽입 속도 (m/s)", min: 0.01, max: 5, step: 0.01 }
]);

const app = {
  running: false,
  accumulator: 0,
  wallRunElapsed: 0,
  wallScaledElapsed: 0,
  lastAnimationTime: performance.now(),
  fsm: createSimulationFSM(),
  plant: new IdealPlant(DEFAULT_INITIAL, currentSimulationFsmConfig()),
  log: [],
  renderedDriveErrorCount: 0,
  visibility: null,
  cameraRange: SIM_CONFIG.CAMERA_RANGE_M,
  canvas: null,
  ctx: null,
  motionCanvas: null,
  motionCtx: null,
  trajectoryCanvas: null,
  trajectoryCtx: null,
  distributionCanvas: null,
  distributionCtx: null,
  estimationDistributionCanvas: null,
  estimationDistributionCtx: null,
  trajectory: [],
  displayedTrajectory: [],
  trajectoryOrigin: null,
  trajectoryLabel: "현재 실행",
  driveErrorRows: EXAMPLE_DRIVE_ERROR_ROWS,
  driveErrorModel: new FittedDriveErrorModel(EXAMPLE_DRIVE_ERROR_ROWS),
  driveErrorSource: "내장 예제 로그 270행",
  estimationErrorRows: EXAMPLE_ESTIMATION_ERROR_ROWS,
  estimationErrorModel: new FittedEstimationErrorModel(EXAMPLE_ESTIMATION_ERROR_ROWS),
  estimationErrorSource: "내장 예제 로그 270행",
  recording: {
    active: false,
    canvas: null,
    ctx: null,
    recorder: null,
    chunks: [],
    blob: null,
    url: null,
    mimeType: "",
    autoStop: false,
    autoDownload: false,
    filenameBase: "fsm_simulation"
  },
  visitedStates: new Set(),
  collisionLogged: false,
  batchResults: [],
  gridSearch: {
    running: false,
    cancelled: false,
    results: [],
    best: null,
    repeats: 0,
    elapsedMs: 0
  },
  replayConditionId: null,
  replayExpected: null,
  world: { minX: -3.0, maxX: 3.0, minY: -2.0, maxY: 4.2 }
};

const el = {};
function $(id) { return document.getElementById(id); }
function numberValue(input, fallback) {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function bindElements() {
  [
    "statePill", "cmdPill", "safetyPill", "simCanvas", "simClock", "runBtn", "stepBtn", "resetBtn",
    "topState", "subState", "commandReadout", "stableReadout", "interlockReadout",
    "detReadout", "visibilityReadout", "cameraReadout", "distReadout", "yawReadout", "offsetReadout", "relYawReadout",
    "initialDist", "initialYaw", "initialOffset", "rotateSpeed", "cameraRange", "forkTineWidth",
    "fwdApplyDuration", "fwdDistancePrediction",
    "timeScale", "applyScenarioBtn", "constantGrid", "applyConstantsBtn", "constantStatus",
    "motionProfileReadout", "motionTimeReadout",
    "motionVelocityReadout", "motionDistanceReadout", "motionChart",
    "collisionReadout", "clearanceReadout", "forkGeometryReadout", "collisionDetail",
    "fsmDiagram", "diagramCurrent", "logBox",
    "runBatchBtn", "batchSummary", "batchResults",
    "gridYawCandidates", "gridOffsetCandidates", "gridWidthCandidates", "gridDistanceCandidates",
    "gridBandCandidates", "gridStableCandidates",
    "gridSearchRepeats", "runGridSearchBtn", "cancelGridSearchBtn",
    "applyBestGridBtn", "downloadGridResultsBtn", "gridSearchStatus", "gridSearchBest", "gridSearchResults",
    "driveErrorEnabled", "driveErrorSeed", "driveLogFile", "refitExampleBtn", "downloadDriveLogBtn",
    "driveErrorStatus", "driveFitResults", "driveFitCondition", "driveDistributionChart",
    "driveDistributionSummary", "estimationErrorEnabled", "estimationErrorSeed", "estimationLogFile",
    "refitEstimationExampleBtn", "downloadEstimationLogBtn", "estimationErrorStatus",
    "estimationFitResults", "estimationFitCondition", "estimationDistributionChart",
    "estimationDistributionSummary", "trajectoryChart", "trajectorySummary",
    "downloadTrajectoryCsvBtn", "downloadTrajectoryJsonBtn",
    "startVideoBtn", "stopVideoBtn", "downloadVideoBtn", "videoStatus"
  ].forEach((id) => { el[id] = $(id); });
  app.canvas = el.simCanvas;
  app.ctx = app.canvas.getContext("2d");
  app.motionCanvas = el.motionChart;
  app.motionCtx = app.motionCanvas.getContext("2d");
  app.trajectoryCanvas = el.trajectoryChart;
  app.trajectoryCtx = app.trajectoryCanvas.getContext("2d");
  app.distributionCanvas = el.driveDistributionChart;
  app.distributionCtx = app.distributionCanvas.getContext("2d");
  app.estimationDistributionCanvas = el.estimationDistributionChart;
  app.estimationDistributionCtx = app.estimationDistributionCanvas.getContext("2d");
}

function wireEvents() {
  el.runBtn.addEventListener("click", () => {
    app.running = !app.running;
    app.lastAnimationTime = performance.now();
    updateRunButton();
  });
  el.stepBtn.addEventListener("click", () => {
    app.running = false;
    updateRunButton();
    simulateFrame();
    draw();
  });
  el.resetBtn.addEventListener("click", () => {
    app.replayConditionId = null;
    resetSimulation();
  });
  el.applyScenarioBtn.addEventListener("click", () => {
    app.replayConditionId = null;
    resetSimulation();
  });
  el.timeScale.addEventListener("change", () => {
    app.accumulator = 0;
    app.lastAnimationTime = performance.now();
    render();
  });
  el.rotateSpeed.addEventListener("input", () => {
    if (app.plant) app.plant.rotateRate = Math.max(0.1, numberValue(el.rotateSpeed, 15));
    render();
  });
  el.cameraRange.addEventListener("input", () => {
    app.cameraRange = Math.max(0.5, numberValue(el.cameraRange, SIM_CONFIG.CAMERA_RANGE_M));
    updatePerception();
    render();
    draw();
  });
  el.fwdApplyDuration.addEventListener("input", () => {
    renderMotion();
  });
  el.forkTineWidth.addEventListener("input", () => {
    if (app.plant) {
      app.plant.forkTineWidth = clamp(
        numberValue(el.forkTineWidth, SIM_CONFIG.FORK_TINE_WIDTH_M),
        0.02,
        SIM_CONFIG.PALLET_POCKET_WIDTH_M
      );
      app.plant.collision = app.plant.evaluateForkSafety();
    }
    render();
    draw();
  });
  el.runBatchBtn.addEventListener("click", runBatchExperiment);
  el.batchResults.addEventListener("click", handleBatchResultAction);
  el.runGridSearchBtn.addEventListener("click", runHyperparameterSearch);
  el.cancelGridSearchBtn.addEventListener("click", () => { app.gridSearch.cancelled = true; });
  el.applyBestGridBtn.addEventListener("click", applyBestHyperparameters);
  el.downloadGridResultsBtn.addEventListener("click", downloadGridSearchResults);
  el.driveErrorEnabled.addEventListener("change", () => resetSimulation());
  el.driveErrorSeed.addEventListener("change", () => resetSimulation());
  el.refitExampleBtn.addEventListener("click", () => {
    fitDriveErrorRows(EXAMPLE_DRIVE_ERROR_ROWS, "내장 예제 로그 270행");
    resetSimulation();
  });
  el.downloadDriveLogBtn.addEventListener("click", () => {
    downloadTextFile(
      driveErrorRowsToCsv(app.driveErrorRows),
      "text/csv;charset=utf-8",
      "drive_error_logs.csv"
    );
  });
  el.driveLogFile.addEventListener("change", importDriveErrorFile);
  el.driveFitCondition.addEventListener("change", drawDriveDistributionChart);
  el.estimationErrorEnabled.addEventListener("change", () => resetSimulation());
  el.estimationErrorSeed.addEventListener("change", () => resetSimulation());
  el.refitEstimationExampleBtn.addEventListener("click", () => {
    fitEstimationErrorRows(EXAMPLE_ESTIMATION_ERROR_ROWS, "내장 예제 로그 270행");
    resetSimulation();
  });
  el.downloadEstimationLogBtn.addEventListener("click", () => {
    downloadTextFile(
      estimationErrorRowsToCsv(app.estimationErrorRows),
      "text/csv;charset=utf-8",
      "estimation_error_logs.csv"
    );
  });
  el.estimationLogFile.addEventListener("change", importEstimationErrorFile);
  el.estimationFitCondition.addEventListener("change", drawEstimationDistributionChart);
  el.applyConstantsBtn.addEventListener("click", applyFsmConstants);
  el.downloadTrajectoryCsvBtn.addEventListener("click", downloadTrajectoryCsv);
  el.downloadTrajectoryJsonBtn.addEventListener("click", downloadTrajectoryJson);
  el.startVideoBtn.addEventListener("click", () => startVideoRecording());
  el.stopVideoBtn.addEventListener("click", stopVideoRecording);
  el.downloadVideoBtn.addEventListener("click", () => downloadRecordedVideo());
  window.addEventListener("resize", () => {
    draw();
    drawTrajectoryChart();
    drawDriveDistributionChart();
    drawEstimationDistributionChart();
  });
}

function driveErrorInitial(seedOffset = 0) {
  const seed = Math.trunc(numberValue(el.driveErrorSeed, 20260713)) + seedOffset;
  return {
    driveErrorEnabled: Boolean(el.driveErrorEnabled.checked),
    driveErrorModel: app.driveErrorModel,
    driveRandom: createSeededRandom(seed)
  };
}

function estimationErrorInitial(seedOffset = 0) {
  const seed = Math.trunc(numberValue(el.estimationErrorSeed, 20260714)) + seedOffset;
  return {
    estimationErrorEnabled: Boolean(el.estimationErrorEnabled.checked),
    estimationErrorModel: app.estimationErrorModel,
    estimationRandom: createSeededRandom(seed)
  };
}

function initialFromInputs(seedOffset = 0) {
  return {
    detOk: true,
    distZ: numberValue(el.initialDist, 3.10),
    yaw: numberValue(el.initialYaw, 12.0),
    offsetX: numberValue(el.initialOffset, 0.42),
    detectedLength: SIM_CONFIG.PALLET_FRONT_WIDTH_M,
    relYaw: 0,
    rotateRate: Math.max(0.1, numberValue(el.rotateSpeed, 15)),
    forkTineWidth: clamp(
      numberValue(el.forkTineWidth, SIM_CONFIG.FORK_TINE_WIDTH_M),
      0.02,
      SIM_CONFIG.PALLET_POCKET_WIDTH_M
    ),
    fixedForkliftPose: true,
    bodyX: 0,
    bodyY: -1.15,
    headingDeg: 90,
    ...driveErrorInitial(seedOffset),
    ...estimationErrorInitial(seedOffset)
  };
}

function resetSimulation(options = {}) {
  const replaySpec = options.replaySpec || null;
  app.running = false;
  app.accumulator = 0;
  app.wallRunElapsed = 0;
  app.wallScaledElapsed = 0;
  app.fsm = replaySpec
    ? new CalibrationFSM({ config: replaySpec.fsmConfig })
    : createSimulationFSM();
  const initial = replaySpec ? {
    ...replaySpec.initial,
    driveErrorModel: replaySpec.driveErrorModel,
    driveRandom: createSeededRandom(replaySpec.effectiveSeed),
    estimationErrorModel: replaySpec.estimationErrorModel,
    estimationRandom: createSeededRandom(replaySpec.estimationEffectiveSeed)
  } : initialFromInputs();
  app.plant = new IdealPlant(initial, app.fsm.cfg);
  app.cameraRange = replaySpec
    ? replaySpec.cameraRange
    : Math.max(0.5, numberValue(el.cameraRange, SIM_CONFIG.CAMERA_RANGE_M));
  app.replayExpected = options.expectedResult || null;
  app.log = [];
  app.renderedDriveErrorCount = 0;
  app.visitedStates = new Set();
  app.collisionLogged = false;
  updatePerception();
  app.plant.sampleObservation();
  resetTrajectory();
  addLog("FSM reset → SEARCH · 팔레트 초기 배치 적용");
  if (app.replayConditionId !== null) {
    const replaySeed = replaySpec
      ? ` · drive seed ${replaySpec.effectiveSeed} · estimation seed ${replaySpec.estimationEffectiveSeed}`
      : "";
    addLog(`배치 조건 #${String(app.replayConditionId).padStart(2, "0")} 재현 시작 · 실험 스냅샷 복원${replaySeed}`);
  }
  updateRunButton();
  render();
  draw();
}

function updateRunButton() {
  el.runBtn.textContent = app.running ? "Pause" : "Run";
  el.runBtn.classList.toggle("running", app.running);
}

function simulateFrame() {
  updatePerception();
  const snapshot = app.fsm.step(app.plant.sensors(), FRAME_DT);
  snapshot.events.forEach(addLog);
  app.plant.advance(snapshot.command, FRAME_DT, app.fsm);
  updatePerception();
  app.plant.sampleObservation();
  recordTrajectoryPoint();

  const collision = app.plant.collisionTelemetry();
  if (collision.detected && !app.collisionLogged) {
    app.collisionLogged = true;
    app.running = false;
    updateRunButton();
    addLog(`COLLISION · ${collision.reason} · 자동 실행 정지`);
  }

  const driveEvents = app.plant.driveErrorTelemetry().events;
  while (app.renderedDriveErrorCount < driveEvents.length) {
    const event = driveEvents[app.renderedDriveErrorCount++];
    const report = (event.kind === "rotation" && event.event === "start")
      || (event.kind === "forward" && event.event === "finish");
    if (report && event.distribution !== "disabled") addLog(formatDriveErrorEvent(event));
  }
  if (app.fsm.topState === "DONE" && app.running) {
    app.running = false;
    updateRunButton();
    addLog("DONE · 자동 실행 정지");
  }
  const replayEnded = finishReplayIfNeeded(collision);
  if (app.recording.active && app.recording.autoStop
      && (collision.detected || app.fsm.topState === "DONE" || replayEnded)) {
    stopVideoRecording();
  }
  render();
}

function finishReplayIfNeeded(collision) {
  const expected = app.replayExpected;
  if (!expected) return false;
  const expectedTimeout = !expected.success && !expected.collisionDetected;
  const reachedExpectedTimeout = expectedTimeout && app.fsm.now + 1e-9 >= expected.simTime;
  const terminal = collision.detected || app.fsm.topState === "DONE" || reachedExpectedTimeout;
  if (!terminal) return false;

  app.running = false;
  updateRunButton();
  if (reachedExpectedTimeout && !collision.detected && app.fsm.topState !== "DONE") {
    addLog(`REPLAY TIME LIMIT · ${expected.simTime.toFixed(2)}s에서 자동 정지`);
  }
  const actualSuccess = app.fsm.topState === "DONE" && !collision.detected;
  const verdictMatch = actualSuccess === expected.success
    && Boolean(collision.detected) === Boolean(expected.collisionDetected);
  const stateMatch = app.fsm.statePath === expected.finalState;
  const timeDifference = app.fsm.now - expected.simTime;
  const timeMatch = Math.abs(timeDifference) < 1e-6;
  const matched = verdictMatch && stateMatch && timeMatch;
  const conditionLabel = `#${String(expected.id).padStart(2, "0")}`;
  const expectedVerdict = expected.success ? "성공" : "실패";
  const actualVerdict = actualSuccess ? "성공" : "실패";
  addLog(`REPLAY ${matched ? "MATCH" : "MISMATCH"} · 기대 ${expectedVerdict} ${expected.simTime.toFixed(2)}s`
    + ` · 실제 ${actualVerdict} ${app.fsm.now.toFixed(2)}s · Δt=${timeDifference.toFixed(6)}s`);
  el.batchSummary.textContent = `배치 ${conditionLabel} 정확 재현 ${matched ? "일치" : "불일치"}`
    + ` · 판정 ${actualVerdict}/${expectedVerdict}`
    + ` · 시간 ${app.fsm.now.toFixed(2)}s/${expected.simTime.toFixed(2)}s`
    + ` · 상태 ${app.fsm.statePath}/${expected.finalState}`;
  app.replayExpected = null;
  app.replayConditionId = null;
  return true;
}

function palletFrontGeometry() {
  return app.plant.palletFrontGeometry();
}

function updatePerception() {
  if (!app.plant) return;
  app.visibility = updatePlantPerception(app.plant, app.cameraRange);
}

function updatePlantPerception(plant, cameraRange) {
  const front = plant.palletFrontGeometry();
  const visibility = computePalletFrontVisibility({
    camera: plant.cameraPose(),
    frontStart: front.start,
    frontEnd: front.end,
    outwardNormal: front.outwardNormal,
    fovDeg: SIM_CONFIG.CAMERA_HFOV_DEG,
    rangeM: cameraRange
  });
  const detected = visibility.fraction + 1e-9 >= SIM_CONFIG.DETECTION_VISIBLE_FRACTION;
  plant.setPerception(detected, visibility.visibleLength, visibility);
  return visibility;
}

function validateDriveErrorRows(rows) {
  const expected = {
    forward: [1.8, 2.0, 2.2],
    rotation: [-90, -60, -30, 30, 60, 90]
  };
  const groups = new Map();
  rows.forEach((row) => {
    const key = `${row.kind}:${Number(row.target)}`;
    groups.set(key, (groups.get(key) || 0) + 1);
  });
  Object.entries(expected).forEach(([kind, targets]) => {
    targets.forEach((target) => {
      const count = groups.get(`${kind}:${target}`) || 0;
      if (count !== 30) throw new Error(`${kind} ${target} 조건은 30행이어야 합니다(현재 ${count}행).`);
    });
  });
  if (rows.length !== 270) throw new Error(`전체 로그는 270행이어야 합니다(현재 ${rows.length}행).`);
}

function fitDriveErrorRows(rows, sourceLabel) {
  validateDriveErrorRows(rows);
  app.driveErrorRows = rows.map((row) => ({ ...row }));
  app.driveErrorModel = new FittedDriveErrorModel(app.driveErrorRows);
  app.driveErrorSource = sourceLabel;
  renderDriveErrorFit();
}

function renderDriveErrorFit() {
  const summary = app.driveErrorModel.summary();
  const selections = app.driveErrorModel.selectionSummary();
  const labels = { gaussian: "Gaussian", "student-t": "Student-t", "gaussian-kde": "Gaussian KDE" };
  el.driveFitResults.innerHTML = summary.map((row) => {
    const unit = row.kind === "forward" ? "m" : "°";
    const digits = row.kind === "forward" ? 4 : 2;
    return `<tr>
      <td>${row.kind === "forward" ? "전진" : "회전"}</td>
      <td>${row.target.toFixed(row.kind === "forward" ? 1 : 0)}${unit}</td>
      <td>${labels[row.selected]}</td>
      <td>${row.bic.toFixed(1)}</td>
      <td>${row.meanError.toFixed(digits)}${unit}</td>
      <td>${row.stdError.toFixed(digits)}${unit}</td>
    </tr>`;
  }).join("");
  const selectionText = selections.map((selection) =>
    `${selection.kind === "forward" ? "전진" : "회전"} ${labels[selection.selected]}`
      + ` (평균 BIC ${selection.averageBic.toFixed(1)})`).join(" · ");
  el.driveErrorStatus.textContent = `${app.driveErrorSource} 적합 완료 · ${selectionText}`;
  const previousKey = el.driveFitCondition.value;
  el.driveFitCondition.innerHTML = app.driveErrorModel.groups.map((group) => {
    const unit = group.kind === "forward" ? "m" : "°";
    const kindLabel = group.kind === "forward" ? "전진" : "회전";
    return `<option value="${group.key}">${kindLabel} ${group.target}${unit} · n=${group.count}</option>`;
  }).join("");
  if (app.driveErrorModel.groups.some((group) => group.key === previousKey)) {
    el.driveFitCondition.value = previousKey;
  }
  drawDriveDistributionChart();
}

function drawDriveDistributionChart() {
  if (!app.distributionCtx || !app.distributionCanvas || !app.driveErrorModel) return;
  const group = app.driveErrorModel.groups.find((candidate) => candidate.key === el.driveFitCondition.value)
    || app.driveErrorModel.groups[0];
  if (!group) return;
  const canvas = app.distributionCanvas;
  const ctx = app.distributionCtx;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(340, Math.floor(rect.width * dpr));
  const height = Math.max(260, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);

  const rawMin = Math.min(...group.errors);
  const rawMax = Math.max(...group.errors);
  const rawSpan = Math.max(1e-6, rawMax - rawMin);
  const minX = rawMin - rawSpan * 0.22;
  const maxX = rawMax + rawSpan * 0.22;
  const spanX = maxX - minX;
  const binCount = Math.max(6, Math.min(12, Math.round(Math.sqrt(group.errors.length) * 1.5)));
  const binWidth = spanX / binCount;
  const bins = Array(binCount).fill(0);
  group.errors.forEach((value) => {
    const index = Math.max(0, Math.min(binCount - 1, Math.floor((value - minX) / binWidth)));
    bins[index] += 1;
  });
  const histogramDensity = bins.map((count) => count / (group.errors.length * binWidth));
  let maxDensity = Math.max(...histogramDensity, 1e-9);
  group.candidates.forEach((model) => {
    for (let index = 0; index <= 160; index += 1) {
      maxDensity = Math.max(maxDensity, distributionDensity(model, minX + spanX * index / 160));
    }
  });
  maxDensity *= 1.14;

  const margin = { left: 48 * dpr, right: 14 * dpr, top: 68 * dpr, bottom: 34 * dpr };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const px = (value) => margin.left + ((value - minX) / spanX) * plotWidth;
  const py = (density) => margin.top + plotHeight - density / maxDensity * plotHeight;

  ctx.strokeStyle = "#dfe4e9";
  ctx.lineWidth = dpr;
  for (let index = 0; index <= 4; index += 1) {
    const y = margin.top + plotHeight * index / 4;
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
  }
  histogramDensity.forEach((density, index) => {
    const left = px(minX + index * binWidth);
    const right = px(minX + (index + 1) * binWidth);
    const top = py(density);
    ctx.fillStyle = "rgba(119, 139, 158, 0.30)";
    ctx.fillRect(left + dpr, top, Math.max(dpr, right - left - 2 * dpr), margin.top + plotHeight - top);
    ctx.strokeStyle = "rgba(91, 111, 130, 0.55)";
    ctx.strokeRect(left + dpr, top, Math.max(dpr, right - left - 2 * dpr), margin.top + plotHeight - top);
  });

  const styles = {
    gaussian: { color: "#2f74b9", label: "Gaussian" },
    "student-t": { color: "#d86f21", label: "Student-t" },
    "gaussian-kde": { color: "#168d68", label: "Gaussian KDE" }
  };
  group.candidates.forEach((model) => {
    const style = styles[model.name];
    const selected = model.name === group.selected.name;
    ctx.save();
    ctx.strokeStyle = style.color;
    ctx.lineWidth = (selected ? 3 : 1.7) * dpr;
    if (!selected) ctx.setLineDash([5 * dpr, 4 * dpr]);
    ctx.beginPath();
    for (let index = 0; index <= 180; index += 1) {
      const value = minX + spanX * index / 180;
      const x = px(value);
      const y = py(distributionDensity(model, value));
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  });

  ctx.fillStyle = "#34414d";
  ctx.font = `700 ${11 * dpr}px system-ui`;
  ctx.fillText("로그 히스토그램 + 후보 확률밀도", margin.left, 18 * dpr);
  group.candidates.forEach((model, index) => {
    const style = styles[model.name];
    const selected = model.name === group.selected.name;
    const x = margin.left + (index % 2) * 155 * dpr;
    const y = (38 + Math.floor(index / 2) * 17) * dpr;
    ctx.fillStyle = style.color;
    ctx.fillRect(x, y - 7 * dpr, 18 * dpr, 3 * dpr);
    ctx.fillStyle = "#52606d";
    ctx.font = `${10 * dpr}px system-ui`;
    ctx.fillText(`${style.label} · BIC ${model.bic.toFixed(1)}${selected ? " · 선택" : ""}`, x + 23 * dpr, y);
  });
  ctx.fillStyle = "#64717d";
  ctx.font = `${10 * dpr}px system-ui`;
  ctx.fillText(minX.toFixed(group.kind === "forward" ? 3 : 1), margin.left, height - 10 * dpr);
  ctx.fillText(maxX.toFixed(group.kind === "forward" ? 3 : 1), width - margin.right - 36 * dpr, height - 10 * dpr);
  const unit = group.kind === "forward" ? "m" : "°";
  const labels = { gaussian: "Gaussian", "student-t": "Student-t", "gaussian-kde": "Gaussian KDE" };
  const averageBic = app.driveErrorModel.kindSelections[group.kind].candidates[group.selected.name];
  el.driveDistributionSummary.textContent = `${group.kind === "forward" ? "전진" : "회전"} ${group.target}${unit}`
    + ` · 굵은 선은 유형 공통 선택 ${labels[group.selected.name]}`
    + ` · 조건 BIC ${group.selected.bic.toFixed(1)} · 유형 평균 BIC ${averageBic.toFixed(1)}`;
}

function validateEstimationErrorRows(rows) {
  const expected = {
    forward: [1.8, 2.0, 2.2],
    lateral: [-0.30, 0, 0.30],
    yaw: [-15, 0, 15]
  };
  const groups = new Map();
  rows.forEach((row) => {
    const key = `${row.variable}:${Number(row.condition)}`;
    groups.set(key, (groups.get(key) || 0) + 1);
  });
  Object.entries(expected).forEach(([variable, conditions]) => {
    conditions.forEach((condition) => {
      const count = groups.get(`${variable}:${condition}`) || 0;
      if (count !== 30) {
        throw new Error(`${variable} ${condition} 조건은 30행이어야 합니다(현재 ${count}행).`);
      }
    });
  });
  if (rows.length !== 270) throw new Error(`전체 로그는 270행이어야 합니다(현재 ${rows.length}행).`);
}

function fitEstimationErrorRows(rows, sourceLabel) {
  validateEstimationErrorRows(rows);
  app.estimationErrorRows = rows.map((row) => ({ ...row }));
  app.estimationErrorModel = new FittedEstimationErrorModel(app.estimationErrorRows);
  app.estimationErrorSource = sourceLabel;
  renderEstimationErrorFit();
}

function estimationVariableLabel(variable) {
  return { forward: "전방거리", lateral: "좌우위치", yaw: "팔레트각" }[variable] || variable;
}

function renderEstimationErrorFit() {
  const summary = app.estimationErrorModel.summary();
  const selections = app.estimationErrorModel.selectionSummary();
  const labels = { gaussian: "Gaussian", "student-t": "Student-t", "gaussian-kde": "Gaussian KDE" };
  el.estimationFitResults.innerHTML = summary.map((row) => {
    const unit = row.variable === "yaw" ? "°" : "m";
    const digits = row.variable === "yaw" ? 2 : 4;
    const conditionDigits = row.variable === "yaw" ? 0 : 2;
    return `<tr>
      <td>${estimationVariableLabel(row.variable)}</td>
      <td>${row.condition.toFixed(conditionDigits)}${unit}</td>
      <td>${labels[row.selected]}</td>
      <td>${row.bic.toFixed(1)}</td>
      <td>${row.meanError.toFixed(digits)}${unit}</td>
      <td>${row.stdError.toFixed(digits)}${unit}</td>
    </tr>`;
  }).join("");
  const selectionText = selections.map((selection) =>
    `${estimationVariableLabel(selection.variable)} ${labels[selection.selected]}`
      + ` (평균 BIC ${selection.averageBic.toFixed(1)})`).join(" · ");
  el.estimationErrorStatus.textContent = `${app.estimationErrorSource} 적합 완료 · ${selectionText}`;
  const previousKey = el.estimationFitCondition.value;
  el.estimationFitCondition.innerHTML = app.estimationErrorModel.groups.map((group) => {
    const unit = group.variable === "yaw" ? "°" : "m";
    const condition = group.condition.toFixed(group.variable === "yaw" ? 0 : 2);
    return `<option value="${group.key}">${estimationVariableLabel(group.variable)} ${condition}${unit} · n=${group.count}</option>`;
  }).join("");
  if (app.estimationErrorModel.groups.some((group) => group.key === previousKey)) {
    el.estimationFitCondition.value = previousKey;
  }
  drawEstimationDistributionChart();
}

function drawEstimationDistributionChart() {
  const canvas = app.estimationDistributionCanvas;
  const ctx = app.estimationDistributionCtx;
  if (!canvas || !ctx || !app.estimationErrorModel) return;
  const group = app.estimationErrorModel.groups.find(
    (candidate) => candidate.key === el.estimationFitCondition.value
  ) || app.estimationErrorModel.groups[0];
  if (!group) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(340, Math.floor(rect.width * dpr));
  const height = Math.max(260, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);

  const rawMin = Math.min(...group.errors);
  const rawMax = Math.max(...group.errors);
  const rawSpan = Math.max(1e-6, rawMax - rawMin);
  const minX = rawMin - rawSpan * 0.22;
  const maxX = rawMax + rawSpan * 0.22;
  const spanX = maxX - minX;
  const binCount = Math.max(6, Math.min(12, Math.round(Math.sqrt(group.errors.length) * 1.5)));
  const binWidth = spanX / binCount;
  const bins = Array(binCount).fill(0);
  group.errors.forEach((value) => {
    const index = Math.max(0, Math.min(binCount - 1, Math.floor((value - minX) / binWidth)));
    bins[index] += 1;
  });
  const histogramDensity = bins.map((count) => count / (group.errors.length * binWidth));
  let maxDensity = Math.max(...histogramDensity, 1e-9);
  group.candidates.forEach((model) => {
    for (let index = 0; index <= 160; index += 1) {
      maxDensity = Math.max(maxDensity, distributionDensity(model, minX + spanX * index / 160));
    }
  });
  maxDensity *= 1.14;
  const margin = { left: 48 * dpr, right: 14 * dpr, top: 68 * dpr, bottom: 34 * dpr };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const px = (value) => margin.left + ((value - minX) / spanX) * plotWidth;
  const py = (density) => margin.top + plotHeight - density / maxDensity * plotHeight;

  ctx.strokeStyle = "#dfe4e9";
  ctx.lineWidth = dpr;
  for (let index = 0; index <= 4; index += 1) {
    const y = margin.top + plotHeight * index / 4;
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
  }
  histogramDensity.forEach((density, index) => {
    const left = px(minX + index * binWidth);
    const right = px(minX + (index + 1) * binWidth);
    const top = py(density);
    ctx.fillStyle = "rgba(119, 139, 158, 0.30)";
    ctx.fillRect(left + dpr, top, Math.max(dpr, right - left - 2 * dpr), margin.top + plotHeight - top);
    ctx.strokeStyle = "rgba(91, 111, 130, 0.55)";
    ctx.strokeRect(left + dpr, top, Math.max(dpr, right - left - 2 * dpr), margin.top + plotHeight - top);
  });
  const styles = {
    gaussian: { color: "#2f74b9", label: "Gaussian" },
    "student-t": { color: "#d86f21", label: "Student-t" },
    "gaussian-kde": { color: "#168d68", label: "Gaussian KDE" }
  };
  group.candidates.forEach((model) => {
    const style = styles[model.name];
    const selected = model.name === group.selected.name;
    ctx.save();
    ctx.strokeStyle = style.color;
    ctx.lineWidth = (selected ? 3 : 1.7) * dpr;
    if (!selected) ctx.setLineDash([5 * dpr, 4 * dpr]);
    ctx.beginPath();
    for (let index = 0; index <= 180; index += 1) {
      const value = minX + spanX * index / 180;
      const x = px(value);
      const y = py(distributionDensity(model, value));
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  });
  ctx.fillStyle = "#34414d";
  ctx.font = `700 ${11 * dpr}px system-ui`;
  ctx.fillText("추정오차 로그 히스토그램 + 후보 확률밀도", margin.left, 18 * dpr);
  group.candidates.forEach((model, index) => {
    const style = styles[model.name];
    const selected = model.name === group.selected.name;
    const x = margin.left + (index % 2) * 155 * dpr;
    const y = (38 + Math.floor(index / 2) * 17) * dpr;
    ctx.fillStyle = style.color;
    ctx.fillRect(x, y - 7 * dpr, 18 * dpr, 3 * dpr);
    ctx.fillStyle = "#52606d";
    ctx.font = `${10 * dpr}px system-ui`;
    ctx.fillText(`${style.label} · BIC ${model.bic.toFixed(1)}${selected ? " · 선택" : ""}`, x + 23 * dpr, y);
  });
  ctx.fillStyle = "#64717d";
  ctx.font = `${10 * dpr}px system-ui`;
  const digits = group.variable === "yaw" ? 1 : 3;
  ctx.fillText(minX.toFixed(digits), margin.left, height - 10 * dpr);
  ctx.fillText(maxX.toFixed(digits), width - margin.right - 36 * dpr, height - 10 * dpr);
  const unit = group.variable === "yaw" ? "°" : "m";
  const labels = { gaussian: "Gaussian", "student-t": "Student-t", "gaussian-kde": "Gaussian KDE" };
  const averageBic = app.estimationErrorModel.variableSelections[group.variable].candidates[group.selected.name];
  el.estimationDistributionSummary.textContent = `${estimationVariableLabel(group.variable)} ${group.condition}${unit}`
    + ` · 굵은 선은 변수 공통 선택 ${labels[group.selected.name]}`
    + ` · 조건 BIC ${group.selected.bic.toFixed(1)} · 변수 평균 BIC ${averageBic.toFixed(1)}`;
}

function importDriveErrorFile() {
  const file = el.driveLogFile.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const rows = parseDriveErrorCsv(reader.result);
      fitDriveErrorRows(rows, file.name);
      resetSimulation();
    } catch (error) {
      el.driveErrorStatus.textContent = `로그 적합 실패 · ${error.message}`;
    }
  };
  reader.onerror = () => { el.driveErrorStatus.textContent = "로그 파일을 읽지 못했습니다."; };
  reader.readAsText(file);
}

function importEstimationErrorFile() {
  const file = el.estimationLogFile.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const rows = parseEstimationErrorCsv(reader.result);
      fitEstimationErrorRows(rows, file.name);
      resetSimulation();
    } catch (error) {
      el.estimationErrorStatus.textContent = `로그 적합 실패 · ${error.message}`;
    }
  };
  reader.onerror = () => { el.estimationErrorStatus.textContent = "로그 파일을 읽지 못했습니다."; };
  reader.readAsText(file);
}

function formatDriveErrorEvent(event) {
  const unit = event.kind === "forward" ? "m" : "°";
  const base = `DRIVE ERROR · ${event.kind} target=${event.requestedTarget.toFixed(3)}${unit}`
    + ` · sample=${event.error >= 0 ? "+" : ""}${event.error.toFixed(4)}${unit}`
    + ` · ${event.distribution}`;
  if (event.kind === "rotation" && event.endpointMode) {
    return `${base} · actual endpoint=${event.endpointAngleDeg.toFixed(3)}°`;
  }
  return `${base} · gain=${event.gain.toFixed(4)}`;
}

function batchConditions() {
  const conditions = [];
  let id = 1;
  BATCH_LATERALS_M.forEach((lateral) => {
    BATCH_FORWARDS_M.forEach((forward) => {
      BATCH_YAWS_DEG.forEach((yaw) => {
        conditions.push({ id, lateral, forward, yaw });
        id += 1;
      });
    });
  });
  return conditions;
}

function batchInitial(condition, seedOffset = condition.id * 1009) {
  return {
    detOk: true,
    distZ: condition.forward,
    yaw: condition.yaw,
    offsetX: condition.lateral,
    detectedLength: SIM_CONFIG.PALLET_FRONT_WIDTH_M,
    relYaw: 0,
    rotateRate: Math.max(0.1, numberValue(el.rotateSpeed, 15)),
    forkTineWidth: clamp(
      numberValue(el.forkTineWidth, SIM_CONFIG.FORK_TINE_WIDTH_M),
      0.02,
      SIM_CONFIG.PALLET_POCKET_WIDTH_M
    ),
    fixedForkliftPose: true,
    bodyX: 0,
    bodyY: -1.15,
    headingDeg: 90,
    ...driveErrorInitial(seedOffset),
    ...estimationErrorInitial(seedOffset)
  };
}

function batchLogLine(fsm, plant, visibility, label) {
  const truth = plant.groundTruth();
  const observed = plant.sensors();
  const observationText = observed.detOk
    ? ` · estimate z=${observed.distZ.toFixed(3)}m x=${observed.offsetX.toFixed(3)}m yaw=${observed.yaw.toFixed(2)}°`
    : " · estimate=미검출";
  return `${fsm.now.toFixed(2).padStart(6, " ")}s  ${label}`
    + ` · ${fsm.statePath} · ${fsm.command}`
    + ` · truth z=${truth.distZ.toFixed(3)}m x=${truth.offsetX.toFixed(3)}m yaw=${truth.yaw.toFixed(2)}°`
    + observationText
    + ` rel=${truth.relYaw.toFixed(2)}°`
    + ` · visible=${(visibility.fraction * 100).toFixed(1)}%`;
}

function runFastCondition(condition, options = {}) {
  const fsm = createSimulationFSM(options.fsmConfig || {});
  const seedOffset = options.seedOffset ?? condition.id * 1009;
  const captureLogs = options.captureLogs !== false;
  const captureTrajectory = options.captureTrajectory !== false;
  const captureReplay = options.captureReplay !== false;
  const initial = batchInitial(condition, seedOffset);
  const plant = new IdealPlant(initial, fsm.cfg);
  const baseSeed = Math.trunc(numberValue(el.driveErrorSeed, 20260713));
  const effectiveSeed = baseSeed + seedOffset;
  const estimationBaseSeed = Math.trunc(numberValue(el.estimationErrorSeed, 20260714));
  const estimationEffectiveSeed = estimationBaseSeed + seedOffset;
  const {
    driveRandom: _consumedDriveRandom,
    estimationRandom: _consumedEstimationRandom,
    ...initialSnapshot
  } = initial;
  const replaySpec = captureReplay ? {
    baseSeed,
    seedOffset,
    effectiveSeed,
    estimationBaseSeed,
    estimationEffectiveSeed,
    fsmConfig: { ...fsm.cfg },
    initial: { ...initialSnapshot },
    driveErrorModel: initial.driveErrorModel,
    driveErrorSource: app.driveErrorSource,
    estimationErrorModel: initial.estimationErrorModel,
    estimationErrorSource: app.estimationErrorSource,
    cameraRange: app.cameraRange
  } : null;
  let visibility = updatePlantPerception(plant, app.cameraRange);
  plant.sampleObservation();
  const trajectoryOrigin = captureTrajectory ? trajectoryOriginForPlant(plant) : null;
  const trajectory = captureTrajectory ? [trajectoryPoint(fsm, plant, trajectoryOrigin)] : [];
  const logs = captureLogs ? [
    `조건 #${String(condition.id).padStart(2, "0")} · lateral=${condition.lateral.toFixed(2)}m`
      + ` · forward=${condition.forward.toFixed(2)}m · yaw=${condition.yaw.toFixed(0)}°`,
    batchLogLine(fsm, plant, visibility, "초기 상태")
  ] : [];
  let previousState = fsm.statePath;
  let previousCommand = fsm.command;
  let renderedDriveEvents = 0;
  let collision = plant.collisionTelemetry();
  const maxFrames = Math.ceil(BATCH_MAX_SIM_SEC / FRAME_DT);

  for (let frame = 0; frame < maxFrames; frame += 1) {
    visibility = updatePlantPerception(plant, app.cameraRange);
    const snapshot = fsm.step(plant.sensors(), FRAME_DT);
    plant.advance(snapshot.command, FRAME_DT, fsm);
    visibility = updatePlantPerception(plant, app.cameraRange);
    plant.sampleObservation();
    if (captureTrajectory) trajectory.push(trajectoryPoint(fsm, plant, trajectoryOrigin));
    collision = plant.collisionTelemetry();
    const driveEvents = plant.driveErrorTelemetry().events;
    while (renderedDriveEvents < driveEvents.length) {
      const event = driveEvents[renderedDriveEvents++];
      const report = (event.kind === "rotation" && event.event === "start")
        || (event.kind === "forward" && event.event === "finish");
      if (captureLogs && report && event.distribution !== "disabled") {
        logs.push(batchLogLine(fsm, plant, visibility, formatDriveErrorEvent(event)));
      }
    }

    const stateChanged = fsm.statePath !== previousState;
    const commandChanged = fsm.command !== previousCommand;
    if (captureLogs && (stateChanged || commandChanged || snapshot.events.length > 0)) {
      const eventText = snapshot.events.length > 0 ? snapshot.events.join(" | ") : "상태/명령 변경";
      logs.push(batchLogLine(fsm, plant, visibility, eventText));
    }
    previousState = fsm.statePath;
    previousCommand = fsm.command;

    if (collision.detected || fsm.topState === "DONE") break;
  }

  const success = fsm.topState === "DONE" && !collision.detected;
  const reason = collision.detected
    ? `충돌: ${collision.reason}`
    : success ? "삽입 완료(DONE)"
      : `제한시간 ${BATCH_MAX_SIM_SEC}s 초과`;
  if (captureLogs) logs.push(batchLogLine(fsm, plant, visibility, `${success ? "SUCCESS" : "FAIL"} · ${reason}`));
  return {
    ...condition,
    success,
    collisionDetected: collision.detected,
    simTime: fsm.now,
    finalState: fsm.statePath,
    reason,
    logs,
    trajectory,
    driveErrorEvents: plant.driveErrorTelemetry().events,
    replaySpec
  };
}

function renderBatchResults(realElapsedMs) {
  const successCount = app.batchResults.filter((result) => result.success).length;
  const failureCount = app.batchResults.length - successCount;
  el.batchSummary.textContent = `27개 완료 · 성공 ${successCount} · 실패 ${failureCount}`
    + ` · 실제 계산 ${realElapsedMs.toFixed(1)} ms · 화면 애니메이션 없음`;
  el.batchResults.innerHTML = app.batchResults.map((result) => `
    <tr class="${result.success ? "batch-success" : "batch-failure"}">
      <td>${String(result.id).padStart(2, "0")}</td>
      <td>${result.lateral.toFixed(2)}</td>
      <td>${result.forward.toFixed(2)}</td>
      <td>${result.yaw.toFixed(0)}</td>
      <td>${result.success ? "성공" : "실패"}</td>
      <td>${result.simTime.toFixed(2)}s</td>
      <td class="batch-actions">
        <button type="button" data-action="trajectory" data-index="${result.id - 1}">궤적</button>
        <button type="button" data-action="replay" data-index="${result.id - 1}">재현</button>
        <button type="button" data-action="save" data-index="${result.id - 1}">저장</button>
      </td>
    </tr>`).join("");
}

function runBatchExperiment() {
  app.running = false;
  updateRunButton();
  el.runBatchBtn.disabled = true;
  el.runBatchBtn.textContent = "계산 중…";
  const startedAt = performance.now();
  app.batchResults = batchConditions().map((condition) => runFastCondition(condition));
  const realElapsedMs = performance.now() - startedAt;
  renderBatchResults(realElapsedMs);
  el.runBatchBtn.disabled = false;
  el.runBatchBtn.textContent = "27조건 고속 실행";
}

function handleBatchResultAction(event) {
  const action = event.target?.dataset?.action;
  const index = Number(event.target?.dataset?.index);
  if (!action || !Number.isInteger(index) || !app.batchResults[index]) return;
  const result = app.batchResults[index];
  if (action === "trajectory") {
    app.displayedTrajectory = result.trajectory;
    app.trajectoryLabel = `배치 #${String(result.id).padStart(2, "0")} · lat ${result.lateral.toFixed(2)}m`
      + ` · fwd ${result.forward.toFixed(2)}m · yaw ${result.yaw.toFixed(0)}°`;
    drawTrajectoryChart();
    return;
  }
  if (action === "replay") {
    startBatchReplay(result);
    return;
  }
  if (action === "save") {
    startBatchReplay(result, { recordAndDownload: true });
  }
}

function startBatchReplay(result, options = {}) {
  const replaySpec = result.replaySpec;
  if (!replaySpec) {
    el.batchSummary.textContent = `배치 #${String(result.id).padStart(2, "0")} · 정확 재현 스냅샷이 없습니다. 27조건 실험을 다시 실행하세요.`;
    return;
  }
  app.replayConditionId = result.id;
  el.initialOffset.value = result.lateral.toFixed(2);
  el.initialDist.value = result.forward.toFixed(2);
  el.initialYaw.value = result.yaw.toFixed(1);
  el.rotateSpeed.value = String(replaySpec.initial.rotateRate);
  el.cameraRange.value = String(replaySpec.cameraRange);
  el.forkTineWidth.value = String(replaySpec.initial.forkTineWidth);
  el.driveErrorEnabled.checked = Boolean(replaySpec.initial.driveErrorEnabled);
  el.driveErrorSeed.value = String(replaySpec.baseSeed);
  el.estimationErrorEnabled.checked = Boolean(replaySpec.initial.estimationErrorEnabled);
  el.estimationErrorSeed.value = String(replaySpec.estimationBaseSeed);
  activeFsmOverrides = { ...replaySpec.fsmConfig };
  app.driveErrorModel = replaySpec.driveErrorModel;
  app.driveErrorRows = replaySpec.driveErrorModel.rows.map((row) => ({ ...row }));
  app.driveErrorSource = replaySpec.driveErrorSource;
  app.estimationErrorModel = replaySpec.estimationErrorModel;
  app.estimationErrorRows = replaySpec.estimationErrorModel.rows.map((row) => ({ ...row }));
  app.estimationErrorSource = replaySpec.estimationErrorSource;
  renderDriveErrorFit();
  renderEstimationErrorFit();
  resetSimulation({ replaySpec, expectedResult: result });
  initConstants(app.fsm.cfg);
  initStateDiagram(app.fsm.cfg);
  renderStateDiagram();
  el.driveErrorStatus.textContent = `정확 재현 · ${replaySpec.driveErrorSource}`
    + ` · base seed ${replaySpec.baseSeed} + offset ${replaySpec.seedOffset}`
    + ` = ${replaySpec.effectiveSeed}`;
  el.estimationErrorStatus.textContent = `정확 재현 · ${replaySpec.estimationErrorSource}`
    + ` · base seed ${replaySpec.estimationBaseSeed} + offset ${replaySpec.seedOffset}`
    + ` = ${replaySpec.estimationEffectiveSeed}`;
  if (options.recordAndDownload) {
    const conditionName = `fsm_condition_${String(result.id).padStart(2, "0")}`;
    if (!startVideoRecording({ autoStop: true, autoDownload: true, filenameBase: conditionName })) return;
  }
  app.running = true;
  app.lastAnimationTime = performance.now();
  updateRunButton();
  addLog(options.recordAndDownload
    ? "배치 재현 자동 실행 · 종료 시 동영상 자동 저장"
    : "배치 재현 자동 실행");
}

function setGridSearchRunning(running) {
  app.gridSearch.running = running;
  el.runGridSearchBtn.disabled = running;
  el.cancelGridSearchBtn.disabled = !running;
  el.applyBestGridBtn.disabled = running || !app.gridSearch.best;
  el.downloadGridResultsBtn.disabled = running || app.gridSearch.results.length === 0;
}

function renderGridSearchResults() {
  const results = app.gridSearch.results;
  el.gridSearchResults.innerHTML = results.map((result) => `
    <tr class="${result.rank === 1 ? "grid-best-row" : ""}">
      <td>${result.rank}</td>
      <td class="grid-parameter-cell">${escapeHtml(formatParameters(result.parameters))}</td>
      <td>${result.successCount}/${result.totalRuns}<br>${(result.successRate * 100).toFixed(2)}%</td>
      <td>${result.collisionCount}<br>${(result.collisionRate * 100).toFixed(2)}%</td>
      <td>${result.meanTime.toFixed(2)}s</td>
      <td>${Number.isFinite(result.meanSuccessTime) ? `${result.meanSuccessTime.toFixed(2)}s` : "N/A"}</td>
    </tr>`).join("");
  if (!app.gridSearch.best) {
    el.gridSearchBest.textContent = "아직 최적 조합이 없습니다.";
    return;
  }
  const best = app.gridSearch.best;
  el.gridSearchBest.textContent = `최적: ${formatParameters(best.parameters)}`
    + ` · 성공 ${best.successCount}/${best.totalRuns} (${(best.successRate * 100).toFixed(2)}%)`
    + ` · 충돌 ${(best.collisionRate * 100).toFixed(2)}% · 평균 ${best.meanTime.toFixed(2)}s`;
}

function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function gridSearchEntriesFromInputs() {
  const lines = GRID_CANDIDATE_INPUTS.map(([parameter, inputId]) => {
    const values = el[inputId].value.trim();
    return values ? `${parameter}=${values}` : "";
  }).filter(Boolean);
  return parseGridSpec(lines.join("\n"));
}

async function runHyperparameterSearch() {
  if (app.gridSearch.running) return;
  try {
    const entries = gridSearchEntriesFromInputs();
    const combinations = cartesianProduct(entries, GRID_SEARCH_MAX_COMBINATIONS);
    const repeats = Number(el.gridSearchRepeats.value);
    if (!Number.isInteger(repeats) || repeats < 1 || repeats > 1000) {
      throw new Error("조건별 반복 횟수는 1~1000의 정수여야 합니다.");
    }
    const conditions = batchConditions();
    const plannedRuns = combinations.length * conditions.length * repeats;
    if (plannedRuns > GRID_SEARCH_MAX_RUNS) {
      throw new Error(`총 ${plannedRuns.toLocaleString()}회는 제한 ${GRID_SEARCH_MAX_RUNS.toLocaleString()}회를 초과합니다.`);
    }

    app.running = false;
    updateRunButton();
    app.gridSearch = {
      running: true,
      cancelled: false,
      results: [],
      best: null,
      repeats,
      elapsedMs: 0
    };
    setGridSearchRunning(true);
    el.gridSearchResults.innerHTML = "";
    el.gridSearchBest.textContent = "계산 중…";
    const startedAt = performance.now();
    const baseOverrides = { ...activeFsmOverrides };
    let completedRuns = 0;
    const evaluations = [];

    for (let combinationIndex = 0; combinationIndex < combinations.length; combinationIndex += 1) {
      if (app.gridSearch.cancelled) break;
      const parameters = combinations[combinationIndex];
      const fsmConfig = currentSimulationFsmConfig(parameters);
      let successCount = 0;
      let collisionCount = 0;
      let totalTime = 0;
      let successfulTime = 0;
      const failureReasons = {};

      for (let repetition = 0; repetition < repeats; repetition += 1) {
        if (app.gridSearch.cancelled) break;
        for (const condition of conditions) {
          const result = runFastCondition(condition, {
            fsmConfig,
            seedOffset: condition.id * 1009 + repetition * 104729,
            captureLogs: false,
            captureTrajectory: false,
            captureReplay: false
          });
          successCount += result.success ? 1 : 0;
          collisionCount += result.collisionDetected ? 1 : 0;
          totalTime += result.simTime;
          if (result.success) successfulTime += result.simTime;
          else failureReasons[result.reason] = (failureReasons[result.reason] || 0) + 1;
          completedRuns += 1;
        }
        el.gridSearchStatus.textContent = `계산 중 · 조합 ${combinationIndex + 1}/${combinations.length}`
          + ` · ${completedRuns.toLocaleString()}/${plannedRuns.toLocaleString()} runs`;
        await yieldToBrowser();
      }

      if (app.gridSearch.cancelled) break;
      const totalRuns = conditions.length * repeats;
      evaluations.push({
        index: combinationIndex,
        parameters: { ...parameters },
        effectiveOverrides: { ...baseOverrides, ...parameters },
        successCount,
        collisionCount,
        totalRuns,
        successRate: successCount / totalRuns,
        collisionRate: collisionCount / totalRuns,
        meanTime: totalTime / totalRuns,
        meanSuccessTime: successCount > 0 ? successfulTime / successCount : Infinity,
        failureReasons
      });
    }

    app.gridSearch.elapsedMs = performance.now() - startedAt;
    app.gridSearch.results = rankEvaluations(evaluations);
    app.gridSearch.best = app.gridSearch.results[0] || null;
    renderGridSearchResults();
    el.gridSearchStatus.textContent = app.gridSearch.cancelled
      ? `사용자 중단 · 완료 조합 ${evaluations.length}/${combinations.length}`
      : `완료 · ${combinations.length}조합 × 27조건 × ${repeats}회 = ${plannedRuns.toLocaleString()} runs`
        + ` · 계산 ${(app.gridSearch.elapsedMs / 1000).toFixed(2)}s`;
  } catch (error) {
    el.gridSearchStatus.textContent = `그리드 서치 실패 · ${error.message}`;
    el.gridSearchBest.textContent = "입력값을 확인하세요.";
  } finally {
    app.gridSearch.running = false;
    setGridSearchRunning(false);
  }
}

function applyBestHyperparameters() {
  const best = app.gridSearch.best;
  if (!best) return;
  activeFsmOverrides = { ...best.effectiveOverrides };
  initConstants();
  initStateDiagram();
  resetSimulation();
  el.gridSearchBest.textContent = `적용됨: ${formatParameters(best.parameters)}`
    + ` · 이후 일반 실행과 27조건 배치에 사용됩니다.`;
}

function downloadGridSearchResults() {
  if (app.gridSearch.results.length === 0) return;
  const payload = {
    generatedAt: new Date().toISOString(),
    objective: ["maximize successRate", "minimize collisionRate", "minimize meanTime"],
    conditions: {
      lateralM: [...BATCH_LATERALS_M],
      forwardM: [...BATCH_FORWARDS_M],
      yawDeg: [...BATCH_YAWS_DEG]
    },
    repeatsPerCondition: app.gridSearch.repeats,
    driveError: {
      enabled: Boolean(el.driveErrorEnabled.checked),
      seed: Math.trunc(numberValue(el.driveErrorSeed, 20260713)),
      source: app.driveErrorSource
    },
    estimationError: {
      enabled: Boolean(el.estimationErrorEnabled.checked),
      seed: Math.trunc(numberValue(el.estimationErrorSeed, 20260714)),
      source: app.estimationErrorSource
    },
    best: app.gridSearch.best,
    results: app.gridSearch.results
  };
  downloadTextFile(
    JSON.stringify(payload, null, 2),
    "application/json;charset=utf-8",
    "fsm_hyperparameter_grid_search.json"
  );
}

function trajectoryOriginForPlant(plant) {
  const heading = plant.headingDeg * DEG;
  return {
    x: plant.bodyX,
    y: plant.bodyY,
    forward: { x: Math.cos(heading), y: Math.sin(heading) },
    right: { x: Math.sin(heading), y: -Math.cos(heading) }
  };
}

function trajectoryPoint(fsm, plant, origin) {
  const delta = { x: plant.bodyX - origin.x, y: plant.bodyY - origin.y };
  const collision = plant.collisionTelemetry();
  const observed = plant.sensors();
  return {
    time: fsm.now,
    lateral: delta.x * origin.right.x + delta.y * origin.right.y,
    forward: delta.x * origin.forward.x + delta.y * origin.forward.y,
    worldX: plant.bodyX,
    worldY: plant.bodyY,
    headingDeg: plant.headingDeg,
    estimatedForward: observed.distZ,
    estimatedLateral: observed.offsetX,
    estimatedYaw: observed.yaw,
    state: fsm.statePath,
    command: fsm.command,
    collision: collision.detected
  };
}

function resetTrajectory() {
  app.trajectoryOrigin = trajectoryOriginForPlant(app.plant);
  app.trajectory = [trajectoryPoint(app.fsm, app.plant, app.trajectoryOrigin)];
  app.displayedTrajectory = app.trajectory;
  app.trajectoryLabel = "현재 실행";
  drawTrajectoryChart();
}

function recordTrajectoryPoint() {
  const point = trajectoryPoint(app.fsm, app.plant, app.trajectoryOrigin);
  const previous = app.trajectory[app.trajectory.length - 1];
  if (previous && Math.abs(previous.time - point.time) < 1e-9) app.trajectory[app.trajectory.length - 1] = point;
  else app.trajectory.push(point);
  if (app.displayedTrajectory === app.trajectory) drawTrajectoryChart();
}

function trajectoryCsv(points) {
  const header = "time_s,lateral_m,forward_m,world_x_m,world_y_m,heading_deg,estimated_forward_m,estimated_lateral_m,estimated_yaw_deg,state,command,collision";
  const rows = points.map((point) => [
    point.time.toFixed(6), point.lateral.toFixed(6), point.forward.toFixed(6),
    point.worldX.toFixed(6), point.worldY.toFixed(6), point.headingDeg.toFixed(6),
    Number.isFinite(point.estimatedForward) ? point.estimatedForward.toFixed(6) : "",
    Number.isFinite(point.estimatedLateral) ? point.estimatedLateral.toFixed(6) : "",
    Number.isFinite(point.estimatedYaw) ? point.estimatedYaw.toFixed(6) : "",
    point.state, point.command, point.collision ? 1 : 0
  ].join(","));
  return [header, ...rows].join("\n");
}

function downloadTextFile(content, mimeType, filename) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadTrajectoryCsv() {
  downloadTextFile(trajectoryCsv(app.displayedTrajectory), "text/csv;charset=utf-8", "fsm_trajectory.csv");
}

function downloadTrajectoryJson() {
  const payload = {
    label: app.trajectoryLabel,
    coordinateFrame: "initial forklift: lateral/right, forward/heading",
    driveError: {
      enabled: Boolean(el.driveErrorEnabled.checked),
      seed: Math.trunc(numberValue(el.driveErrorSeed, 20260713)),
      source: app.driveErrorSource
    },
    estimationError: {
      enabled: Boolean(el.estimationErrorEnabled.checked),
      seed: Math.trunc(numberValue(el.estimationErrorSeed, 20260714)),
      source: app.estimationErrorSource
    },
    points: app.displayedTrajectory
  };
  downloadTextFile(JSON.stringify(payload, null, 2), "application/json;charset=utf-8", "fsm_trajectory.json");
}

function drawTrajectoryChart() {
  if (!app.trajectoryCtx || !app.trajectoryCanvas || !app.displayedTrajectory.length) return;
  const canvas = app.trajectoryCanvas;
  const context = app.trajectoryCtx;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(420, Math.round(rect.width * dpr));
  const height = Math.max(360, Math.round((rect.height || 440) * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const points = app.displayedTrajectory;
  const times = points.map((point) => point.time);
  const laterals = points.map((point) => point.lateral);
  const forwards = points.map((point) => point.forward);
  const paddedBounds = (values, minimumSpan, includeZero = false) => {
    let minimum = Math.min(...values);
    let maximum = Math.max(...values);
    if (includeZero) {
      minimum = Math.min(0, minimum);
      maximum = Math.max(0, maximum);
    }
    const span = Math.max(minimumSpan, maximum - minimum);
    const center = (minimum + maximum) / 2;
    return { min: center - span * 0.58, max: center + span * 0.58 };
  };
  const lateralBounds = paddedBounds(laterals, 0.20, true);
  const forwardBounds = paddedBounds(forwards, 0.30, true);
  const bounds = {
    timeMin: 0,
    timeMax: Math.max(1, ...times),
    lateralMin: lateralBounds.min,
    lateralMax: lateralBounds.max,
    forwardMin: forwardBounds.min,
    forwardMax: forwardBounds.max
  };
  // Back-left floor corner: x and z remain on the floor, y(time) rises vertically.
  const origin = { x: width * 0.19, y: height * 0.56 };
  const vectors = {
    x: { x: width * 0.56, y: height * 0.07 },
    z: { x: -width * 0.11, y: height * 0.27 },
    y: { x: 0, y: -height * 0.42 }
  };
  const normalize = (value, min, max) => (value - min) / Math.max(1e-9, max - min);
  const projectNormalized = (lateral, forward, time) => ({
    x: origin.x + vectors.x.x * lateral + vectors.z.x * forward + vectors.y.x * time,
    y: origin.y + vectors.x.y * lateral + vectors.z.y * forward + vectors.y.y * time
  });
  const project = (point, timeOverride = null) => {
    const time = normalize(point.time, bounds.timeMin, bounds.timeMax);
    const lateral = normalize(point.lateral, bounds.lateralMin, bounds.lateralMax);
    const forward = normalize(point.forward, bounds.forwardMin, bounds.forwardMax);
    return projectNormalized(lateral, forward, timeOverride === null ? time : timeOverride);
  };
  const strokeSegment = (from, to, color, lineWidth, dash = []) => {
    context.save();
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.setLineDash(dash);
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.restore();
  };
  const gridColor = "#dce3e9";
  const edgeColor = "#9aa7b4";

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfcfd";
  context.fillRect(0, 0, width, height);

  context.fillStyle = "rgba(225, 232, 238, 0.34)";
  context.beginPath();
  [projectNormalized(0, 0, 0), projectNormalized(1, 0, 0), projectNormalized(1, 1, 0), projectNormalized(0, 1, 0)]
    .forEach((point, index) => index === 0 ? context.moveTo(point.x, point.y) : context.lineTo(point.x, point.y));
  context.closePath();
  context.fill();

  for (let index = 0; index <= 4; index += 1) {
    const ratio = index / 4;
    strokeSegment(projectNormalized(ratio, 0, 0), projectNormalized(ratio, 1, 0), gridColor, dpr);
    strokeSegment(projectNormalized(0, ratio, 0), projectNormalized(1, ratio, 0), gridColor, dpr);
    strokeSegment(projectNormalized(0, 0, ratio), projectNormalized(1, 0, ratio), gridColor, dpr);
    strokeSegment(projectNormalized(0, 0, ratio), projectNormalized(0, 1, ratio), gridColor, dpr);
  }

  const corners = {
    o: projectNormalized(0, 0, 0), x: projectNormalized(1, 0, 0), z: projectNormalized(0, 1, 0),
    xz: projectNormalized(1, 1, 0), y: projectNormalized(0, 0, 1),
    xy: projectNormalized(1, 0, 1), zy: projectNormalized(0, 1, 1), xyz: projectNormalized(1, 1, 1)
  };
  [[corners.o, corners.x], [corners.o, corners.z], [corners.o, corners.y], [corners.x, corners.xz],
    [corners.z, corners.xz], [corners.x, corners.xy], [corners.z, corners.zy], [corners.xz, corners.xyz],
    [corners.y, corners.xy], [corners.y, corners.zy], [corners.xy, corners.xyz], [corners.zy, corners.xyz]]
    .forEach(([from, to]) => strokeSegment(from, to, edgeColor, 1.3 * dpr));

  context.fillStyle = "#53606d";
  context.font = `700 ${11 * dpr}px system-ui`;
  context.textAlign = "center";
  context.fillText(`x · lateral ${bounds.lateralMax.toFixed(2)} m`, corners.x.x, corners.x.y + 22 * dpr);
  context.textAlign = "left";
  context.fillText(`z · forward ${bounds.forwardMax.toFixed(2)} m`, corners.z.x - 2 * dpr, corners.z.y + 18 * dpr);
  context.save();
  context.translate(corners.y.x - 18 * dpr, corners.y.y + 4 * dpr);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText(`y · time ${bounds.timeMax.toFixed(1)} s`, 0, 0);
  context.restore();
  context.font = `${10 * dpr}px system-ui`;
  context.fillStyle = "#7a8692";
  context.textAlign = "right";
  context.fillText("origin", corners.o.x - 7 * dpr, corners.o.y + 13 * dpr);

  context.save();
  context.strokeStyle = "rgba(113, 128, 143, 0.55)";
  context.lineWidth = 1.5 * dpr;
  context.setLineDash([5 * dpr, 4 * dpr]);
  context.beginPath();
  points.forEach((point, index) => {
    const projected = project(point, 0);
    if (index === 0) context.moveTo(projected.x, projected.y);
    else context.lineTo(projected.x, projected.y);
  });
  context.stroke();
  context.restore();

  context.strokeStyle = "rgba(255, 255, 255, 0.92)";
  context.lineWidth = 6 * dpr;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.beginPath();
  points.forEach((point, index) => {
    const projected = project(point);
    if (index === 0) context.moveTo(projected.x, projected.y);
    else context.lineTo(projected.x, projected.y);
  });
  context.stroke();

  for (let index = 1; index < points.length; index += 1) {
    const ratio = index / Math.max(1, points.length - 1);
    const red = Math.round(47 + (225 - 47) * ratio);
    const green = Math.round(116 + (116 - 116) * ratio);
    const blue = Math.round(185 + (36 - 185) * ratio);
    strokeSegment(project(points[index - 1]), project(points[index]), `rgb(${red},${green},${blue})`, 3 * dpr);
  }

  const start = project(points[0]);
  const end = project(points[points.length - 1]);
  const endFloor = project(points[points.length - 1], 0);
  strokeSegment(end, endFloor, "rgba(118, 132, 146, 0.55)", dpr, [4 * dpr, 4 * dpr]);
  if (Math.hypot(end.x - start.x, end.y - start.y) < 12 * dpr) {
    drawTrajectoryMarker(context, end, points[points.length - 1].collision ? "#d83939" : "#e17424", dpr, "start / end");
  } else {
    drawTrajectoryMarker(context, start, "#17865c", dpr, "start");
    drawTrajectoryMarker(context, end, points[points.length - 1].collision ? "#d83939" : "#e17424", dpr, "end");
  }

  const last = points[points.length - 1];
  el.trajectorySummary.textContent = `${app.trajectoryLabel} · ${points.length} points · ${last.time.toFixed(2)}s`
    + ` · lateral ${last.lateral.toFixed(3)}m · forward ${last.forward.toFixed(3)}m`;
}

function drawTrajectoryMarker(context, point, color, dpr, label = "") {
  context.beginPath();
  context.arc(point.x, point.y, 5 * dpr, 0, TAU);
  context.fillStyle = color;
  context.fill();
  context.strokeStyle = "#ffffff";
  context.lineWidth = 2 * dpr;
  context.stroke();
  if (label) {
    context.fillStyle = "#3f4b57";
    context.font = `700 ${10 * dpr}px system-ui`;
    context.textAlign = "left";
    context.fillText(label, point.x + 8 * dpr, point.y - 8 * dpr);
  }
}

function supportedVideoMimeType() {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return candidates.find((type) => window.MediaRecorder?.isTypeSupported?.(type)) || "";
}

function startVideoRecording(options = {}) {
  if (app.recording.active) return false;
  if (!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream) {
    el.videoStatus.textContent = "이 브라우저는 Canvas MediaRecorder 녹화를 지원하지 않습니다.";
    return false;
  }
  if (app.recording.url) URL.revokeObjectURL(app.recording.url);
  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 720;
  const context = canvas.getContext("2d");
  const mimeType = supportedVideoMimeType();
  const stream = canvas.captureStream(CAMERA_RATE_HZ);
  const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  app.recording = {
    active: true,
    canvas,
    ctx: context,
    recorder,
    chunks: [],
    blob: null,
    url: null,
    mimeType: recorder.mimeType || mimeType || "video/webm",
    autoStop: Boolean(options.autoStop),
    autoDownload: Boolean(options.autoDownload),
    filenameBase: options.filenameBase || "fsm_simulation"
  };
  recorder.ondataavailable = (event) => {
    if (event.data?.size > 0) app.recording.chunks.push(event.data);
  };
  recorder.onstop = () => {
    const blob = new Blob(app.recording.chunks, { type: app.recording.mimeType });
    app.recording.blob = blob;
    app.recording.url = URL.createObjectURL(blob);
    el.downloadVideoBtn.disabled = false;
    el.videoStatus.textContent = `녹화 완료 · ${(blob.size / 1024 / 1024).toFixed(2)} MB · 다운로드 가능`;
    if (app.recording.autoDownload) downloadRecordedVideo(app.recording.filenameBase);
  };
  drawRecordingFrame();
  recorder.start(500);
  el.startVideoBtn.disabled = true;
  el.stopVideoBtn.disabled = false;
  el.downloadVideoBtn.disabled = true;
  el.videoStatus.textContent = `녹화 중 · 시뮬레이션 화면과 FSM 다이어그램을 ${CAMERA_RATE_HZ} fps로 합성`;
  return true;
}

function stopVideoRecording() {
  if (!app.recording.active || !app.recording.recorder) return;
  app.recording.active = false;
  app.recording.recorder.stop();
  app.recording.recorder.stream.getTracks().forEach((track) => track.stop());
  el.startVideoBtn.disabled = false;
  el.stopVideoBtn.disabled = true;
  el.videoStatus.textContent = "동영상 파일 생성 중…";
}

function downloadRecordedVideo(filenameBase = "fsm_simulation") {
  if (!app.recording.url) return;
  const anchor = document.createElement("a");
  anchor.href = app.recording.url;
  anchor.download = app.recording.mimeType.includes("mp4") ? `${filenameBase}.mp4` : `${filenameBase}.webm`;
  anchor.click();
}

function drawRecordingFrame() {
  if (!app.recording.active || !app.recording.ctx || !app.recording.canvas) return;
  const context = app.recording.ctx;
  const width = app.recording.canvas.width;
  const height = app.recording.canvas.height;
  context.fillStyle = "#f4f6f8";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#20262d";
  context.font = "700 22px system-ui";
  context.fillText("Pallet FSM Simulation", 24, 34);
  context.font = "14px system-ui";
  context.fillStyle = "#5d6874";
  context.fillText(`t=${app.fsm.now.toFixed(2)}s · ${app.fsm.statePath} · ${app.fsm.command}`, 320, 32);

  const left = { x: 20, y: 52, width: 850, height: 648 };
  context.fillStyle = "#ffffff";
  context.fillRect(left.x, left.y, left.width, left.height);
  const sourceRatio = app.canvas.width / Math.max(1, app.canvas.height);
  const targetRatio = left.width / left.height;
  let drawWidth = left.width;
  let drawHeight = left.height;
  if (sourceRatio > targetRatio) drawHeight = drawWidth / sourceRatio;
  else drawWidth = drawHeight * sourceRatio;
  context.drawImage(
    app.canvas,
    left.x + (left.width - drawWidth) / 2,
    left.y + (left.height - drawHeight) / 2,
    drawWidth,
    drawHeight
  );

  const panelX = 892;
  context.fillStyle = "#ffffff";
  context.fillRect(panelX, 52, 368, 648);
  drawVideoFsmFlowchart(context, panelX);
  context.fillStyle = "#5d6874";
  context.font = "13px system-ui";
  context.fillText(
    `drive error: ${el.driveErrorEnabled.checked ? "ON" : "OFF"} · seed ${Math.trunc(numberValue(el.driveErrorSeed, 20260713))}`,
    panelX + 16,
    666
  );
  context.fillText(
    `estimation error: ${el.estimationErrorEnabled.checked ? "ON" : "OFF"} · seed ${Math.trunc(numberValue(el.estimationErrorSeed, 20260714))}`,
    panelX + 16,
    686
  );
}

function drawVideoArrow(context, fromX, fromY, toX, toY) {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  context.save();
  context.strokeStyle = "#a3adb7";
  context.fillStyle = "#a3adb7";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(fromX, fromY);
  context.lineTo(toX, toY);
  context.stroke();
  context.beginPath();
  context.moveTo(toX, toY);
  context.lineTo(toX - 6 * Math.cos(angle - 0.45), toY - 6 * Math.sin(angle - 0.45));
  context.lineTo(toX - 6 * Math.cos(angle + 0.45), toY - 6 * Math.sin(angle + 0.45));
  context.closePath();
  context.fill();
  context.restore();
}

function drawVideoFsmFlowchart(context, panelX) {
  const node = (kind, state, x, y, width, label) => drawVideoStateNode(context, {
    x: panelX + x, y, width, height: 28, label,
    active: kind === "top" ? app.fsm.topState === state
      : app.fsm.topState === "ALIGN" && app.fsm.alignSub === state,
    visited: app.visitedStates.has(`${kind}:${state}`)
  });
  context.fillStyle = "#20262d";
  context.font = "700 14px system-ui";
  context.fillText("TOP FSM", panelX + 14, 76);
  drawVideoArrow(context, panelX + 82, 108, panelX + 94, 108);
  drawVideoArrow(context, panelX + 166, 108, panelX + 178, 108);
  drawVideoArrow(context, panelX + 250, 108, panelX + 266, 108);
  drawVideoArrow(context, panelX + 130, 122, panelX + 130, 151);
  drawVideoArrow(context, panelX + 202, 165, panelX + 218, 165);
  node("top", "SEARCH", 14, 94, 68, "SEARCH");
  node("top", "DETECTED", 94, 94, 72, "DETECTED");
  node("top", "ALIGN", 178, 94, 72, "ALIGN");
  node("top", "DONE", 266, 94, 72, "DONE");
  node("top", "RECOVER", 94, 151, 72, "RECOVER");
  node("top", "CHECK", 218, 151, 72, "CHECK");

  context.fillStyle = "#20262d";
  context.font = "700 14px system-ui";
  context.fillText("ALIGN FSM", panelX + 14, 222);
  drawVideoArrow(context, panelX + 184, 264, panelX + 184, 296);
  drawVideoArrow(context, panelX + 184, 338, panelX + 184, 370);
  drawVideoArrow(context, panelX + 226, 384, panelX + 274, 384);
  drawVideoArrow(context, panelX + 316, 398, panelX + 316, 430);
  drawVideoArrow(context, panelX + 142, 250, panelX + 88, 275);
  drawVideoArrow(context, panelX + 226, 250, panelX + 280, 275);
  drawVideoArrow(context, panelX + 142, 324, panelX + 88, 349);
  drawVideoArrow(context, panelX + 226, 324, panelX + 280, 349);
  node("align", "DIST_CHECK", 142, 236, 84, "DIST CHECK");
  node("align", "ALIGN_FWD_ADJUST", 26, 275, 124, "FWD ADJUST");
  node("align", "ALIGN_BWD_ADJUST", 218, 275, 124, "BWD ADJUST");
  node("align", "YAW_CHECK", 142, 310, 84, "YAW CHECK");
  node("align", "ROTATE_RIGHT_UNTIL_YAW_TOL", 26, 349, 124, "ROT R TARGET");
  node("align", "ROTATE_LEFT_UNTIL_YAW_TOL", 218, 349, 124, "ROT L TARGET");
  node("align", "OFFSET_CHECK", 142, 370, 84, "OFFSET");
  node("align", "INSERT_FORWARD", 274, 370, 84, "INSERT");
  node("align", "READY_TO_DONE", 274, 430, 84, "READY");

  const chainY = [490, 550];
  chainY.forEach((y) => {
    drawVideoArrow(context, panelX + 106, y + 14, panelX + 132, y + 14);
    drawVideoArrow(context, panelX + 220, y + 14, panelX + 246, y + 14);
  });
  node("align", "ALIGN_ROTATE_RIGHT", 18, chainY[0], 88, "ROT R 90°");
  node("align", "FORWARD_AFTER_RIGHT", 132, chainY[0], 88, "FWD");
  node("align", "ALIGN_ROTATE_LEFT_90", 246, chainY[0], 104, "ROT L 90°");
  node("align", "ALIGN_ROTATE_LEFT", 18, chainY[1], 88, "ROT L 90°");
  node("align", "FORWARD_AFTER_LEFT", 132, chainY[1], 88, "FWD");
  node("align", "ALIGN_ROTATE_RIGHT_90", 246, chainY[1], 104, "ROT R 90°");
}

function drawVideoStateNode(context, node) {
  context.fillStyle = node.active ? "#f3a33b" : node.visited ? "#d8f0e4" : "#eef1f4";
  context.strokeStyle = node.active ? "#b46100" : node.visited ? "#4a9a70" : "#c7ced6";
  context.lineWidth = node.active ? 3 : 1;
  context.beginPath();
  context.roundRect(node.x, node.y, node.width, node.height, 6);
  context.fill();
  context.stroke();
  context.fillStyle = "#20262d";
  context.font = `${node.active ? "700" : "500"} 11px system-ui`;
  const label = node.label.length > 22 ? `${node.label.slice(0, 20)}…` : node.label;
  context.textAlign = "center";
  context.fillText(label, node.x + node.width / 2, node.y + 19);
  context.textAlign = "left";
}

function addLog(message) {
  app.log.unshift(`${app.fsm.now.toFixed(2).padStart(6, " ")}s  ${message}`);
  app.log = app.log.slice(0, 120);
}

function render() {
  const fsm = app.fsm;
  const sensors = app.plant.sensors();
  const truth = app.plant.groundTruth();
  const sub = fsm.topState === "ALIGN" ? fsm.alignSub
    : fsm.topState === "RECOVER" ? fsm.recoverSub : "—";
  el.statePill.textContent = fsm.statePath;
  el.cmdPill.textContent = fsm.command;
  el.topState.textContent = fsm.topState;
  el.subState.textContent = sub;
  el.commandReadout.textContent = fsm.command;
  const stabilizer = fsm.topState === "CHECK" ? fsm.checkStabilizer
    : fsm.topState === "RECOVER" ? fsm.recoverStabilizer : fsm.alignStabilizer;
  el.stableReadout.textContent = `${stabilizer.count} / ${stabilizer.threshold} frames`;
  el.interlockReadout.textContent = fsm.interlockActive
    ? `${Math.max(0, fsm.interlockUntil - fsm.now).toFixed(2)} s → ${fsm.afterInterlockSub}`
    : "inactive";
  el.detReadout.textContent = `${sensors.detOk ? "인식" : "미인식"} / ${formatNumber(sensors.detectedLength, 3, "m")}`;
  const visiblePercent = (app.visibility?.fraction || 0) * 100;
  el.visibilityReadout.textContent = `${visiblePercent.toFixed(1)}% / ${sensors.detOk ? "인식 가능" : "인식 불가"}`;
  el.visibilityReadout.classList.toggle("blocked", !sensors.detOk);
  el.cameraReadout.textContent = `${SIM_CONFIG.CAMERA_HFOV_DEG.toFixed(1)}° / ${app.cameraRange.toFixed(2)} m`;
  el.distReadout.textContent = `${formatNumber(sensors.distZ, 3, "m")} (실제 ${truth.distZ.toFixed(3)} m)`;
  el.yawReadout.textContent = `${formatNumber(sensors.yaw, 2, "°")} (실제 ${truth.yaw.toFixed(2)}°)`;
  el.offsetReadout.textContent = `${formatNumber(sensors.offsetX, 3, "m")} (실제 ${truth.offsetX.toFixed(3)} m)`;
  el.relYawReadout.textContent = formatNumber(sensors.relYaw, 2, "°");
  const timeScale = clamp(numberValue(el.timeScale, 1), 0.25, 8);
  const drift = fsm.now - app.wallScaledElapsed;
  el.simClock.textContent = `sim ${fsm.now.toFixed(2)} s · wall ${app.wallRunElapsed.toFixed(2)} s`
    + ` · ${timeScale}× · drift ${drift >= 0 ? "+" : ""}${drift.toFixed(3)} s · ${CAMERA_RATE_HZ} Hz`;
  el.logBox.innerHTML = app.log.map((line) => `<div>${escapeHtml(line)}</div>`).join("");
  renderStateDiagram();
  renderCollision();
  renderMotion();
}

function renderCollision() {
  const collision = app.plant.collisionTelemetry();
  const status = collision.detected ? "COLLISION" : collision.inserting ? "INSERTING · CLEAR" : "CLEAR";
  el.collisionReadout.textContent = status;
  el.collisionReadout.classList.toggle("collision", collision.detected);
  el.clearanceReadout.textContent = collision.clearanceM === null || collision.clearanceM === undefined
    ? "N/A" : `${collision.clearanceM.toFixed(3)} m`;
  el.forkGeometryReadout.textContent = `outer ${SIM_CONFIG.FORK_OUTER_SPAN_M.toFixed(2)}m · tine ${app.plant.forkTineWidth.toFixed(2)}m`;
  el.collisionDetail.textContent = collision.reason;
  el.safetyPill.textContent = collision.detected ? "FORK COLLISION" : collision.inserting ? "FORK INSERTING" : "FORK CLEAR";
  el.safetyPill.classList.toggle("collision", collision.detected);
}

function renderMotion() {
  const motion = app.plant.motionTelemetry();
  if (!motion) {
    el.motionProfileReadout.textContent = "FSM 명령 대기";
    el.motionTimeReadout.textContent = "0.00 s";
    el.motionVelocityReadout.textContent = "0.000 m/s";
    el.motionDistanceReadout.textContent = "0.000 m";
  } else {
    const profileLabel = motion.profile === "constant"
      ? `${motion.command} · 원본 INSERT 정속`
      : motion.command === "BACK"
        ? "BACK · FWD 피팅 대칭"
        : `${motion.command} · 원본 가속→정속`;
    el.motionProfileReadout.textContent = profileLabel;
    el.motionTimeReadout.textContent = Number.isFinite(motion.plannedDuration)
      ? `${motion.elapsed.toFixed(2)} / ${motion.plannedDuration.toFixed(2)} s · timer`
      : `${motion.elapsed.toFixed(2)} s · sensor-band stop`;
    el.motionVelocityReadout.textContent = `${motion.velocity.toFixed(3)} m/s`;
    const requested = motion.requestedDistance ?? motion.targetDistance;
    const requestSuffix = Number.isFinite(requested) && requested > 0
      ? ` · stop ref ${requested.toFixed(3)} m` : "";
    el.motionDistanceReadout.textContent = `${motion.distance.toFixed(3)} m${requestSuffix}`;
  }
  const selectedDuration = clamp(numberValue(el.fwdApplyDuration, 10), 0, CONFIG.FWD_MAX_SEC);
  const predictedDistance = distanceFromDurationPiecewise(selectedDuration, CONFIG);
  el.fwdDistancePrediction.textContent = `${selectedDuration.toFixed(2)} s → ${predictedDistance.toFixed(3)} m`;
  drawMotionChart(selectedDuration);
}

function formatNumber(value, digits, unit) {
  return value === null || value === undefined ? "N/A" : `${value.toFixed(digits)} ${unit}`;
}

function initConstants(cfg = currentSimulationFsmConfig()) {
  el.constantGrid.innerHTML = FSM_CONSTANT_FIELDS.map((field) =>
    `<div><label for="constant_${field.key}">${field.label}</label>`
      + `<input id="constant_${field.key}" type="number" min="${field.min}" max="${field.max}"`
      + ` step="${field.step}" value="${cfg[field.key]}"></div>`).join("");
}

function applyFsmConstants() {
  try {
    const updates = {};
    FSM_CONSTANT_FIELDS.forEach((field) => {
      const input = $(`constant_${field.key}`);
      let value = Number(input?.value);
      if (!Number.isFinite(value) || value < field.min || value > field.max) {
        throw new Error(`${field.label}: ${field.min}~${field.max} 범위의 숫자가 필요합니다.`);
      }
      if (field.integer) value = Math.round(value);
      updates[field.key] = value;
    });
    activeFsmOverrides = { ...activeFsmOverrides, ...updates };
    initConstants();
    initStateDiagram();
    resetSimulation();
    el.constantStatus.textContent = "적용 완료 · 일반 실행, 재현, 27조건 실험, 그리드 서치에 동일하게 사용됩니다.";
  } catch (error) {
    el.constantStatus.textContent = `적용 실패 · ${error.message}`;
  }
}

function diagramNode(kind, state, x, y, width, height, label, shape = "rect") {
  const lines = label.split("|");
  const geometry = shape === "diamond"
    ? `<polygon class="node-shape" points="${width / 2},0 ${width},${height / 2} ${width / 2},${height} 0,${height / 2}"></polygon>`
    : `<rect class="node-shape" x="0" y="0" width="${width}" height="${height}" rx="10"></rect>`;
  const startY = height / 2 - (lines.length - 1) * 10;
  const text = lines.map((line, index) =>
    `<tspan x="${width / 2}" y="${startY + index * 20}">${line}</tspan>`).join("");
  return `<g class="diagram-node" data-kind="${kind}" data-state="${state}" transform="translate(${x} ${y})">
    ${geometry}<text class="node-label">${text}</text></g>`;
}

function diagramEdge(path, label = "", labelX = 0, labelY = 0) {
  return `<g class="diagram-edge"><path d="${path}" marker-end="url(#fsm-arrow)"></path>`
    + (label ? `<text x="${labelX}" y="${labelY}">${label}</text>` : "") + `</g>`;
}

function initStateDiagram(cfg = currentSimulationFsmConfig()) {
  const n = diagramNode;
  const e = diagramEdge;
  const turn = `${cfg.REL_YAW_TARGET_DEG.toFixed(0)}°`;
  el.fsmDiagram.innerHTML = `
    <svg class="fsm-flowchart" viewBox="0 0 720 1260" role="img" aria-label="실시간 TOP, RECOVER, ALIGN FSM 순서도">
      <defs>
        <marker id="fsm-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L8,4 L0,8 z"></path>
        </marker>
      </defs>
      <text class="flowchart-title" x="24" y="32">TOP FSM</text>
      <rect class="flowchart-boundary" x="12" y="44" width="696" height="244" rx="14"></rect>
      ${e("M146 110 H178", "detect", 149, 98)}
      ${e("M302 110 H354", "width OK", 306, 98)}
      ${e("M486 110 H556", "ready", 500, 98)}
      ${e("M240 145 V194 H194", "width low", 246, 178)}
      ${e("M310 220 H354")}
      ${e("M486 220 H610 V144", "OK", 533, 210)}
      ${e("M420 194 V160 H420 V144", "retry", 427, 173)}
      ${e("M240 76 V62 H86 V84", "lost", 151, 57)}
      ${n("top", "SEARCH", 26, 84, 120, 52, "SEARCH")}
      ${n("top", "DETECTED", 178, 74, 124, 72, "DETECTED", "diamond")}
      ${n("top", "ALIGN", 354, 84, 132, 52, "ALIGN")}
      ${n("top", "DONE", 556, 84, 108, 52, "DONE")}
      ${n("top", "RECOVER", 70, 194, 124, 52, "RECOVER")}
      ${n("top", "CHECK", 354, 184, 132, 72, "CHECK", "diamond")}

      <text class="flowchart-title" x="24" y="326">RECOVER SUB-FSM</text>
      <rect class="flowchart-boundary" x="12" y="338" width="696" height="226" rx="14"></rect>
      ${e("M360 422 H190 V466", "left", 214, 409)}
      ${e("M360 422 H530 V466", "right", 474, 409)}
      ${e("M190 518 V538 H360", "front visible", 206, 548)}
      ${e("M530 518 V538 H360")}
      ${n("recover", "DECIDE_TURN", 292, 370, 136, 76, "DECIDE|TURN", "diamond")}
      ${n("recover", "RECOVER_ROTATE_LEFT", 116, 466, 148, 52, "ROTATE LEFT")}
      ${n("recover", "RECOVER_ROTATE_RIGHT", 456, 466, 148, 52, "ROTATE RIGHT")}
      ${n("recover", "HOLD", 306, 516, 108, 42, "HOLD")}

      <text class="flowchart-title" x="24" y="604">ALIGN SUB-FSM</text>
      <rect class="flowchart-boundary" x="12" y="616" width="696" height="630" rx="14"></rect>
      ${e("M360 700 V744", "distance OK", 368, 726)}
      ${e("M290 674 H154 V742", "far", 202, 665)}
      ${e("M430 674 H566 V742", "near", 493, 665)}
      ${e("M154 794 V814 H320 V700")}
      ${e("M566 794 V814 H400 V700")}
      ${e("M360 820 V860", "yaw OK", 368, 846)}
      ${e("M290 784 H154 V862", "+yaw", 201, 775)}
      ${e("M430 784 H566 V862", "−yaw", 493, 775)}
      ${e("M154 914 V934 H320 V896")}
      ${e("M566 914 V934 H400 V896")}
      ${e("M430 896 H496", "aligned", 439, 885)}
      ${e("M584 922 V962", "timer", 591, 951)}
      ${e("M360 932 V1000 H106", "+offset", 267, 987)}
      ${e("M360 932 V1110 H106", "−offset", 267, 1097)}
      ${e("M214 1026 H270")}${e("M386 1026 H442")}
      ${e("M214 1136 H270")}${e("M386 1136 H442")}
      ${e("M550 1026 H680 V790 H430", "yaw recheck", 572, 1015)}
      ${e("M550 1136 H694 V804 H430")}
      ${e("M290 896 H34 V674 H290", "offset N/A", 42, 884)}
      ${n("align", "DIST_CHECK", 290, 638, 140, 72, "DIST|CHECK", "diamond")}
      ${n("align", "ALIGN_FWD_ADJUST", 86, 742, 136, 52, "FWD ADJUST")}
      ${n("align", "ALIGN_BWD_ADJUST", 498, 742, 136, 52, "BWD ADJUST")}
      ${n("align", "YAW_CHECK", 290, 748, 140, 72, "YAW|CHECK", "diamond")}
      ${n("align", "ROTATE_RIGHT_UNTIL_YAW_TOL", 80, 862, 148, 52, "ROT RIGHT CMD")}
      ${n("align", "ROTATE_LEFT_UNTIL_YAW_TOL", 492, 862, 148, 52, "ROT LEFT CMD")}
      ${n("align", "OFFSET_CHECK", 290, 860, 140, 72, "OFFSET|CHECK", "diamond")}
      ${n("align", "INSERT_FORWARD", 496, 870, 176, 52, "INSERT FORWARD")}
      ${n("align", "READY_TO_DONE", 510, 962, 148, 48, "READY → DONE")}
      ${n("align", "ALIGN_ROTATE_RIGHT", 70, 1000, 144, 52, `ROT RIGHT ${turn}`)}
      ${n("align", "FORWARD_AFTER_RIGHT", 270, 1000, 116, 52, "FWD")}
      ${n("align", "ALIGN_ROTATE_LEFT_90", 442, 1000, 108, 52, `ROT LEFT ${turn}`)}
      ${n("align", "ALIGN_ROTATE_LEFT", 70, 1110, 144, 52, `ROT LEFT ${turn}`)}
      ${n("align", "FORWARD_AFTER_LEFT", 270, 1110, 116, 52, "FWD")}
      ${n("align", "ALIGN_ROTATE_RIGHT_90", 442, 1110, 108, 52, `ROT RIGHT ${turn}`)}
    </svg>`;
}

function renderStateDiagram() {
  const currentKeys = new Set([`top:${app.fsm.topState}`]);
  app.visitedStates.add(`top:${app.fsm.topState}`);
  if (app.fsm.topState === "ALIGN") {
    currentKeys.add(`align:${app.fsm.alignSub}`);
    app.visitedStates.add(`align:${app.fsm.alignSub}`);
  }
  if (app.fsm.topState === "RECOVER") {
    currentKeys.add(`recover:${app.fsm.recoverSub}`);
    app.visitedStates.add(`recover:${app.fsm.recoverSub}`);
  }
  el.diagramCurrent.textContent = `현재: ${app.fsm.statePath} · command ${app.fsm.command}`;
  el.fsmDiagram.querySelectorAll(".diagram-node").forEach((item) => {
    const key = `${item.dataset.kind}:${item.dataset.state}`;
    item.classList.toggle("active", currentKeys.has(key));
    item.classList.toggle("visited", app.visitedStates.has(key));
  });
}

function resizeCanvas() {
  const rect = app.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(640, Math.floor(rect.width * dpr));
  const height = Math.max(480, Math.floor(rect.height * dpr));
  if (app.canvas.width !== width || app.canvas.height !== height) {
    app.canvas.width = width;
    app.canvas.height = height;
  }
}

function worldToCanvas(point) {
  const world = app.world;
  const padding = 36 * (window.devicePixelRatio || 1);
  const scale = Math.min(
    (app.canvas.width - padding * 2) / (world.maxX - world.minX),
    (app.canvas.height - padding * 2) / (world.maxY - world.minY)
  );
  return {
    x: padding + (point.x - world.minX) * scale,
    y: app.canvas.height - padding - (point.y - world.minY) * scale,
    scale
  };
}

function draw() {
  if (!app.ctx) return;
  resizeCanvas();
  const ctx = app.ctx;
  ctx.clearRect(0, 0, app.canvas.width, app.canvas.height);
  drawBackground(ctx);
  drawTargetBand(ctx);
  drawCameraFov(ctx);
  drawPallet(ctx);
  drawForklift(ctx);
  drawCommandVector(ctx);
  drawRecordingFrame();
}

function drawBackground(ctx) {
  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.fillStyle = "#f7f8fa";
  ctx.fillRect(0, 0, app.canvas.width, app.canvas.height);
  ctx.strokeStyle = "#dfe4ea";
  ctx.lineWidth = 1;
  for (let x = Math.ceil(app.world.minX / SIM_CONFIG.GRID_SIZE_M) * SIM_CONFIG.GRID_SIZE_M; x <= app.world.maxX; x += SIM_CONFIG.GRID_SIZE_M) {
    line(ctx, worldToCanvas({ x, y: app.world.minY }), worldToCanvas({ x, y: app.world.maxY }));
  }
  for (let y = Math.ceil(app.world.minY / SIM_CONFIG.GRID_SIZE_M) * SIM_CONFIG.GRID_SIZE_M; y <= app.world.maxY; y += SIM_CONFIG.GRID_SIZE_M) {
    line(ctx, worldToCanvas({ x: app.world.minX, y }), worldToCanvas({ x: app.world.maxX, y }));
  }
  ctx.strokeStyle = "#9ba6b2";
  ctx.lineWidth = 1.5 * dpr;
  line(ctx, worldToCanvas({ x: 0, y: app.world.minY }), worldToCanvas({ x: 0, y: app.world.maxY }));
  const scaleStart = worldToCanvas({ x: -2.75, y: -1.72 });
  const scaleEnd = worldToCanvas({ x: -1.75, y: -1.72 });
  ctx.strokeStyle = "#34414d";
  ctx.lineWidth = 4;
  line(ctx, scaleStart, scaleEnd);
  ctx.restore();
}

function drawTargetBand(ctx) {
  const front = palletFrontGeometry();
  const center = app.plant.palletFaceCenter;
  const tangent = {
    x: (front.end.x - front.start.x) / SIM_CONFIG.PALLET_FRONT_WIDTH_M,
    y: (front.end.y - front.start.y) / SIM_CONFIG.PALLET_FRONT_WIDTH_M
  };
  const outward = front.outwardNormal;
  const lateralExtent = 5;
  const cfg = currentSimulationFsmConfig();
  const nearDistance = cfg.ALIGN_DIST_M - cfg.ALIGN_BAND_M;
  const farDistance = cfg.ALIGN_DIST_M + cfg.ALIGN_BAND_M;
  const point = (lateral, distance) => ({
    x: center.x + tangent.x * lateral + outward.x * distance,
    y: center.y + tangent.y * lateral + outward.y * distance
  });
  const polygon = [
    point(-lateralExtent, farDistance),
    point(lateralExtent, farDistance),
    point(lateralExtent, nearDistance),
    point(-lateralExtent, nearDistance)
  ].map(worldToCanvas);
  ctx.save();
  ctx.fillStyle = "rgba(45, 160, 108, 0.08)";
  ctx.beginPath();
  ctx.moveTo(polygon[0].x, polygon[0].y);
  polygon.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(45, 160, 108, 0.5)";
  ctx.setLineDash([8, 6]);
  line(ctx, worldToCanvas(point(-lateralExtent, farDistance)), worldToCanvas(point(lateralExtent, farDistance)));
  line(ctx, worldToCanvas(point(-lateralExtent, nearDistance)), worldToCanvas(point(lateralExtent, nearDistance)));
  ctx.setLineDash([]);
  ctx.restore();
}

function drawPallet(ctx) {
  const frontWidth = SIM_CONFIG.PALLET_FRONT_WIDTH_M;
  const depth = SIM_CONFIG.PALLET_DEPTH_M;
  const center = app.plant.palletPoint(0, depth / 2);
  const c = worldToCanvas(center);
  const scale = c.scale;
  ctx.save();
  ctx.translate(c.x, c.y);
  ctx.rotate(Math.PI / 2 - app.plant.palletHeadingDeg * DEG);
  ctx.fillStyle = "#c89b62";
  ctx.strokeStyle = "#59401f";
  ctx.lineWidth = 2.5;
  ctx.fillRect(-frontWidth * scale / 2, -depth * scale / 2, frontWidth * scale, depth * scale);
  ctx.strokeRect(-frontWidth * scale / 2, -depth * scale / 2, frontWidth * scale, depth * scale);
  ctx.fillStyle = "rgba(30, 22, 12, 0.65)";
  const pocketWidth = SIM_CONFIG.PALLET_POCKET_WIDTH_M;
  const halfGap = SIM_CONFIG.PALLET_POCKET_INNER_GAP_M / 2;
  ctx.fillRect((-halfGap - pocketWidth) * scale, -depth * scale / 2, pocketWidth * scale, depth * scale);
  ctx.fillRect(halfGap * scale, -depth * scale / 2, pocketWidth * scale, depth * scale);
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = "#4c3519";
  ctx.lineWidth = 4;
  for (const [startX, endX] of [
    [-frontWidth / 2, -halfGap - pocketWidth],
    [-halfGap, halfGap],
    [halfGap + pocketWidth, frontWidth / 2]
  ]) {
    line(ctx, worldToCanvas(app.plant.palletPoint(startX, 0)), worldToCanvas(app.plant.palletPoint(endX, 0)));
  }
  if (app.visibility?.visibleStart && app.visibility?.visibleEnd) {
    const outward = palletFrontGeometry().outwardNormal;
    ctx.strokeStyle = app.plant.detOk ? "#1c9b5f" : "#d99026";
    ctx.lineWidth = 4;
    line(
      ctx,
      worldToCanvas({
        x: app.visibility.visibleStart.x + outward.x * 0.07,
        y: app.visibility.visibleStart.y + outward.y * 0.07
      }),
      worldToCanvas({
        x: app.visibility.visibleEnd.x + outward.x * 0.07,
        y: app.visibility.visibleEnd.y + outward.y * 0.07
      })
    );
  }
  drawDot(ctx, worldToCanvas(app.plant.palletFaceCenter), 5, "#4c3519");
  ctx.restore();
}

function drawCameraFov(ctx) {
  const camera = app.plant.cameraPose();
  const half = SIM_CONFIG.CAMERA_HFOV_DEG / 2;
  const center = worldToCanvas(camera);
  ctx.save();
  ctx.strokeStyle = "rgba(43, 116, 177, 0.38)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([7, 6]);
  ctx.beginPath();
  ctx.arc(center.x, center.y, app.cameraRange * center.scale, 0, TAU);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  for (let index = 0; index <= 48; index += 1) {
    const angle = (camera.headingDeg - half + SIM_CONFIG.CAMERA_HFOV_DEG * index / 48) * DEG;
    const point = worldToCanvas({
      x: camera.x + Math.cos(angle) * app.cameraRange,
      y: camera.y + Math.sin(angle) * app.cameraRange
    });
    ctx.lineTo(point.x, point.y);
  }
  ctx.closePath();
  ctx.fillStyle = app.plant.detOk ? "rgba(43, 133, 196, 0.12)" : "rgba(205, 91, 71, 0.12)";
  ctx.strokeStyle = app.plant.detOk ? "rgba(43, 116, 177, 0.65)" : "rgba(190, 75, 60, 0.65)";
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function forkliftPose() {
  return {
    x: app.plant.bodyX,
    y: app.plant.bodyY,
    heading: app.plant.headingDeg
  };
}

function drawForklift(ctx) {
  const pose = forkliftPose();
  const center = worldToCanvas({ x: pose.x, y: pose.y });
  const scale = center.scale;
  const heading = pose.heading * DEG;
  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(Math.PI / 2 - heading);
  ctx.fillStyle = "#edbe28";
  ctx.strokeStyle = "#171b21";
  ctx.lineWidth = 2;
  ctx.beginPath();
  const bodyFront = SIM_CONFIG.CAMERA_MOUNT_FORWARD_M;
  const bodyLength = 1.30;
  ctx.roundRect(-0.46 * scale, -bodyFront * scale, 0.92 * scale, bodyLength * scale, 0.10 * scale);
  ctx.fill();
  ctx.stroke();
  const collision = app.plant.collisionTelemetry();
  const outerHalf = SIM_CONFIG.FORK_OUTER_SPAN_M / 2;
  const tineWidth = app.plant.forkTineWidth;
  const forkRoot = SIM_CONFIG.FORK_FORWARD_ROOT_M;
  const forkTip = SIM_CONFIG.FORK_FORWARD_TIP_M;
  ctx.fillStyle = collision.detected ? "#d83939" : "#262c34";
  ctx.fillRect(-outerHalf * scale, -forkTip * scale, tineWidth * scale, (forkTip - forkRoot) * scale);
  ctx.fillRect((outerHalf - tineWidth) * scale, -forkTip * scale, tineWidth * scale, (forkTip - forkRoot) * scale);
  ctx.strokeStyle = collision.detected ? "#8f1212" : "#171b21";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(-outerHalf * scale, -(forkTip + 0.10) * scale);
  ctx.lineTo(outerHalf * scale, -(forkTip + 0.10) * scale);
  ctx.stroke();
  ctx.fillStyle = collision.detected ? "#a51f1f" : "#3a4652";
  ctx.fillRect(-0.55 * scale, -0.38 * scale, 0.12 * scale, 0.38 * scale);
  ctx.fillRect(0.43 * scale, -0.38 * scale, 0.12 * scale, 0.38 * scale);
  ctx.restore();
  drawDot(ctx, center, 4, "#111", "#fff");
  drawDot(ctx, worldToCanvas(app.plant.cameraPose()), 6, "#2f74b9", "#fff");
}

function drawCommandVector(ctx) {
  const command = app.fsm.command;
  if (command === "STOP") return;
  const pose = forkliftPose();
  const center = worldToCanvas({ x: pose.x, y: pose.y });
  const heading = pose.heading * DEG;
  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  if (["FWD", "BACK"].includes(command)) {
    const direction = command === "FWD" ? 1 : -1;
    const end = {
      x: center.x + Math.cos(heading) * 78 * dpr * direction,
      y: center.y - Math.sin(heading) * 78 * dpr * direction
    };
    arrow(ctx, center, end, "#16906a");
  } else {
    ctx.strokeStyle = "#8b55cc";
    ctx.lineWidth = 4 * dpr;
    ctx.beginPath();
    const ccw = command === "ROT_LEFT";
    ctx.arc(center.x, center.y, 50 * dpr, ccw ? 0.2 : -2.9, ccw ? 5.0 : 1.9, !ccw);
    ctx.stroke();
  }
  ctx.restore();
}

function drawMotionChart(selectedDuration) {
  if (!app.motionCtx || !app.motionCanvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = app.motionCanvas.getBoundingClientRect();
  const width = Math.max(340, Math.floor(rect.width * dpr));
  const height = Math.max(200, Math.floor(rect.height * dpr));
  if (app.motionCanvas.width !== width || app.motionCanvas.height !== height) {
    app.motionCanvas.width = width;
    app.motionCanvas.height = height;
  }
  const ctx = app.motionCtx;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);

  const horizon = CONFIG.FWD_MAX_SEC;
  const applyDuration = clamp(Number(selectedDuration) || 0, 0, horizon);
  const maxDistance = Math.max(0.001, distanceFromDurationPiecewise(horizon, CONFIG));
  const margin = { left: 47 * dpr, right: 18 * dpr, top: 22 * dpr, bottom: 25 * dpr };
  const gap = 22 * dpr;
  const panelHeight = (height - margin.top - margin.bottom - gap) / 2;
  const plotWidth = width - margin.left - margin.right;

  const sampleDistance = (time) => distanceFromDurationPiecewise(time, CONFIG);
  const sampleVelocity = (time) => rawMotionVelocityAtTime(time, CONFIG);
  let maxVelocity = 0;
  for (let index = 0; index <= 100; index += 1) {
    maxVelocity = Math.max(maxVelocity, sampleVelocity(horizon * index / 100));
  }
  maxVelocity = Math.max(0.01, maxVelocity);

  function drawPanel(top, maxY, sampler, color, title, unit) {
    const left = margin.left;
    const bottom = top + panelHeight;
    ctx.save();
    ctx.fillStyle = "rgba(230, 157, 43, 0.10)";
    ctx.fillRect(left, top, plotWidth * applyDuration / horizon, panelHeight);
    ctx.strokeStyle = "#d7dde3";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const x = left + plotWidth * i / 4;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    }
    for (let i = 0; i <= 2; i += 1) {
      const y = bottom - panelHeight * i / 2;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotWidth, y); ctx.stroke();
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.3 * dpr;
    ctx.beginPath();
    for (let index = 0; index <= 120; index += 1) {
      const time = horizon * index / 120;
      const x = left + plotWidth * time / horizon;
      const y = bottom - panelHeight * clamp(sampler(time) / maxY, 0, 1);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    const progressX = left + plotWidth * applyDuration / horizon;
    ctx.strokeStyle = "#2b3540";
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath(); ctx.moveTo(progressX, top); ctx.lineTo(progressX, bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#3f4b57";
    ctx.font = `${10 * dpr}px system-ui`;
    ctx.fillText(title, left, top - 7 * dpr);
    ctx.fillText(`0`, 9 * dpr, bottom + 3 * dpr);
    ctx.fillText(`${maxY.toFixed(3)} ${unit}`, 5 * dpr, top + 10 * dpr);
    ctx.restore();
  }

  drawPanel(margin.top, maxVelocity, sampleVelocity, "#7b4fc6", "forward velocity · v(t)", "m/s");
  drawPanel(margin.top + panelHeight + gap, maxDistance, sampleDistance, "#168d68", "duration → d(t)", "m");
  ctx.fillStyle = "#596572";
  ctx.font = `${10 * dpr}px system-ui`;
  ctx.fillText("0 s", margin.left, height - 7 * dpr);
  ctx.fillText(`${horizon.toFixed(0)} s`, width - margin.right - 28 * dpr, height - 7 * dpr);
}

function line(ctx, a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function arrow(ctx, a, b, color) {
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const head = 10 * (window.devicePixelRatio || 1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3;
  line(ctx, a, b);
  ctx.beginPath();
  ctx.moveTo(b.x, b.y);
  ctx.lineTo(b.x - head * Math.cos(angle - 0.5), b.y - head * Math.sin(angle - 0.5));
  ctx.lineTo(b.x - head * Math.cos(angle + 0.5), b.y - head * Math.sin(angle + 0.5));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawDot(ctx, point, radius, fill, stroke) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius * (window.devicePixelRatio || 1), 0, TAU);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
  ctx.restore();
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[character]));
}

function animationLoop(now) {
  const elapsed = Math.max(0, (now - app.lastAnimationTime) / 1000);
  app.lastAnimationTime = now;
  if (app.running) {
    const timeScale = clamp(numberValue(el.timeScale, 1), 0.25, 8);
    app.wallRunElapsed += elapsed;
    app.wallScaledElapsed += elapsed * timeScale;
    app.accumulator += elapsed * timeScale;
    let guard = 0;
    while (app.accumulator >= FRAME_DT && guard < 900) {
      simulateFrame();
      app.accumulator -= FRAME_DT;
      guard += 1;
      if (!app.running) break;
    }
  }
  draw();
  requestAnimationFrame(animationLoop);
}

function init() {
  bindElements();
  renderDriveErrorFit();
  renderEstimationErrorFit();
  initConstants();
  initStateDiagram();
  wireEvents();
  resetSimulation();
  requestAnimationFrame(animationLoop);
}

init();
