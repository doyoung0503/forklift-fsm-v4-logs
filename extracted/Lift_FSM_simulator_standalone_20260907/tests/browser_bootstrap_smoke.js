"use strict";

const core = require("../original_fsm_core.js");
const driveError = require("../drive_error_model.js");
const estimationError = require("../estimation_error_model.js");
const hyperparameterSearch = require("../hyperparameter_search.js");

const noop = () => {};
const canvasTextCalls = {
  simCanvas: [], motionChart: [], trajectoryChart: [],
  driveDistributionChart: [], estimationDistributionChart: []
};
const contexts = new Map();
function contextFor(id) {
  if (!contexts.has(id)) {
    contexts.set(id, new Proxy({}, {
      get: (_target, property) => {
        if (property === "measureText") return () => ({ width: 10 });
        if (property === "fillText") return (...args) => canvasTextCalls[id]?.push(args);
        return noop;
      },
      set: () => true
    }));
  }
  return contexts.get(id);
}

const defaultValues = {
  timeScale: "1",
  initialDist: "3.10",
  initialYaw: "12.0",
  initialOffset: "0.42",
  rotateSpeed: "15",
  cameraRange: "3.5",
  forkTineWidth: "0.10",
  fwdApplyDuration: "10",
  driveErrorSeed: "20260713",
  estimationErrorSeed: "20260714",
  gridYawCandidates: "3",
  gridOffsetCandidates: "",
  gridWidthCandidates: "",
  gridDistanceCandidates: "",
  gridBandCandidates: "",
  gridStableCandidates: "",
  gridSearchRepeats: "1"
};

function makeElement(id) {
  const listeners = {};
  const makeClassList = () => ({
    values: new Set(),
    toggle(name, force) {
      if (force === undefined ? !this.values.has(name) : force) this.values.add(name);
      else this.values.delete(name);
    },
    add(name) { this.values.add(name); },
    remove(name) { this.values.delete(name); }
  });
  const element = {
    id,
    value: defaultValues[id] || "",
    checked: id === "driveErrorEnabled" || id === "estimationErrorEnabled",
    files: [],
    disabled: false,
    textContent: "",
    _innerHTML: "",
    _diagramNodes: [],
    dataset: {},
    classList: makeClassList(),
    listeners,
    addEventListener: (event, handler) => { listeners[event] = handler; },
    getContext: () => contextFor(id),
    getBoundingClientRect: () => ({ width: 900, height: 700 }),
    querySelectorAll(selector) { return selector === ".diagram-node" ? this._diagramNodes : []; },
    setAttribute: noop
  };
  Object.defineProperty(element, "innerHTML", {
    get() { return this._innerHTML; },
    set(value) {
      this._innerHTML = value;
      if (id !== "fsmDiagram") return;
      this._diagramNodes = [];
      const pattern = /data-kind="([^"]+)" data-state="([^"]+)"/g;
      let match;
      while ((match = pattern.exec(value))) {
        this._diagramNodes.push({
          dataset: { kind: match[1], state: match[2] },
          classList: makeClassList()
        });
      }
    }
  });
  return element;
}

const elements = new Map();
global.window = {
  OriginalFSM: core,
  DriveError: driveError,
  EstimationError: estimationError,
  HyperparameterSearch: hyperparameterSearch,
  devicePixelRatio: 1,
  addEventListener: noop
};
global.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  }
};
let currentTimeMs = 0;
let animationCallback = null;
global.performance = { now: () => currentTimeMs };
global.requestAnimationFrame = (callback) => { animationCallback = callback; };

require("../sim.js");

if (elements.size !== 90) throw new Error(`unexpected bound element count: ${elements.size}`);
const driveFitRows = (elements.get("driveFitResults").innerHTML.match(/<tr>/g) || []).length;
if (driveFitRows !== 9) throw new Error(`drive error fit rendered ${driveFitRows} conditions instead of 9`);
if (!elements.get("driveErrorStatus").textContent.includes("예제 로그 270행")) {
  throw new Error("drive error example log was not fitted at bootstrap");
}
if (!elements.get("driveErrorStatus").textContent.includes("전진")
    || !elements.get("driveErrorStatus").textContent.includes("회전")) {
  throw new Error("drive error UI is missing type-level distribution selections");
}
const diagram = elements.get("fsmDiagram").innerHTML;
for (const state of ["SEARCH", "DETECTED", "RECOVER", "CHECK", "DIST_CHECK", "INSERT_FORWARD", "READY_TO_DONE"]) {
  if (!diagram.includes(`data-state="${state}"`)) throw new Error(`diagram missing state ${state}`);
}
if (!diagram.includes("ROT RIGHT 90°") || !diagram.includes("ROT LEFT 90°") || diagram.includes("85°")) {
  throw new Error("simulator diagram does not show the 90-degree offset-turn override");
}
if (!elements.get("constantGrid").innerHTML.includes('id="constant_REL_YAW_TARGET_DEG"')
    || !elements.get("constantGrid").innerHTML.includes('value="90"')) {
  throw new Error("simulator constants do not show REL_YAW=90°");
}
if (!canvasTextCalls.driveDistributionChart.some((args) => String(args[0]).includes("BIC"))) {
  throw new Error("drive-error histogram/distribution chart did not render BIC labels");
}
const estimationFitRows = (elements.get("estimationFitResults").innerHTML.match(/<tr>/g) || []).length;
if (estimationFitRows !== 9) {
  throw new Error(`estimation error fit rendered ${estimationFitRows} conditions instead of 9`);
}
if (!elements.get("estimationErrorStatus").textContent.includes("전방거리")
    || !elements.get("estimationErrorStatus").textContent.includes("좌우위치")
    || !elements.get("estimationErrorStatus").textContent.includes("팔레트각")) {
  throw new Error("estimation error UI is missing variable-level distribution selections");
}
if (!canvasTextCalls.estimationDistributionChart.some((args) => String(args[0]).includes("BIC"))) {
  throw new Error("estimation-error histogram/distribution chart did not render BIC labels");
}
if (!elements.get("fsmDiagram")._diagramNodes.some((node) =>
  node.dataset.state === "SEARCH" && node.classList.values.has("active"))) {
  throw new Error("initial SEARCH diagram node is not active");
}

elements.get("runBtn").listeners.click();
for (currentTimeMs = 1000 / 60; currentTimeMs <= 3000; currentTimeMs += 1000 / 60) {
  const callback = animationCallback;
  callback(currentTimeMs);
}
const clock = elements.get("simClock").textContent;
const times = clock.match(/sim ([\d.]+) s · wall ([\d.]+) s · [\d.]+× · drift ([+\-\d.]+) s/);
if (!times) throw new Error(`unexpected realtime clock: ${clock}`);
if (Math.abs(Number(times[1]) - Number(times[2])) > 1 / 30 + 0.002) {
  throw new Error(`simulation drift exceeds one 30 Hz frame: ${clock}`);
}
elements.get("resetBtn").listeners.click();
elements.get("timeScale").value = "2";
elements.get("timeScale").listeners.change();
elements.get("runBtn").listeners.click();
const scaledStartMs = currentTimeMs;
for (currentTimeMs += 1000 / 60; currentTimeMs <= scaledStartMs + 1000; currentTimeMs += 1000 / 60) {
  animationCallback(currentTimeMs);
}
const scaledClock = elements.get("simClock").textContent;
const scaledTimes = scaledClock.match(/sim ([\d.]+) s · wall ([\d.]+) s · 2×/);
if (!scaledTimes || Number(scaledTimes[1]) < Number(scaledTimes[2]) * 1.9) {
  throw new Error(`2x realtime speed was not applied: ${scaledClock}`);
}
elements.get("timeScale").value = "1";
elements.get("timeScale").listeners.change();
elements.get("resetBtn").listeners.click();

const constantValues = {
  ALIGN_DIST_M: "2.2", ALIGN_BAND_M: "0.3", YAW_TOL_DEG: "2", OFF_TOL_M: "0.12",
  WIDTH_MIN_FULL: "0", CMD_STABLE_THR: "5", REL_YAW_TARGET_DEG: "90", STOP_SEC: "1.2",
  INSERT_FWD_MPS: "0.25"
};
Object.entries(constantValues).forEach(([key, value]) => {
  document.getElementById(`constant_${key}`).value = value;
});
elements.get("applyConstantsBtn").listeners.click();
if (!elements.get("constantStatus").textContent.includes("적용 완료")) {
  throw new Error("editable FSM constants did not apply");
}
if (!elements.get("fsmDiagram")._diagramNodes.some((node) => node.classList.values.has("active"))) {
  throw new Error("realtime diagram has no active state");
}
if (!elements.get("trajectorySummary").textContent.includes("points")) {
  throw new Error("live trajectory summary did not update");
}
if (!canvasTextCalls.trajectoryChart.some((args) => String(args[0]).includes("time"))
    || !canvasTextCalls.trajectoryChart.some((args) => String(args[0]).includes("forward"))) {
  throw new Error("3D trajectory chart is missing readable time/forward axes");
}

elements.get("initialDist").value = "2.20";
elements.get("initialYaw").value = "0.0";
elements.get("initialOffset").value = "0.10";
elements.get("applyScenarioBtn").listeners.click();
elements.get("runBtn").listeners.click();
for (currentTimeMs += 1000 / 60; currentTimeMs <= 11000; currentTimeMs += 1000 / 60) {
  const callback = animationCallback;
  callback(currentTimeMs);
  if (elements.get("safetyPill").textContent === "FORK COLLISION") break;
}
if (elements.get("safetyPill").textContent !== "FORK COLLISION") {
  throw new Error("collision preset did not stop with FORK COLLISION");
}
if (elements.get("runBtn").textContent !== "Run") throw new Error("collision did not pause automatic run");
if (!elements.get("fsmDiagram")._diagramNodes.some((node) =>
  node.dataset.state === "INSERT_FORWARD" && node.classList.values.has("active"))) {
  throw new Error("collision did not leave INSERT_FORWARD highlighted");
}
if (canvasTextCalls.simCanvas.length !== 0) {
  throw new Error(`main simulation canvas rendered ${canvasTextCalls.simCanvas.length} text labels`);
}
if (canvasTextCalls.motionChart.length === 0) throw new Error("motion chart text instrumentation is not active");

elements.get("runBatchBtn").listeners.click();
const batchHtml = elements.get("batchResults").innerHTML;
const batchRows = (batchHtml.match(/<tr class=/g) || []).length;
if (batchRows !== 27) throw new Error(`batch rendered ${batchRows} rows instead of 27`);
const batchSummary = elements.get("batchSummary").textContent;
const batchCounts = batchSummary.match(/27개 완료 · 성공 (\d+) · 실패 (\d+)/);
if (!batchCounts || Number(batchCounts[1]) + Number(batchCounts[2]) !== 27) {
  throw new Error(`invalid batch summary: ${batchSummary}`);
}
if (batchHtml.includes('data-action="log"') || !batchHtml.includes('data-action="save"')) {
  throw new Error("batch action buttons are not trajectory/replay/save");
}
elements.get("batchResults").listeners.click({ target: { dataset: { action: "trajectory", index: "0" } } });
if (!elements.get("trajectorySummary").textContent.includes("배치 #01")) {
  throw new Error("batch trajectory did not replace the trajectory chart");
}
elements.get("batchResults").listeners.click({ target: { dataset: { action: "replay", index: "0" } } });
if (elements.get("initialOffset").value !== "-0.30"
    || elements.get("initialDist").value !== "1.80"
    || elements.get("initialYaw").value !== "15.0") {
  throw new Error("batch replay did not restore exact condition #01 inputs");
}
if (!elements.get("logBox").innerHTML.includes("배치 조건 #01 재현 시작")) {
  throw new Error("batch replay marker missing from simulator log");
}
if (!elements.get("logBox").innerHTML.includes("drive seed 20261722")
    || !elements.get("logBox").innerHTML.includes("estimation seed 20261723")) {
  throw new Error("batch replay did not restore the fast-run drive/estimation seeds");
}
if (elements.get("runBtn").textContent !== "Pause") {
  throw new Error("batch replay did not start automatic animation");
}
elements.get("timeScale").value = "8";
elements.get("timeScale").listeners.change();
const replayDeadlineMs = currentTimeMs + 20000;
for (currentTimeMs += 1000 / 60; currentTimeMs <= replayDeadlineMs; currentTimeMs += 1000 / 60) {
  animationCallback(currentTimeMs);
  if (elements.get("runBtn").textContent === "Run") break;
}
if (!elements.get("batchSummary").textContent.includes("정확 재현 일치")) {
  throw new Error(`batch replay did not reproduce verdict/time exactly: ${elements.get("batchSummary").textContent}`);
}
if (!elements.get("logBox").innerHTML.includes("REPLAY MATCH")) {
  throw new Error("batch replay match was not recorded in the transition log");
}
const failedReplayMatch = batchHtml.match(/<tr class="batch-failure">[\s\S]*?data-action="replay" data-index="(\d+)"/);
if (!failedReplayMatch) throw new Error("stochastic batch produced no failed row for exact-failure replay test");
const failedReplayIndex = Number(failedReplayMatch[1]);
elements.get("batchResults").listeners.click({
  target: { dataset: { action: "replay", index: String(failedReplayIndex) } }
});
const failedReplayDeadlineMs = currentTimeMs + 20000;
for (currentTimeMs += 1000 / 60; currentTimeMs <= failedReplayDeadlineMs; currentTimeMs += 1000 / 60) {
  animationCallback(currentTimeMs);
  if (elements.get("runBtn").textContent === "Run") break;
}
if (!elements.get("batchSummary").textContent.includes("정확 재현 일치")
    || !elements.get("batchSummary").textContent.includes("판정 실패/실패")) {
  throw new Error(`failed batch row did not replay as the same failure: ${elements.get("batchSummary").textContent}`);
}
elements.get("timeScale").value = "1";
elements.get("timeScale").listeners.change();
elements.get("startVideoBtn").listeners.click();
if (!elements.get("videoStatus").textContent.includes("지원하지 않습니다")) {
  throw new Error("video recorder did not report unsupported test-browser capability");
}

async function verifyGridSearchUi() {
  await elements.get("runGridSearchBtn").listeners.click();
  const searchStatus = elements.get("gridSearchStatus").textContent;
  if (!searchStatus.includes("완료 · 1조합 × 27조건 × 1회 = 27 runs")) {
    throw new Error(`grid search did not complete its requested 27 runs: ${searchStatus}`);
  }
  const resultRows = (elements.get("gridSearchResults").innerHTML.match(/<tr class=/g) || []).length;
  if (resultRows !== 1) throw new Error(`grid search rendered ${resultRows} rows instead of 1`);
  if (!elements.get("gridSearchBest").textContent.includes("YAW_TOL_DEG=3")) {
    throw new Error("grid search did not select the only candidate combination");
  }
  if (elements.get("applyBestGridBtn").disabled) throw new Error("best-grid apply button stayed disabled");
  elements.get("applyBestGridBtn").listeners.click();
  if (!elements.get("constantGrid").innerHTML.includes('id="constant_YAW_TOL_DEG"')
      || !elements.get("constantGrid").innerHTML.includes('value="3"')) {
    throw new Error("best grid-search parameters were not applied to the simulator override");
  }
  console.log(`PASS browser bootstrap/realtime/diagram/collision/batch/replay/grid-search/text-free canvas smoke: ${clock} · ${batchSummary}`);
}

verifyGridSearchUi().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
