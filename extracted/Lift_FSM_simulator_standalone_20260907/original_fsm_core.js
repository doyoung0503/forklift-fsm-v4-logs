(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.OriginalFSM = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const CONFIG = Object.freeze({
    YAW_TOL_DEG: 2.0,
    OFF_TOL_M: 0.12,
    WIDTH_MIN_FULL: 0.0,
    ALIGN_DIST_M: 2.20,
    ALIGN_BAND_M: 0.30,
    CMD_STABLE_THR: 5,
    REL_YAW_TARGET_DEG: 85.0,
    STOP_SEC: 1.2,
    USE_PIECEWISE_FWD_FIT: true,
    FWD_T0: -0.0202,
    FWD_T1: 4.2780,
    FWD_A: 0.071565,
    FWD_SCALE: 1.0,
    FWD_BIAS: 0.0,
    FWD_MIN_SEC: 1.0,
    FWD_MAX_SEC: 15.0,
    PALLET_POCKET_M: 0.0,
    INSERT_FWD_MPS: 0.25,
    INS_FWD_MIN_SEC: 0.5,
    INS_FWD_MAX_SEC: 10.0
  });

  const SIM_CONFIG = Object.freeze({
    PALLET_FRONT_WIDTH_M: 1.10,
    PALLET_DEPTH_M: 1.30,
    PALLET_POCKET_WIDTH_M: 0.25,
    PALLET_POCKET_INNER_GAP_M: 0.30,
    FORK_OUTER_SPAN_M: 0.60,
    FORK_TINE_WIDTH_M: 0.10,
    FORK_FORWARD_ROOT_M: 0.57,
    FORK_FORWARD_TIP_M: 1.55,
    CAMERA_HFOV_DEG: 69.4,
    CAMERA_RANGE_M: 3.50,
    CAMERA_MOUNT_FORWARD_M: 0.45,
    DETECTION_VISIBLE_FRACTION: 0.50,
    GRID_SIZE_M: 1.00
  });

  const TOP_STATES = Object.freeze(["SEARCH", "DETECTED", "RECOVER", "CHECK", "ALIGN", "DONE"]);
  const ALIGN_STATES = Object.freeze([
    "DIST_CHECK", "ALIGN_FWD_ADJUST", "ALIGN_BWD_ADJUST", "YAW_CHECK",
    "ROTATE_RIGHT_UNTIL_YAW_TOL", "ROTATE_LEFT_UNTIL_YAW_TOL", "OFFSET_CHECK",
    "ALIGN_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT", "ALIGN_ROTATE_LEFT_90",
    "ALIGN_ROTATE_LEFT", "FORWARD_AFTER_LEFT", "ALIGN_ROTATE_RIGHT_90",
    "INSERT_FORWARD", "READY_TO_DONE"
  ]);

  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function wrap180(deg) {
    let value = ((deg + 180) % 360 + 360) % 360 - 180;
    if (value === -180) value = 180;
    return value;
  }
  function withinBand(value, target, band) {
    return value !== null && value !== undefined && Math.abs(value - target) <= band;
  }

  class Stabilizer {
    constructor(threshold = CONFIG.CMD_STABLE_THR) {
      this.threshold = threshold;
      this.reset();
    }
    reset() { this.tag = null; this.count = 0; }
    stable(tag) {
      if (this.tag === tag) this.count += 1;
      else { this.tag = tag; this.count = 1; }
      return this.count >= this.threshold;
    }
  }

  function timeFromDistancePiecewise(distance, cfg = CONFIG) {
    const d = Math.max(0, distance);
    const dAcc = 0.5 * cfg.FWD_A * cfg.FWD_T1 ** 2;
    const vmax = Math.max(1e-6, cfg.FWD_A * cfg.FWD_T1);
    const seconds = d <= dAcc
      ? cfg.FWD_T0 + Math.sqrt(2 * d / Math.max(1e-9, cfg.FWD_A))
      : cfg.FWD_T0 + cfg.FWD_T1 + (d - dAcc) / vmax;
    return clamp(seconds, cfg.FWD_MIN_SEC, cfg.FWD_MAX_SEC);
  }

  function fwdSecFromOffset(offset, cfg = CONFIG) {
    const effectiveDistance = Math.max(0, cfg.FWD_SCALE * Math.abs(offset) + cfg.FWD_BIAS);
    return timeFromDistancePiecewise(effectiveDistance, cfg);
  }

  function insertionSeconds(distZ, cfg = CONFIG) {
    if (distZ === null || distZ === undefined) return null;
    const total = Math.max(0, Number(distZ) + cfg.PALLET_POCKET_M);
    const seconds = total / Math.max(1e-3, cfg.INSERT_FWD_MPS);
    return clamp(seconds, cfg.INS_FWD_MIN_SEC, cfg.INS_FWD_MAX_SEC);
  }

  function rawMotionDistanceAtTime(seconds, cfg = CONFIG) {
    const tau = Math.max(0, Number(seconds) - cfg.FWD_T0);
    const dAcc = 0.5 * cfg.FWD_A * cfg.FWD_T1 ** 2;
    const vmax = cfg.FWD_A * cfg.FWD_T1;
    return tau <= cfg.FWD_T1
      ? 0.5 * cfg.FWD_A * tau ** 2
      : dAcc + vmax * (tau - cfg.FWD_T1);
  }

  function rawMotionVelocityAtTime(seconds, cfg = CONFIG) {
    const tau = Math.max(0, Number(seconds) - cfg.FWD_T0);
    return tau <= cfg.FWD_T1 ? cfg.FWD_A * tau : cfg.FWD_A * cfg.FWD_T1;
  }

  function distanceFromDurationPiecewise(duration, cfg = CONFIG) {
    const seconds = Math.max(0, Number(duration) || 0);
    const base = rawMotionDistanceAtTime(0, cfg);
    return Math.max(0, rawMotionDistanceAtTime(seconds, cfg) - base);
  }

  function clipPolygonAtY(polygon, boundaryY, keepGreater) {
    if (!polygon.length) return [];
    const result = [];
    const inside = (point) => keepGreater ? point.y >= boundaryY - 1e-10 : point.y <= boundaryY + 1e-10;
    for (let index = 0; index < polygon.length; index += 1) {
      const current = polygon[index];
      const previous = polygon[(index + polygon.length - 1) % polygon.length];
      const currentInside = inside(current);
      const previousInside = inside(previous);
      if (currentInside !== previousInside) {
        const dy = current.y - previous.y;
        const ratio = Math.abs(dy) < 1e-12 ? 0 : (boundaryY - previous.y) / dy;
        result.push({
          x: previous.x + (current.x - previous.x) * ratio,
          y: boundaryY
        });
      }
      if (currentInside) result.push(current);
    }
    return result;
  }

  function forkPalletCollisionAtPose(options = {}) {
    const simCfg = options.simConfig || SIM_CONFIG;
    const bodyX = Number(options.bodyX) || 0;
    const bodyY = Number(options.bodyY) || 0;
    const headingDeg = Number(options.headingDeg ?? 90);
    const faceCenter = options.palletFaceCenter || { x: 0, y: 2.20 };
    const palletHeadingDeg = Number(options.palletHeadingDeg ?? 90);
    const palletDepth = Number(options.palletDepth ?? simCfg.PALLET_DEPTH_M);
    const tineWidth = clamp(
      Number(options.forkTineWidth ?? simCfg.FORK_TINE_WIDTH_M),
      0.001,
      simCfg.PALLET_POCKET_WIDTH_M
    );
    const pocketHalfGap = simCfg.PALLET_POCKET_INNER_GAP_M / 2;
    const pocketWidth = simCfg.PALLET_POCKET_WIDTH_M;
    const pockets = {
      left: {
        minX: -pocketHalfGap - pocketWidth,
        maxX: -pocketHalfGap
      },
      right: {
        minX: pocketHalfGap,
        maxX: pocketHalfGap + pocketWidth
      }
    };
    const outerHalf = simCfg.FORK_OUTER_SPAN_M / 2;
    const tines = {
      left: { minLateral: -outerHalf, maxLateral: -outerHalf + tineWidth },
      right: { minLateral: outerHalf - tineWidth, maxLateral: outerHalf }
    };
    const heading = direction(headingDeg);
    const right = { x: heading.y, y: -heading.x };
    const pointAt = (lateral, forward) => ({
      x: bodyX + right.x * lateral + heading.x * forward,
      y: bodyY + right.y * lateral + heading.y * forward
    });
    const palletHeading = direction(palletHeadingDeg);
    const palletRight = { x: palletHeading.y, y: -palletHeading.x };
    const toPalletLocal = (point) => {
      const relative = { x: point.x - faceCenter.x, y: point.y - faceCenter.y };
      return { x: dot(relative, palletRight), y: dot(relative, palletHeading) };
    };
    const details = [];
    let inserting = false;
    let collided = false;
    let minimumClearance = Infinity;

    for (const name of ["left", "right"]) {
      const tine = tines[name];
      let polygon = [
        pointAt(tine.minLateral, simCfg.FORK_FORWARD_ROOT_M),
        pointAt(tine.maxLateral, simCfg.FORK_FORWARD_ROOT_M),
        pointAt(tine.maxLateral, simCfg.FORK_FORWARD_TIP_M),
        pointAt(tine.minLateral, simCfg.FORK_FORWARD_TIP_M)
      ].map(toPalletLocal);
      polygon = clipPolygonAtY(polygon, 0, true);
      polygon = clipPolygonAtY(polygon, palletDepth, false);
      if (!polygon.length) {
        details.push({ name, penetrating: false, collision: false, clearanceM: null });
        continue;
      }
      inserting = true;
      const minX = Math.min(...polygon.map((point) => point.x));
      const maxX = Math.max(...polygon.map((point) => point.x));
      const pocket = pockets[name];
      const clearance = Math.min(minX - pocket.minX, pocket.maxX - maxX);
      const tineCollision = clearance < -1e-9;
      collided ||= tineCollision;
      minimumClearance = Math.min(minimumClearance, clearance);
      details.push({
        name,
        penetrating: true,
        collision: tineCollision,
        clearanceM: clearance,
        occupiedMinX: minX,
        occupiedMaxX: maxX,
        pocketMinX: pocket.minX,
        pocketMaxX: pocket.maxX
      });
    }

    return {
      detected: collided,
      inserting,
      clearanceM: inserting ? minimumClearance : null,
      forkTineWidthM: tineWidth,
      pockets,
      details,
      reason: collided
        ? "포크 단면이 25 cm 포켓 경계를 침범"
        : inserting ? "두 포크가 포켓 내부" : "포크가 팔레트 전면에 도달하기 전"
    };
  }

  function cross(a, b) { return a.x * b.y - a.y * b.x; }
  function dot(a, b) { return a.x * b.x + a.y * b.y; }
  function direction(degrees) {
    const radians = degrees * Math.PI / 180;
    return { x: Math.cos(radians), y: Math.sin(radians) };
  }

  function clipLinearInterval(interval, valueAtZero, slope) {
    const epsilon = 1e-10;
    if (Math.abs(slope) < epsilon) return valueAtZero >= -epsilon;
    const root = -valueAtZero / slope;
    if (slope > 0) interval.lo = Math.max(interval.lo, root);
    else interval.hi = Math.min(interval.hi, root);
    return interval.lo <= interval.hi + epsilon;
  }

  function computePalletFrontVisibility(options) {
    const camera = options.camera;
    const start = options.frontStart;
    const end = options.frontEnd;
    const center = { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 };
    const outwardNormal = options.outwardNormal || { x: 0, y: -1 };
    const fovDeg = Number(options.fovDeg ?? SIM_CONFIG.CAMERA_HFOV_DEG);
    const rangeM = Math.max(0, Number(options.rangeM ?? SIM_CONFIG.CAMERA_RANGE_M));
    const cameraToFace = { x: camera.x - center.x, y: camera.y - center.y };
    const frontFacing = dot(outwardNormal, cameraToFace) > 0;
    const result = {
      fraction: 0,
      visibleLength: 0,
      frontLength: Math.hypot(end.x - start.x, end.y - start.y),
      frontFacing,
      interval: null,
      visibleStart: null,
      visibleEnd: null
    };
    if (!frontFacing || result.frontLength <= 1e-10 || fovDeg <= 0 || rangeM <= 0) return result;

    const v0 = { x: start.x - camera.x, y: start.y - camera.y };
    const segment = { x: end.x - start.x, y: end.y - start.y };
    const half = fovDeg / 2;
    const rightBoundary = direction(camera.headingDeg - half);
    const leftBoundary = direction(camera.headingDeg + half);
    const interval = { lo: 0, hi: 1 };

    if (!clipLinearInterval(interval, cross(rightBoundary, v0), cross(rightBoundary, segment))) return result;
    if (!clipLinearInterval(interval, cross(v0, leftBoundary), cross(segment, leftBoundary))) return result;

    const quadraticA = dot(segment, segment);
    const quadraticB = dot(v0, segment);
    const quadraticC = dot(v0, v0) - rangeM ** 2;
    const discriminant = quadraticB ** 2 - quadraticA * quadraticC;
    if (discriminant < 0) return result;
    const root = Math.sqrt(Math.max(0, discriminant));
    const circleLo = (-quadraticB - root) / quadraticA;
    const circleHi = (-quadraticB + root) / quadraticA;
    interval.lo = Math.max(interval.lo, circleLo, 0);
    interval.hi = Math.min(interval.hi, circleHi, 1);
    if (interval.lo > interval.hi) return result;

    result.fraction = clamp(interval.hi - interval.lo, 0, 1);
    result.visibleLength = result.frontLength * result.fraction;
    result.interval = { lo: interval.lo, hi: interval.hi };
    result.visibleStart = { x: start.x + segment.x * interval.lo, y: start.y + segment.y * interval.lo };
    result.visibleEnd = { x: start.x + segment.x * interval.hi, y: start.y + segment.y * interval.hi };
    return result;
  }

  class CommandExecutor {
    constructor(onTransaction) {
      this.onTransaction = onTransaction || (() => {});
      this.lastCommand = null;
      this.lastDirection = +1;
    }
    resetLast() { this.lastCommand = null; }
    execute(rawCommand, now) {
      const command = rawCommand === "SPIN_LEFT_UNTIL_DETECTED" ? "ROT_LEFT"
        : rawCommand === "SPIN_RIGHT_UNTIL_DETECTED" ? "ROT_RIGHT" : rawCommand;
      if (command === this.lastCommand) return false;
      this.lastCommand = command;
      if (["ROT_LEFT", "FWD_LEFT", "BACK_LEFT"].includes(command)) this.lastDirection = +1;
      if (["ROT_RIGHT", "FWD_RIGHT", "BACK_RIGHT"].includes(command)) this.lastDirection = -1;
      this.onTransaction({ command, now });
      return true;
    }
  }

  class CalibrationFSM {
    constructor(options = {}) {
      this.cfg = Object.freeze({ ...CONFIG, ...(options.config || {}) });
      this.commandTransactions = [];
      this.executor = new CommandExecutor((tx) => {
        this.command = tx.command;
        this.commandTransactions.push(tx);
        this.lastTransaction = tx;
      });
      this.reset();
    }

    reset() {
      this.now = 0;
      this.topState = "SEARCH";
      this.command = "STOP";
      this.events = [];
      this.history = [];
      this.lastTransaction = null;
      this.commandTransactions.length = 0;
      this.executor.lastCommand = null;
      this.executor.lastDirection = +1;
      this.checkStabilizer = new Stabilizer(this.cfg.CMD_STABLE_THR);
      this.recoverStabilizer = new Stabilizer(this.cfg.CMD_STABLE_THR);
      this.topLastCommand = null;
      this.topLastTimestamp = 0;
      this.alignLastCommand = null;
      this.alignLastTimestamp = 0;
      this.resetAlign("DIST_CHECK");
      this.resetRecover();
      this.record("FSM reset → SEARCH");
    }

    get statePath() {
      if (this.topState === "ALIGN") return `ALIGN.${this.alignSub}`;
      if (this.topState === "RECOVER") return `RECOVER.${this.recoverSub}`;
      return this.topState;
    }

    record(message) {
      this.events.push(message);
      this.history.push({ t: this.now, state: this.statePath, command: this.command, message });
    }

    setTop(next, reason) {
      const previous = this.topState;
      this.topState = next;
      this.record(`${previous} → ${next}${reason ? ` · ${reason}` : ""}`);
    }

    setAlign(next, reason) {
      const previous = this.alignSub;
      this.alignSub = next;
      this.record(`ALIGN.${previous} → ALIGN.${next}${reason ? ` · ${reason}` : ""}`);
    }

    resetAlign(sub = "DIST_CHECK") {
      this.alignSub = sub;
      this.alignStabilizer = new Stabilizer(this.cfg.CMD_STABLE_THR);
      this.interlockActive = false;
      this.interlockUntil = 0;
      this.afterInterlockSub = null;
      this.relYawRef = null;
      this.fwdSecCached = 0;
      this.fwdDeadline = null;
      this.insertSecCached = null;
      this.insertDeadline = null;
      this.latestDistZ = null;
      this.rotationCommandTargetDeg = null;
    }

    resetRecover() {
      this.recoverSub = "DECIDE_TURN";
      this.recoverStabilizer.reset();
    }

    exec(command) { this.executor.execute(command, this.now); }

    alignExec(command) {
      if (this.alignLastCommand === command && (this.now - this.alignLastTimestamp) < 0.10) return;
      this.exec(command);
      this.alignLastCommand = command;
      this.alignLastTimestamp = this.now;
    }

    ensureStop() {
      if (this.topLastCommand !== "STOP" || (this.now - this.topLastTimestamp) > 0.2) {
        this.exec("STOP");
        this.topLastCommand = "STOP";
        this.topLastTimestamp = this.now;
      }
    }

    startInterlock(nextSub) {
      this.interlockActive = true;
      this.afterInterlockSub = nextSub;
      this.alignExec("STOP");
      this.interlockUntil = this.now + this.cfg.STOP_SEC;
      this.record(`STOP 인터록 ${this.cfg.STOP_SEC.toFixed(1)}s → ALIGN.${nextSub}`);
    }

    finishInterlockIfNeeded() {
      if (!this.interlockActive) return false;
      this.alignExec("STOP");
      if (this.now >= this.interlockUntil) {
        this.interlockActive = false;
        if (this.afterInterlockSub) {
          const next = this.afterInterlockSub;
          this.afterInterlockSub = null;
          this.setAlign(next, "STOP 인터록 완료");
        }
      }
      return true;
    }

    step(rawSensors, dt = 1 / 15) {
      this.now += Math.max(0, Number(dt) || 0);
      this.events = [];
      const sensors = {
        detOk: Boolean(rawSensors.detOk),
        detectedLength: rawSensors.detectedLength ?? null,
        distZ: rawSensors.distZ ?? null,
        yaw: rawSensors.yaw ?? null,
        offsetX: rawSensors.offsetX ?? null,
        relYaw: rawSensors.relYaw ?? null,
        rotationDone: Boolean(rawSensors.rotationDone)
      };

      const yawOk = sensors.yaw !== null && Math.abs(sensors.yaw) <= this.cfg.YAW_TOL_DEG;
      const offOk = sensors.offsetX !== null && Math.abs(sensors.offsetX) <= this.cfg.OFF_TOL_M;

      if (this.topState === "SEARCH") {
        this.resetAlign("DIST_CHECK");
        this.resetRecover();
        this.executor.resetLast();
        if (sensors.detOk) this.setTop("DETECTED", "front 탐지");
        else this.ensureStop();
        return this.snapshot(sensors);
      }

      if (this.topState === "DETECTED") {
        if (!sensors.detOk) {
          this.setTop("SEARCH", "탐지 유실");
          this.exec("STOP");
          return this.snapshot(sensors);
        }
        if (sensors.detectedLength !== null && sensors.detectedLength >= this.cfg.WIDTH_MIN_FULL) {
          this.setTop("ALIGN", `폭 ${sensors.detectedLength.toFixed(3)}m`);
          this.resetAlign("DIST_CHECK");
        } else {
          this.setTop("RECOVER", "시야 확보 필요");
          this.resetRecover();
        }
        return this.snapshot(sensors);
      }

      if (this.topState === "RECOVER") {
        this.recoverStep(sensors);
        if (this.recoverSub === "HOLD") {
          this.setTop("CHECK", "RECOVER HOLD");
          this.checkStabilizer.reset();
        }
        return this.snapshot(sensors);
      }

      if (this.topState === "CHECK") {
        this.ensureStop();
        const bothOk = yawOk && offOk;
        if (this.checkStabilizer.stable(bothOk ? "CHECK_OK" : "CHECK_NOK")) {
          if (bothOk) this.setTop("DONE", "yaw/offset 연속 판정 완료");
          else {
            this.setTop("ALIGN", "재정렬 필요");
            this.resetAlign("DIST_CHECK");
          }
        }
        return this.snapshot(sensors);
      }

      if (this.topState === "ALIGN") {
        this.alignStep(sensors);
        if (this.alignSub === "READY_TO_DONE") {
          this.setTop("DONE", "정렬 및 포켓 삽입 완료");
          this.ensureStop();
        }
        return this.snapshot(sensors);
      }

      if (this.topState === "DONE") {
        this.ensureStop();
        return this.snapshot(sensors);
      }

      this.ensureStop();
      return this.snapshot(sensors);
    }

    recoverStep(s) {
      if (!s.detOk) {
        if (!this.recoverSub.startsWith("RECOVER_ROTATE_")) {
          this.exec(this.executor.lastDirection > 0 ? "ROT_LEFT" : "ROT_RIGHT");
        }
        return;
      }
      if (this.recoverSub === "DECIDE_TURN") {
        let tag;
        let direction;
        if (s.offsetX === null || Math.abs(s.offsetX) <= this.cfg.OFF_TOL_M) {
          tag = "DECIDE_CENTER";
          direction = this.executor.lastDirection > 0 ? "LEFT" : "RIGHT";
        } else if (s.offsetX > 0) {
          tag = "DECIDE_RIGHT";
          direction = "RIGHT";
        } else {
          tag = "DECIDE_LEFT";
          direction = "LEFT";
        }
        if (this.recoverStabilizer.stable(tag)) {
          this.recoverSub = `RECOVER_ROTATE_${direction}`;
          this.exec(`ROT_${direction}`);
          this.record(`RECOVER.DECIDE_TURN → RECOVER.${this.recoverSub}`);
        }
        return;
      }
      if (["RECOVER_ROTATE_LEFT", "RECOVER_ROTATE_RIGHT"].includes(this.recoverSub)) {
        if (s.detectedLength !== null && s.detectedLength >= this.cfg.WIDTH_MIN_FULL) {
          this.recoverSub = "HOLD";
          this.record("RECOVER 회전 → HOLD · 전면 폭 확보");
        } else {
          this.exec(this.recoverSub.endsWith("LEFT") ? "ROT_LEFT" : "ROT_RIGHT");
        }
      }
    }

    alignStep(s) {
      if (s.distZ !== null) this.latestDistZ = s.distZ;
      const yawOk = s.yaw !== null && Math.abs(s.yaw) <= this.cfg.YAW_TOL_DEG;
      const bandOk = withinBand(s.distZ, this.cfg.ALIGN_DIST_M, this.cfg.ALIGN_BAND_M);
      const chainStates = [
        "ALIGN_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT", "ALIGN_ROTATE_LEFT_90",
        "ALIGN_ROTATE_LEFT", "FORWARD_AFTER_LEFT", "ALIGN_ROTATE_RIGHT_90",
        "ROTATE_RIGHT_UNTIL_YAW_TOL", "ROTATE_LEFT_UNTIL_YAW_TOL", "INSERT_FORWARD"
      ];
      if (!s.detOk && !chainStates.includes(this.alignSub)) {
        this.alignExec(this.executor.lastDirection > 0 ? "ROT_LEFT" : "ROT_RIGHT");
        return;
      }
      if (this.finishInterlockIfNeeded()) return;

      if (this.alignSub === "DIST_CHECK") {
        this.alignExec("STOP");
        if (bandOk) {
          if (this.alignStabilizer.stable("DIST_BAND_OK")) {
            this.resetAlign("YAW_CHECK");
            this.record("ALIGN.DIST_CHECK → ALIGN.YAW_CHECK · 거리 밴드 OK");
          }
        } else if (s.distZ !== null && s.distZ > this.cfg.ALIGN_DIST_M) {
          if (this.alignStabilizer.stable("DIST_FWD")) {
            this.resetAlign("ALIGN_FWD_ADJUST");
            this.record("ALIGN.DIST_CHECK → ALIGN.ALIGN_FWD_ADJUST");
          }
        } else if (s.distZ !== null && s.distZ < this.cfg.ALIGN_DIST_M) {
          if (this.alignStabilizer.stable("DIST_BWD")) {
            this.resetAlign("ALIGN_BWD_ADJUST");
            this.record("ALIGN.DIST_CHECK → ALIGN.ALIGN_BWD_ADJUST");
          }
        }
        return;
      }

      if (this.alignSub === "ALIGN_FWD_ADJUST") {
        if (s.distZ !== null && Math.abs(s.distZ - this.cfg.ALIGN_DIST_M) > this.cfg.ALIGN_BAND_M) this.alignExec("FWD");
        else this.startInterlock("DIST_CHECK");
        return;
      }

      if (this.alignSub === "ALIGN_BWD_ADJUST") {
        if (s.distZ !== null && Math.abs(s.distZ - this.cfg.ALIGN_DIST_M) > this.cfg.ALIGN_BAND_M) this.alignExec("BACK");
        else this.startInterlock("DIST_CHECK");
        return;
      }

      if (this.alignSub === "YAW_CHECK") {
        this.alignExec("STOP");
        if (s.yaw !== null && Math.abs(s.yaw) > this.cfg.YAW_TOL_DEG) {
          const tag = s.yaw > 0 ? "YAW_POS" : "YAW_NEG";
          if (this.alignStabilizer.stable(tag)) {
            const next = s.yaw > 0 ? "ROTATE_RIGHT_UNTIL_YAW_TOL" : "ROTATE_LEFT_UNTIL_YAW_TOL";
            if (this.cfg.ROTATION_ENDPOINT_VALIDATION) {
              this.rotationCommandTargetDeg = Math.abs(s.yaw);
            }
            this.setAlign(next, `yaw=${s.yaw.toFixed(2)}°`);
            this.alignExec(s.yaw > 0 ? "ROT_RIGHT" : "ROT_LEFT");
          }
        } else if (this.alignStabilizer.stable("YAW_OK")) {
          this.resetAlign("OFFSET_CHECK");
          this.record("ALIGN.YAW_CHECK → ALIGN.OFFSET_CHECK · yaw 허용치");
        }
        return;
      }

      if (["ROTATE_RIGHT_UNTIL_YAW_TOL", "ROTATE_LEFT_UNTIL_YAW_TOL"].includes(this.alignSub)) {
        this.alignExec(this.alignSub.startsWith("ROTATE_RIGHT") ? "ROT_RIGHT" : "ROT_LEFT");
        if (this.cfg.ROTATION_ENDPOINT_VALIDATION) {
          if (s.rotationDone) {
            this.rotationCommandTargetDeg = null;
            this.alignStabilizer.reset();
            this.startInterlock("YAW_CHECK");
            this.record("회전 명령 종점 도달 → yaw 잔차 재검사");
          }
        } else if (yawOk) this.startInterlock("OFFSET_CHECK");
        return;
      }

      if (this.alignSub === "OFFSET_CHECK") {
        if (s.offsetX === null) {
          if (this.alignStabilizer.stable("OFF_NA")) {
            this.resetAlign("DIST_CHECK");
            this.record("ALIGN.OFFSET_CHECK → ALIGN.DIST_CHECK · offset N/A");
          }
          return;
        }
        if (Math.abs(s.offsetX) > this.cfg.OFF_TOL_M) {
          this.fwdSecCached = Math.max(0.1, this.cfg.USE_PIECEWISE_FWD_FIT ? fwdSecFromOffset(s.offsetX, this.cfg) : 2.0);
          const right = s.offsetX > 0;
          if (this.alignStabilizer.stable(right ? "OFF_RIGHT" : "OFF_LEFT")) {
            this.relYawRef = s.relYaw;
            this.alignSub = right ? "ALIGN_ROTATE_RIGHT" : "ALIGN_ROTATE_LEFT";
            this.alignExec(right ? "ROT_RIGHT" : "ROT_LEFT");
            this.fwdDeadline = null;
            this.record(`ALIGN.OFFSET_CHECK → ALIGN.${this.alignSub} · FWD_SEC=${this.fwdSecCached.toFixed(3)}s`);
          }
          return;
        }
        if (s.yaw !== null && Math.abs(s.yaw) > this.cfg.YAW_TOL_DEG) {
          if (this.alignStabilizer.stable("OFF_OK_YAW_NOK")) {
            this.resetAlign("DIST_CHECK");
            this.record("ALIGN.OFFSET_CHECK → ALIGN.DIST_CHECK · yaw 재보정");
          }
          return;
        }
        if (this.alignStabilizer.stable("READY_TO_DONE")) {
          this.insertSecCached = insertionSeconds(this.latestDistZ, this.cfg);
          if (this.insertSecCached !== null && this.insertSecCached > 0) this.startInterlock("INSERT_FORWARD");
          else {
            this.alignSub = "READY_TO_DONE";
            this.alignExec("STOP");
            this.record("ALIGN.OFFSET_CHECK → ALIGN.READY_TO_DONE · dist N/A, 삽입 생략");
          }
        }
        return;
      }

      if (this.alignSub === "ALIGN_ROTATE_RIGHT") {
        this.alignExec("ROT_RIGHT");
        if (this.cfg.ROTATION_ENDPOINT_VALIDATION && s.rotationDone) {
          this.startInterlock("FORWARD_AFTER_RIGHT");
        } else if (!this.cfg.ROTATION_ENDPOINT_VALIDATION && s.relYaw !== null && this.relYawRef !== null) {
          const delta = wrap180(s.relYaw - this.relYawRef);
          if (delta >= this.cfg.REL_YAW_TARGET_DEG) this.startInterlock("FORWARD_AFTER_RIGHT");
        }
        return;
      }

      if (this.alignSub === "ALIGN_ROTATE_LEFT") {
        this.alignExec("ROT_LEFT");
        if (this.cfg.ROTATION_ENDPOINT_VALIDATION && s.rotationDone) {
          this.startInterlock("FORWARD_AFTER_LEFT");
        } else if (!this.cfg.ROTATION_ENDPOINT_VALIDATION && s.relYaw !== null && this.relYawRef !== null) {
          const delta = wrap180(s.relYaw - this.relYawRef);
          if (delta <= -this.cfg.REL_YAW_TARGET_DEG) this.startInterlock("FORWARD_AFTER_LEFT");
        }
        return;
      }

      if (["FORWARD_AFTER_RIGHT", "FORWARD_AFTER_LEFT"].includes(this.alignSub)) {
        this.alignExec("FWD");
        if (this.fwdDeadline === null) this.fwdDeadline = this.now + this.fwdSecCached;
        if (this.now >= this.fwdDeadline) {
          this.relYawRef = s.relYaw;
          this.fwdDeadline = null;
          this.startInterlock(this.alignSub.endsWith("RIGHT") ? "ALIGN_ROTATE_LEFT_90" : "ALIGN_ROTATE_RIGHT_90");
        }
        return;
      }

      if (this.alignSub === "ALIGN_ROTATE_LEFT_90") {
        this.alignExec("ROT_LEFT");
        const endpointDone = this.cfg.ROTATION_ENDPOINT_VALIDATION && s.rotationDone;
        const legacyDone = !this.cfg.ROTATION_ENDPOINT_VALIDATION && s.relYaw !== null && this.relYawRef !== null
          && wrap180(s.relYaw - this.relYawRef) <= -this.cfg.REL_YAW_TARGET_DEG;
        if (endpointDone || legacyDone) {
          this.relYawRef = null;
          this.alignStabilizer.reset();
          this.startInterlock("YAW_CHECK");
        }
        return;
      }

      if (this.alignSub === "ALIGN_ROTATE_RIGHT_90") {
        this.alignExec("ROT_RIGHT");
        const endpointDone = this.cfg.ROTATION_ENDPOINT_VALIDATION && s.rotationDone;
        const legacyDone = !this.cfg.ROTATION_ENDPOINT_VALIDATION && s.relYaw !== null && this.relYawRef !== null
          && wrap180(s.relYaw - this.relYawRef) >= this.cfg.REL_YAW_TARGET_DEG;
        if (endpointDone || legacyDone) {
          this.relYawRef = null;
          this.alignStabilizer.reset();
          this.startInterlock("YAW_CHECK");
        }
        return;
      }

      if (this.alignSub === "INSERT_FORWARD") {
        this.alignExec("FWD");
        if (this.insertDeadline === null) {
          if (this.insertSecCached === null || this.insertSecCached <= 0) {
            this.startInterlock("READY_TO_DONE");
            return;
          }
          this.insertDeadline = this.now + this.insertSecCached;
        }
        if (this.now >= this.insertDeadline) {
          this.insertDeadline = null;
          this.startInterlock("READY_TO_DONE");
        }
        return;
      }

      if (this.alignSub === "READY_TO_DONE") this.alignExec("STOP");
      else this.alignExec("STOP");
    }

    snapshot(sensors) {
      return {
        now: this.now,
        topState: this.topState,
        alignSub: this.alignSub,
        recoverSub: this.recoverSub,
        statePath: this.statePath,
        command: this.command,
        interlockActive: this.interlockActive,
        interlockRemaining: this.interlockActive ? Math.max(0, this.interlockUntil - this.now) : 0,
        stabilizerCount: this.topState === "CHECK" ? this.checkStabilizer.count
          : this.topState === "RECOVER" ? this.recoverStabilizer.count : this.alignStabilizer.count,
        stabilizerThreshold: this.cfg.CMD_STABLE_THR,
        fwdSecCached: this.fwdSecCached,
        insertSecCached: this.insertSecCached,
        rotationCommandTargetDeg: this.rotationCommandTargetDeg,
        events: [...this.events],
        sensors: { ...sensors },
        lastTransaction: this.lastTransaction
      };
    }
  }

  class IdealPlant {
    constructor(initial = {}, config = CONFIG, simConfig = SIM_CONFIG) {
      this.cfg = config;
      this.simCfg = simConfig;
      this.reset(initial);
    }
    reset(initial = {}) {
      this.rotateRate = Math.max(0.1, Number(initial.rotateRate ?? 45.0));
      this.forkTineWidth = clamp(
        Number(initial.forkTineWidth ?? this.simCfg.FORK_TINE_WIDTH_M),
        0.001,
        this.simCfg.PALLET_POCKET_WIDTH_M
      );
      this.relYaw = Number(initial.relYaw ?? 0.0);
      this.driveErrorModel = initial.driveErrorModel || null;
      this.driveRandom = typeof initial.driveRandom === "function" ? initial.driveRandom : Math.random;
      this.driveErrorEnabled = Boolean(initial.driveErrorEnabled && this.driveErrorModel);
      this.estimationErrorModel = initial.estimationErrorModel || null;
      this.estimationRandom = typeof initial.estimationRandom === "function" ? initial.estimationRandom : Math.random;
      this.estimationErrorEnabled = Boolean(initial.estimationErrorEnabled && this.estimationErrorModel);
      this.observedSensors = null;
      this.estimationSampleIndex = 0;
      this.lastEstimation = null;
      this.driveErrorEvents = [];
      this.activeRotationError = null;
      this.lastRotationError = null;
      this.rotationDone = false;
      this.detOk = initial.detOk ?? true;
      this.detectedLength = initial.detectedLength ?? this.simCfg.PALLET_FRONT_WIDTH_M;
      this.visibility = null;
      this.trace = [];
      this.activeMotion = null;
      this.lastMotion = null;
      this.completedMotions = [];
      this.lastAppliedCommand = "STOP";
      this.collision = {
        detected: false,
        inserting: false,
        clearanceM: null,
        reason: "포크가 팔레트 전면에 도달하기 전",
        details: []
      };

      const yaw = Number(initial.yaw ?? 12.0);
      const offset = Number(initial.offsetX ?? 0.42);
      const distance = Number(initial.distZ ?? 3.10);
      if (initial.fixedForkliftPose) {
        this.headingDeg = Number(initial.headingDeg ?? 90);
        this.bodyX = Number(initial.bodyX ?? 0);
        this.bodyY = Number(initial.bodyY ?? -1.15);
        const camera = this.cameraPose();
        const heading = direction(this.headingDeg);
        const right = { x: heading.y, y: -heading.x };
        this.palletFaceCenter = {
          x: camera.x + heading.x * distance + right.x * offset,
          y: camera.y + heading.y * distance + right.y * offset
        };
        this.palletHeadingDeg = wrap180(this.headingDeg - yaw);
      } else {
        this.palletFaceCenter = { x: 0, y: initial.palletFaceY ?? 2.20 };
        this.palletHeadingDeg = 90;
        this.headingDeg = 90 + yaw;
        const heading = direction(this.headingDeg);
        const right = { x: heading.y, y: -heading.x };
        const camera = {
          x: this.palletFaceCenter.x - right.x * offset - heading.x * distance,
          y: this.palletFaceCenter.y - right.y * offset - heading.y * distance
        };
        this.bodyX = camera.x - heading.x * this.simCfg.CAMERA_MOUNT_FORWARD_M;
        this.bodyY = camera.y - heading.y * this.simCfg.CAMERA_MOUNT_FORWARD_M;
      }
    }
    cameraPose() {
      const heading = direction(this.headingDeg);
      return {
        x: this.bodyX + heading.x * this.simCfg.CAMERA_MOUNT_FORWARD_M,
        y: this.bodyY + heading.y * this.simCfg.CAMERA_MOUNT_FORWARD_M,
        headingDeg: this.headingDeg
      };
    }
    palletPoint(lateral = 0, depth = 0) {
      const heading = direction(this.palletHeadingDeg);
      const right = { x: heading.y, y: -heading.x };
      return {
        x: this.palletFaceCenter.x + right.x * lateral + heading.x * depth,
        y: this.palletFaceCenter.y + right.y * lateral + heading.y * depth
      };
    }
    palletFrontGeometry() {
      const half = this.simCfg.PALLET_FRONT_WIDTH_M / 2;
      const heading = direction(this.palletHeadingDeg);
      return {
        start: this.palletPoint(-half, 0),
        end: this.palletPoint(half, 0),
        outwardNormal: { x: -heading.x, y: -heading.y }
      };
    }
    groundTruth() {
      const camera = this.cameraPose();
      const heading = direction(this.headingDeg);
      const right = { x: heading.y, y: -heading.x };
      const relative = {
        x: this.palletFaceCenter.x - camera.x,
        y: this.palletFaceCenter.y - camera.y
      };
      return {
        distZ: dot(relative, heading),
        offsetX: dot(relative, right),
        yaw: wrap180(this.headingDeg - this.palletHeadingDeg),
        relYaw: this.relYaw
      };
    }
    get distZ() { return this.groundTruth().distZ; }
    get offsetX() { return this.groundTruth().offsetX; }
    get yaw() { return this.groundTruth().yaw; }
    setPerception(detOk, detectedLength, visibility = null) {
      const nextDetOk = Boolean(detOk);
      const nextDetectedLength = nextDetOk ? Math.max(0, Number(detectedLength) || 0) : null;
      if (nextDetOk !== this.detOk || nextDetectedLength !== this.detectedLength) {
        this.observedSensors = null;
      }
      this.detOk = nextDetOk;
      this.detectedLength = nextDetectedLength;
      this.visibility = visibility;
    }
    sampleEstimationError(variable, groundTruth) {
      if (!this.estimationErrorEnabled || !this.estimationErrorModel) {
        return {
          error: 0,
          groundTruth,
          sourceCondition: groundTruth,
          distribution: "disabled",
          lowerCondition: groundTruth,
          upperCondition: groundTruth,
          weightUpper: 0
        };
      }
      return this.estimationErrorModel.sample(variable, groundTruth, this.estimationRandom);
    }
    sampleObservation() {
      const truth = this.groundTruth();
      const samples = {
        forward: this.sampleEstimationError("forward", truth.distZ),
        lateral: this.sampleEstimationError("lateral", truth.offsetX),
        yaw: this.sampleEstimationError("yaw", truth.yaw)
      };
      this.observedSensors = {
        detOk: this.detOk,
        detectedLength: this.detOk ? this.detectedLength : null,
        distZ: this.detOk ? truth.distZ + samples.forward.error : null,
        yaw: this.detOk ? truth.yaw + samples.yaw.error : null,
        offsetX: this.detOk ? truth.offsetX + samples.lateral.error : null,
        relYaw: this.relYaw,
        rotationDone: this.rotationDone
      };
      this.estimationSampleIndex += 1;
      this.lastEstimation = {
        sampleIndex: this.estimationSampleIndex,
        enabled: this.estimationErrorEnabled,
        truth: { ...truth },
        observed: { ...this.observedSensors },
        samples: Object.fromEntries(Object.entries(samples).map(([key, value]) => [key, { ...value }]))
      };
      return { ...this.observedSensors };
    }
    sensors() {
      if (!this.observedSensors) this.sampleObservation();
      return { ...this.observedSensors };
    }
    evaluateForkSafety(bodyX = this.bodyX, bodyY = this.bodyY) {
      return forkPalletCollisionAtPose({
        bodyX,
        bodyY,
        headingDeg: this.headingDeg,
        palletFaceCenter: this.palletFaceCenter,
        palletHeadingDeg: this.palletHeadingDeg,
        palletDepth: this.simCfg.PALLET_DEPTH_M,
        forkTineWidth: this.forkTineWidth,
        simConfig: this.simCfg
      });
    }
    sampleDriveError(kind, target) {
      if (!this.driveErrorEnabled || !this.driveErrorModel || Math.abs(target) < 1e-6) {
        return {
          error: 0,
          requestedTarget: target,
          sourceTarget: target,
          distribution: "disabled",
          lowerTarget: target,
          upperTarget: target,
          weightUpper: 0
        };
      }
      return this.driveErrorModel.sample(kind, target, this.driveRandom);
    }
    rotationTarget(command, fsm) {
      const chainStates = [
        "ALIGN_ROTATE_RIGHT", "ALIGN_ROTATE_LEFT",
        "ALIGN_ROTATE_LEFT_90", "ALIGN_ROTATE_RIGHT_90"
      ];
      const directTarget = Number(fsm.rotationCommandTargetDeg);
      const magnitude = chainStates.includes(fsm.alignSub)
        ? this.cfg.REL_YAW_TARGET_DEG
        : Number.isFinite(directTarget) && directTarget > 0
          ? directTarget
          : Math.max(1, Math.abs(this.groundTruth().yaw));
      return command === "ROT_LEFT" ? magnitude : -magnitude;
    }
    beginRotationError(command, fsm) {
      const target = this.rotationTarget(command, fsm);
      const sample = this.sampleDriveError("rotation", target);
      const endpointMode = Boolean(this.cfg.ROTATION_ENDPOINT_VALIDATION);
      const gain = endpointMode
        ? (target + sample.error) / target
        : clamp((target + sample.error) / target, 0.2, 1.8);
      this.activeRotationError = {
        ...sample,
        kind: "rotation",
        command,
        gain,
        endpointMode,
        endpointAngleDeg: endpointMode ? target + sample.error : null,
        travelledAngleDeg: 0,
        startedAt: fsm.now,
        startHeadingDeg: this.headingDeg,
        startRelYawDeg: this.relYaw,
        actualAngleDeg: 0
      };
      this.rotationDone = false;
      this.driveErrorEvents.push({ ...this.activeRotationError, event: "start" });
    }
    finishRotationError(fsm) {
      if (!this.activeRotationError) return;
      const completedAngle = wrap180(this.headingDeg - this.activeRotationError.startHeadingDeg);
      const sufficientlyComplete = Math.abs(completedAngle)
        >= 0.8 * Math.abs(this.activeRotationError.requestedTarget);
      if (!this.activeRotationError.endpointMode && this.driveErrorEnabled && sufficientlyComplete) {
        const endpointAngle = this.activeRotationError.requestedTarget + this.activeRotationError.error;
        this.headingDeg = wrap180(this.activeRotationError.startHeadingDeg + endpointAngle);
        this.relYaw = wrap180(this.activeRotationError.startRelYawDeg - endpointAngle);
      }
      this.lastRotationError = {
        ...this.activeRotationError,
        finishedAt: fsm.now,
        actualAngleDeg: wrap180(this.headingDeg - this.activeRotationError.startHeadingDeg)
      };
      this.driveErrorEvents.push({ ...this.lastRotationError, event: "finish" });
      this.activeRotationError = null;
    }
    buildMotion(command, fsm) {
      const truth = this.groundTruth();
      const sub = fsm.alignSub;
      let targetDistance = 0;
      let duration = null;
      let profile = "piecewise";
      let basis = "원본 FWD 가속→정속 피팅";
      let requestedDistance = 0;
      let stopMode = "sensor-band";
      if (sub === "INSERT_FORWARD") {
        requestedDistance = Math.max(0, truth.distZ + this.cfg.PALLET_POCKET_M);
        duration = Math.max(0, fsm.insertSecCached ?? insertionSeconds(truth.distZ, this.cfg) ?? 0);
        targetDistance = this.cfg.INSERT_FWD_MPS * duration;
        profile = "constant";
        basis = "원본 INSERT_FWD_MPS 정속";
        stopMode = "timer";
      } else if (["FORWARD_AFTER_RIGHT", "FORWARD_AFTER_LEFT"].includes(sub)) {
        duration = Math.max(0, fsm.fwdSecCached);
        targetDistance = distanceFromDurationPiecewise(duration, this.cfg);
        requestedDistance = targetDistance;
        stopMode = "timer";
      } else if (command === "FWD") {
        targetDistance = Math.max(0, truth.distZ - (this.cfg.ALIGN_DIST_M + this.cfg.ALIGN_BAND_M) + 1e-6);
        requestedDistance = targetDistance;
      } else if (command === "BACK") {
        targetDistance = Math.max(0, (this.cfg.ALIGN_DIST_M - this.cfg.ALIGN_BAND_M) - truth.distZ + 1e-6);
        requestedDistance = targetDistance;
        basis = "FWD 피팅 대칭 적용(BWD 전용 피팅 없음)";
      }
      const driveSample = this.sampleDriveError("forward", Math.max(0.001, targetDistance));
      const driveGain = clamp(
        (Math.max(0.001, targetDistance) + driveSample.error) / Math.max(0.001, targetDistance),
        0.2,
        1.8
      );
      return {
        command,
        state: fsm.statePath,
        profile,
        basis,
        stopMode,
        effort: 60,
        requestedDistance,
        targetDistance,
        plannedDuration: duration,
        elapsed: 0,
        distance: 0,
        actualDistance: 0,
        velocity: 0,
        actualVelocity: 0,
        driveError: driveSample,
        driveGain,
        startedAt: fsm.now,
        completed: false
      };
    }
    finishActiveMotion(fsm) {
      if (!this.activeMotion) return;
      this.activeMotion.completed = true;
      this.activeMotion.finishedAt = fsm.now;
      this.activeMotion.velocity = 0;
      this.completedMotions.push({ ...this.activeMotion });
      this.lastMotion = { ...this.activeMotion };
      this.driveErrorEvents.push({
        event: "finish",
        kind: "forward",
        command: this.activeMotion.command,
        requestedTarget: this.activeMotion.driveError.requestedTarget,
        error: this.activeMotion.driveError.error,
        distribution: this.activeMotion.driveError.distribution,
        gain: this.activeMotion.driveGain,
        actualDistance: this.activeMotion.actualDistance,
        finishedAt: fsm.now
      });
      this.activeMotion = null;
    }
    advance(command, dt, fsm) {
      const deltaTime = Math.max(0, Number(dt) || 0);
      if (command !== this.lastAppliedCommand) {
        this.finishActiveMotion(fsm);
        this.finishRotationError(fsm);
        if (command === "FWD" || command === "BACK") this.activeMotion = this.buildMotion(command, fsm);
        if (command === "ROT_LEFT" || command === "ROT_RIGHT") this.beginRotationError(command, fsm);
        else this.rotationDone = false;
        this.lastAppliedCommand = command;
      }

      if ((command === "ROT_LEFT" || command === "ROT_RIGHT")
          && this.activeRotationError?.endpointMode) {
        if (!this.rotationDone) {
          const motion = this.activeRotationError;
          const directionSign = command === "ROT_LEFT" ? 1 : -1;
          const remaining = motion.endpointAngleDeg - motion.travelledAngleDeg;
          const stepMagnitude = this.rotateRate * deltaTime;
          const sameDirection = Math.sign(remaining) === directionSign || Math.abs(remaining) < 1e-12;
          const delta = !sameDirection || Math.abs(remaining) <= stepMagnitude
            ? remaining
            : directionSign * stepMagnitude;
          this.headingDeg += delta;
          this.relYaw = wrap180(this.relYaw - delta);
          motion.travelledAngleDeg += delta;
          motion.actualAngleDeg = motion.travelledAngleDeg;
          if (Math.abs(motion.endpointAngleDeg - motion.travelledAngleDeg) < 1e-9) {
            this.rotationDone = true;
          }
        }
      } else if (command === "ROT_RIGHT") {
        const gain = this.activeRotationError?.gain ?? 1;
        const rotation = this.rotateRate * gain * deltaTime;
        this.headingDeg -= rotation;
        this.relYaw = wrap180(this.relYaw + rotation);
      } else if (command === "ROT_LEFT") {
        const gain = this.activeRotationError?.gain ?? 1;
        const rotation = this.rotateRate * gain * deltaTime;
        this.headingDeg += rotation;
        this.relYaw = wrap180(this.relYaw - rotation);
      } else if ((command === "FWD" || command === "BACK") && this.activeMotion) {
        const motion = this.activeMotion;
        const previousDistance = motion.distance;
        const timed = Number.isFinite(motion.plannedDuration);
        motion.elapsed = timed
          ? Math.min(motion.plannedDuration, motion.elapsed + deltaTime)
          : motion.elapsed + deltaTime;
        if (motion.profile === "constant") {
          motion.distance = Math.min(motion.targetDistance, this.cfg.INSERT_FWD_MPS * motion.elapsed);
          motion.velocity = motion.elapsed < motion.plannedDuration && motion.plannedDuration > 0
            ? this.cfg.INSERT_FWD_MPS : 0;
        } else {
          // The original fit defines distance from a fixed command duration.
          // Never rescale this curve to force a requested-distance endpoint.
          motion.distance = distanceFromDurationPiecewise(motion.elapsed, this.cfg);
          motion.velocity = (!timed || motion.elapsed < motion.plannedDuration)
            ? rawMotionVelocityAtTime(motion.elapsed, this.cfg) : 0;
        }
        const modelTravelled = Math.max(0, motion.distance - previousDistance);
        const sign = command === "FWD" ? 1 : -1;
        const heading = direction(this.headingDeg);
        let travelled = modelTravelled * motion.driveGain;
        motion.actualVelocity = motion.velocity * motion.driveGain;
        if (sign > 0 && fsm.alignSub === "INSERT_FORWARD" && travelled > 0) {
          const proposedX = this.bodyX + heading.x * travelled;
          const proposedY = this.bodyY + heading.y * travelled;
          const proposedSafety = this.evaluateForkSafety(proposedX, proposedY);
          if (proposedSafety.detected) {
            let safe = 0;
            let unsafe = travelled;
            for (let iteration = 0; iteration < 24; iteration += 1) {
              const middle = (safe + unsafe) / 2;
              const trial = this.evaluateForkSafety(
                this.bodyX + heading.x * middle,
                this.bodyY + heading.y * middle
              );
              if (trial.detected) unsafe = middle;
              else safe = middle;
            }
            travelled = safe;
            motion.distance = previousDistance + safe / Math.max(1e-9, motion.driveGain);
            motion.velocity = 0;
            motion.actualVelocity = 0;
            this.collision = {
              ...proposedSafety,
              detected: true,
              atTime: fsm.now,
              state: fsm.statePath,
              command
            };
          }
        }
        this.bodyX += heading.x * travelled * sign;
        this.bodyY += heading.y * travelled * sign;
        motion.actualDistance += travelled;
        if (!this.collision.detected) this.collision = this.evaluateForkSafety();
        this.lastMotion = { ...motion };
      } else if (command === "STOP") {
        this.finishActiveMotion(fsm);
      }

      this.headingDeg = wrap180(this.headingDeg);
      const truth = this.groundTruth();
      this.trace.push({ t: fsm.now, command, state: fsm.statePath, ...truth });
      this.observedSensors = null;
    }
    motionTelemetry() {
      return this.activeMotion ? { ...this.activeMotion }
        : this.lastMotion ? { ...this.lastMotion } : null;
    }
    driveErrorTelemetry() {
      return {
        enabled: this.driveErrorEnabled,
        activeRotation: this.activeRotationError ? { ...this.activeRotationError } : null,
        lastRotation: this.lastRotationError ? { ...this.lastRotationError } : null,
        events: this.driveErrorEvents.map((event) => ({ ...event }))
      };
    }
    estimationErrorTelemetry() {
      if (!this.lastEstimation) return {
        enabled: this.estimationErrorEnabled,
        sampleIndex: 0,
        truth: null,
        observed: null,
        samples: null
      };
      return {
        ...this.lastEstimation,
        truth: { ...this.lastEstimation.truth },
        observed: { ...this.lastEstimation.observed },
        samples: Object.fromEntries(Object.entries(this.lastEstimation.samples)
          .map(([key, value]) => [key, { ...value }]))
      };
    }
    collisionTelemetry() { return { ...this.collision, details: [...(this.collision.details || [])] }; }
  }

  return {
    CONFIG, SIM_CONFIG, TOP_STATES, ALIGN_STATES, CalibrationFSM, IdealPlant,
    Stabilizer, clamp, wrap180, withinBand, timeFromDistancePiecewise,
    fwdSecFromOffset, insertionSeconds, rawMotionDistanceAtTime,
    rawMotionVelocityAtTime,
    distanceFromDurationPiecewise, computePalletFrontVisibility,
    forkPalletCollisionAtPose
  };
}));
