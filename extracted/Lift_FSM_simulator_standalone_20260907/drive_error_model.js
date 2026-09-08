(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DriveError = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SQRT_TWO_PI = Math.sqrt(2 * Math.PI);
  const MIN_SCALE = 1e-6;
  const CANDIDATE_NAMES = Object.freeze(["gaussian", "student-t", "gaussian-kde"]);

  function mean(values) {
    return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
  }

  function variance(values, center = mean(values)) {
    return values.reduce((sum, value) => sum + (value - center) ** 2, 0) / Math.max(1, values.length);
  }

  function standardDeviation(values) {
    return Math.sqrt(Math.max(MIN_SCALE ** 2, variance(values)));
  }

  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function quantile(values, probability) {
    const sorted = [...values].sort((a, b) => a - b);
    const index = (sorted.length - 1) * probability;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  function logGamma(value) {
    const coefficients = [
      676.5203681218851, -1259.1392167224028, 771.3234287776531,
      -176.6150291621406, 12.507343278686905, -0.13857109526572012,
      9.984369578019571e-6, 1.5056327351493116e-7
    ];
    if (value < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
    let x = 0.9999999999998099;
    const z = value - 1;
    coefficients.forEach((coefficient, index) => { x += coefficient / (z + index + 1); });
    const t = z + coefficients.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  function gaussianLogPdf(value, location, scale) {
    const sigma = Math.max(MIN_SCALE, scale);
    const z = (value - location) / sigma;
    return -Math.log(sigma * SQRT_TWO_PI) - 0.5 * z * z;
  }

  function studentTLogPdf(value, location, scale, degreesOfFreedom) {
    const sigma = Math.max(MIN_SCALE, scale);
    const nu = Math.max(2.01, degreesOfFreedom);
    const z = (value - location) / sigma;
    return logGamma((nu + 1) / 2) - logGamma(nu / 2)
      - 0.5 * Math.log(nu * Math.PI) - Math.log(sigma)
      - ((nu + 1) / 2) * Math.log1p((z * z) / nu);
  }

  function fitGaussian(values) {
    const location = mean(values);
    const scale = standardDeviation(values);
    const logLikelihood = values.reduce((sum, value) => sum + gaussianLogPdf(value, location, scale), 0);
    return {
      name: "gaussian",
      params: { location, scale },
      logLikelihood,
      effectiveParameters: 2,
      bic: -2 * logLikelihood + 2 * Math.log(values.length)
    };
  }

  function fitStudentForNu(values, degreesOfFreedom) {
    let location = median(values);
    let scale = Math.max(MIN_SCALE, (quantile(values, 0.75) - quantile(values, 0.25)) / 1.349);
    if (scale <= MIN_SCALE * 2) scale = standardDeviation(values);
    for (let iteration = 0; iteration < 80; iteration += 1) {
      const weights = values.map((value) => {
        const z = (value - location) / Math.max(MIN_SCALE, scale);
        return (degreesOfFreedom + 1) / (degreesOfFreedom + z * z);
      });
      const weightSum = weights.reduce((sum, weight) => sum + weight, 0);
      const nextLocation = values.reduce((sum, value, index) => sum + weights[index] * value, 0) / weightSum;
      const nextScale = Math.sqrt(Math.max(
        MIN_SCALE ** 2,
        values.reduce((sum, value, index) => sum + weights[index] * (value - nextLocation) ** 2, 0) / values.length
      ));
      if (Math.abs(nextLocation - location) + Math.abs(nextScale - scale) < 1e-10) {
        location = nextLocation;
        scale = nextScale;
        break;
      }
      location = nextLocation;
      scale = nextScale;
    }
    const logLikelihood = values.reduce(
      (sum, value) => sum + studentTLogPdf(value, location, scale, degreesOfFreedom), 0
    );
    return { location, scale, degreesOfFreedom, logLikelihood };
  }

  function fitStudentT(values) {
    const degreesCandidates = [2.1, 2.5, 3, 4, 5, 7, 10, 15, 25, 40, 80, 200];
    const best = degreesCandidates
      .map((degreesOfFreedom) => fitStudentForNu(values, degreesOfFreedom))
      .sort((a, b) => b.logLikelihood - a.logLikelihood)[0];
    return {
      name: "student-t",
      params: {
        location: best.location,
        scale: best.scale,
        degreesOfFreedom: best.degreesOfFreedom
      },
      logLikelihood: best.logLikelihood,
      effectiveParameters: 3,
      bic: -2 * best.logLikelihood + 3 * Math.log(values.length)
    };
  }

  function gaussianKernel(delta, bandwidth) {
    const z = delta / Math.max(MIN_SCALE, bandwidth);
    return Math.exp(-0.5 * z * z) / (Math.max(MIN_SCALE, bandwidth) * SQRT_TWO_PI);
  }

  function kdeLeaveOneOutLogLikelihood(values, bandwidth) {
    return values.reduce((total, value, index) => {
      let density = 0;
      for (let other = 0; other < values.length; other += 1) {
        if (other !== index) density += gaussianKernel(value - values[other], bandwidth);
      }
      density /= Math.max(1, values.length - 1);
      return total + Math.log(Math.max(1e-300, density));
    }, 0);
  }

  function kdeEffectiveParameters(values, bandwidth) {
    let trace = 0;
    values.forEach((value) => {
      let denominator = 0;
      values.forEach((other) => { denominator += gaussianKernel(value - other, bandwidth); });
      trace += gaussianKernel(0, bandwidth) / Math.max(1e-300, denominator);
    });
    return Math.max(1, Math.min(values.length - 1, trace + 1));
  }

  function fitKde(values) {
    const sigma = standardDeviation(values);
    const iqrScale = (quantile(values, 0.75) - quantile(values, 0.25)) / 1.349;
    const robustScale = Math.max(MIN_SCALE, Math.min(sigma, iqrScale > MIN_SCALE ? iqrScale : sigma));
    const silverman = Math.max(MIN_SCALE, 0.9 * robustScale * values.length ** (-0.2));
    const multipliers = [0.30, 0.40, 0.55, 0.70, 0.85, 1.0, 1.25, 1.55, 1.9, 2.4];
    const candidates = multipliers.map((multiplier) => {
      const bandwidth = silverman * multiplier;
      const logLikelihood = kdeLeaveOneOutLogLikelihood(values, bandwidth);
      const effectiveParameters = kdeEffectiveParameters(values, bandwidth);
      return {
        bandwidth,
        logLikelihood,
        effectiveParameters,
        bic: -2 * logLikelihood + effectiveParameters * Math.log(values.length)
      };
    });
    const best = candidates.sort((a, b) => a.bic - b.bic)[0];
    return {
      name: "gaussian-kde",
      params: { bandwidth: best.bandwidth, samples: [...values] },
      logLikelihood: best.logLikelihood,
      effectiveParameters: best.effectiveParameters,
      bic: best.bic
    };
  }

  function fitCandidates(values) {
    if (!Array.isArray(values) || values.length < 5) throw new Error("각 조건에는 최소 5개의 오차값이 필요합니다.");
    const candidates = [fitGaussian(values), fitStudentT(values), fitKde(values)]
      .sort((a, b) => a.bic - b.bic);
    return { selected: candidates[0], candidates };
  }

  function parseCsv(text) {
    const lines = String(text).trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) throw new Error("CSV에 헤더와 데이터가 필요합니다.");
    const headers = lines[0].split(",").map((header) => header.trim());
    const required = ["trial", "kind", "target", "actual"];
    required.forEach((name) => {
      if (!headers.includes(name)) throw new Error(`CSV 필수 열 누락: ${name}`);
    });
    return lines.slice(1).map((line, rowIndex) => {
      const columns = line.split(",").map((column) => column.trim());
      const raw = Object.fromEntries(headers.map((header, index) => [header, columns[index]]));
      const kind = raw.kind;
      const target = Number(raw.target);
      const actual = Number(raw.actual);
      if (!["forward", "rotation"].includes(kind) || !Number.isFinite(target) || !Number.isFinite(actual)) {
        throw new Error(`CSV ${rowIndex + 2}행 값이 올바르지 않습니다.`);
      }
      return {
        trial: Number(raw.trial),
        kind,
        target,
        actual,
        error: Number.isFinite(Number(raw.error)) ? Number(raw.error) : actual - target,
        unit: raw.unit || (kind === "forward" ? "m" : "deg")
      };
    });
  }

  function rowsToCsv(rows) {
    const header = "trial,kind,target,actual,error,unit";
    const body = rows.map((row) => [
      row.trial, row.kind, Number(row.target).toFixed(row.kind === "forward" ? 3 : 1),
      Number(row.actual).toFixed(row.kind === "forward" ? 6 : 4),
      Number(row.error ?? row.actual - row.target).toFixed(row.kind === "forward" ? 6 : 4),
      row.unit || (row.kind === "forward" ? "m" : "deg")
    ].join(","));
    return [header, ...body].join("\n");
  }

  function randomNormal(random) {
    const u1 = Math.max(1e-12, random());
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function randomGamma(shape, random) {
    if (shape < 1) return randomGamma(shape + 1, random) * random() ** (1 / shape);
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      const normal = randomNormal(random);
      const vBase = 1 + c * normal;
      if (vBase <= 0) continue;
      const v = vBase ** 3;
      const uniform = random();
      if (uniform < 1 - 0.0331 * normal ** 4) return d * v;
      if (Math.log(uniform) < 0.5 * normal ** 2 + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  function sampleDistribution(model, random) {
    const { params } = model;
    if (model.name === "gaussian") return params.location + params.scale * randomNormal(random);
    if (model.name === "student-t") {
      const chiSquare = 2 * randomGamma(params.degreesOfFreedom / 2, random);
      return params.location + params.scale * randomNormal(random) / Math.sqrt(chiSquare / params.degreesOfFreedom);
    }
    const center = params.samples[Math.min(params.samples.length - 1, Math.floor(random() * params.samples.length))];
    return center + params.bandwidth * randomNormal(random);
  }

  function distributionDensity(model, value) {
    if (!model || !Number.isFinite(Number(value))) return 0;
    const numericValue = Number(value);
    const { params } = model;
    if (model.name === "gaussian") {
      return Math.exp(gaussianLogPdf(numericValue, params.location, params.scale));
    }
    if (model.name === "student-t") {
      return Math.exp(studentTLogPdf(
        numericValue,
        params.location,
        params.scale,
        params.degreesOfFreedom
      ));
    }
    if (model.name === "gaussian-kde") {
      return params.samples.reduce(
        (sum, sample) => sum + gaussianKernel(numericValue - sample, params.bandwidth),
        0
      ) / Math.max(1, params.samples.length);
    }
    return 0;
  }

  function groupKey(kind, target) {
    return `${kind}:${Number(target)}`;
  }

  class FittedDriveErrorModel {
    constructor(rows) {
      this.rows = rows.map((row) => ({ ...row, error: Number(row.error ?? row.actual - row.target) }));
      const grouped = new Map();
      this.rows.forEach((row) => {
        const key = groupKey(row.kind, row.target);
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(row.error);
      });
      this.groups = [...grouped.entries()].map(([key, errors]) => {
        const [kind, targetText] = key.split(":");
        const fit = fitCandidates(errors);
        return { key, kind, target: Number(targetText), count: errors.length, errors, ...fit };
      }).sort((a, b) => a.kind.localeCompare(b.kind) || a.target - b.target);
      this.kindSelections = {};
      [...new Set(this.groups.map((group) => group.kind))].forEach((kind) => {
        const kindGroups = this.groups.filter((group) => group.kind === kind);
        const averageBics = Object.fromEntries(CANDIDATE_NAMES.map((name) => [
          name,
          mean(kindGroups.map((group) => group.candidates.find((candidate) => candidate.name === name).bic))
        ]));
        const selectedName = CANDIDATE_NAMES.reduce((best, name) =>
          averageBics[name] < averageBics[best] ? name : best, CANDIDATE_NAMES[0]);
        this.kindSelections[kind] = {
          kind,
          selected: selectedName,
          averageBic: averageBics[selectedName],
          candidates: averageBics,
          conditionCount: kindGroups.length
        };
        kindGroups.forEach((group) => {
          group.conditionBest = group.selected;
          group.selected = group.candidates.find((candidate) => candidate.name === selectedName);
        });
      });
    }

    group(kind, target) {
      return this.groups.find((group) => group.kind === kind && group.target === Number(target));
    }

    bracket(kind, target) {
      const groups = this.groups.filter((group) => group.kind === kind).sort((a, b) => a.target - b.target);
      if (groups.length === 0) throw new Error(`구동오차 조건 없음: ${kind}`);
      if (target <= groups[0].target) return { lower: groups[0], upper: groups[0], weightUpper: 0 };
      if (target >= groups[groups.length - 1].target) {
        return { lower: groups[groups.length - 1], upper: groups[groups.length - 1], weightUpper: 0 };
      }
      for (let index = 0; index < groups.length - 1; index += 1) {
        const lower = groups[index];
        const upper = groups[index + 1];
        if (target >= lower.target && target <= upper.target) {
          return {
            lower,
            upper,
            weightUpper: (target - lower.target) / Math.max(MIN_SCALE, upper.target - lower.target)
          };
        }
      }
      return { lower: groups[0], upper: groups[0], weightUpper: 0 };
    }

    sample(kind, target, random = Math.random) {
      const bracket = this.bracket(kind, Number(target));
      const chosen = random() < bracket.weightUpper ? bracket.upper : bracket.lower;
      return {
        error: sampleDistribution(chosen.selected, random),
        requestedTarget: Number(target),
        lowerTarget: bracket.lower.target,
        upperTarget: bracket.upper.target,
        weightUpper: bracket.weightUpper,
        sourceTarget: chosen.target,
        distribution: chosen.selected.name,
        bic: chosen.selected.bic
      };
    }

    summary() {
      return this.groups.map((group) => ({
        kind: group.kind,
        target: group.target,
        count: group.count,
        selected: group.selected.name,
        bic: group.selected.bic,
        averageBic: this.kindSelections[group.kind].averageBic,
        candidates: Object.fromEntries(group.candidates.map((candidate) => [candidate.name, candidate.bic])),
        meanError: mean(group.errors),
        stdError: standardDeviation(group.errors)
      }));
    }

    selectionSummary() {
      return Object.values(this.kindSelections).map((selection) => ({
        ...selection,
        candidates: { ...selection.candidates }
      }));
    }
  }

  function createSeededRandom(seed = 20260713) {
    let state = (Number(seed) >>> 0) || 1;
    return function random() {
      state = (state + 0x6D2B79F5) >>> 0;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function generateExampleRows() {
    let state = 0x5a17c9e3;
    let spareNormal = null;
    const random = () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    };
    const normal = () => {
      if (spareNormal !== null) {
        const value = spareNormal;
        spareNormal = null;
        return value;
      }
      const magnitude = Math.sqrt(-2 * Math.log(Math.max(1e-9, random())));
      const angle = 2 * Math.PI * random();
      spareNormal = magnitude * Math.sin(angle);
      return magnitude * Math.cos(angle);
    };
    const specifications = [
      { kind: "forward", target: 1.8, unit: "m", type: "g", mean: 0.010, sd: 0.016 },
      { kind: "forward", target: 2.0, unit: "m", type: "t", mean: 0.004, sd: 0.014 },
      { kind: "forward", target: 2.2, unit: "m", type: "b", mean1: -0.034, mean2: 0.042, sd: 0.006 },
      { kind: "rotation", target: -90, unit: "deg", type: "g", mean: -1.1, sd: 0.9 },
      { kind: "rotation", target: -60, unit: "deg", type: "t", mean: -0.5, sd: 0.7 },
      { kind: "rotation", target: -30, unit: "deg", type: "b", mean1: -1.7, mean2: 1.1, sd: 0.35 },
      { kind: "rotation", target: 30, unit: "deg", type: "g", mean: 0.35, sd: 0.55 },
      { kind: "rotation", target: 60, unit: "deg", type: "b", mean1: -1.8, mean2: 1.6, sd: 0.4 },
      { kind: "rotation", target: 90, unit: "deg", type: "t", mean: 0.8, sd: 0.85 }
    ];
    const rows = [];
    specifications.forEach((specification) => {
      for (let trial = 1; trial <= 30; trial += 1) {
        let error;
        if (specification.type === "g") {
          error = specification.mean + specification.sd * normal();
        } else if (specification.type === "b") {
          error = (trial <= 15 ? specification.mean1 : specification.mean2) + specification.sd * normal();
        } else {
          error = specification.mean + specification.sd * normal();
          if (trial === 4) error += specification.kind === "forward" ? 0.075 : 4.2;
          if (trial === 18) error -= specification.kind === "forward" ? 0.070 : 3.8;
          if (trial === 27) error += specification.kind === "forward" ? 0.052 : 3.0;
        }
        rows.push({
          trial,
          kind: specification.kind,
          target: specification.target,
          actual: specification.target + error,
          error,
          unit: specification.unit
        });
      }
    });
    return rows;
  }

  return {
    FittedDriveErrorModel,
    createSeededRandom,
    fitCandidates,
    generateExampleRows,
    parseCsv,
    rowsToCsv,
    sampleDistribution,
    distributionDensity
  };
}));
