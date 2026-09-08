(function (root, factory) {
  const driveError = typeof module === "object" && module.exports
    ? require("./drive_error_model.js")
    : root.DriveError;
  const api = factory(driveError);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.EstimationError = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function (DriveError) {
  "use strict";

  const VARIABLES = Object.freeze(["forward", "lateral", "yaw"]);
  const CANDIDATE_NAMES = Object.freeze(["gaussian", "student-t", "gaussian-kde"]);

  function mean(values) {
    return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
  }

  function standardDeviation(values) {
    const center = mean(values);
    return Math.sqrt(values.reduce((sum, value) => sum + (value - center) ** 2, 0) / Math.max(1, values.length));
  }

  function groupKey(variable, condition) {
    return `${variable}:${Number(condition)}`;
  }

  function parseCsv(text) {
    const lines = String(text).trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) throw new Error("CSV에 헤더와 데이터가 필요합니다.");
    const headers = lines[0].split(",").map((header) => header.trim());
    const required = ["trial", "variable", "condition", "ground_truth", "estimate"];
    required.forEach((name) => {
      if (!headers.includes(name)) throw new Error(`CSV 필수 열 누락: ${name}`);
    });
    return lines.slice(1).map((line, rowIndex) => {
      const columns = line.split(",").map((column) => column.trim());
      const raw = Object.fromEntries(headers.map((header, index) => [header, columns[index]]));
      const variable = raw.variable;
      const condition = Number(raw.condition);
      const groundTruth = Number(raw.ground_truth);
      const estimate = Number(raw.estimate);
      if (!VARIABLES.includes(variable) || !Number.isFinite(condition)
          || !Number.isFinite(groundTruth) || !Number.isFinite(estimate)) {
        throw new Error(`CSV ${rowIndex + 2}행 값이 올바르지 않습니다.`);
      }
      return {
        trial: Number(raw.trial),
        variable,
        condition,
        groundTruth,
        estimate,
        error: Number.isFinite(Number(raw.error)) ? Number(raw.error) : estimate - groundTruth,
        unit: raw.unit || (variable === "yaw" ? "deg" : "m")
      };
    });
  }

  function rowsToCsv(rows) {
    const header = "trial,variable,condition,ground_truth,estimate,error,unit";
    const body = rows.map((row) => {
      const digits = row.variable === "yaw" ? 5 : 6;
      return [
        row.trial,
        row.variable,
        Number(row.condition).toFixed(row.variable === "yaw" ? 1 : 3),
        Number(row.groundTruth).toFixed(row.variable === "yaw" ? 3 : 4),
        Number(row.estimate).toFixed(digits),
        Number(row.error ?? row.estimate - row.groundTruth).toFixed(digits),
        row.unit || (row.variable === "yaw" ? "deg" : "m")
      ].join(",");
    });
    return [header, ...body].join("\n");
  }

  class FittedEstimationErrorModel {
    constructor(rows) {
      this.rows = rows.map((row) => ({
        ...row,
        condition: Number(row.condition),
        error: Number(row.error ?? row.estimate - row.groundTruth)
      }));
      const grouped = new Map();
      this.rows.forEach((row) => {
        const key = groupKey(row.variable, row.condition);
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(row.error);
      });
      this.groups = [...grouped.entries()].map(([key, errors]) => {
        const splitAt = key.indexOf(":");
        const variable = key.slice(0, splitAt);
        const condition = Number(key.slice(splitAt + 1));
        const fit = DriveError.fitCandidates(errors);
        return { key, variable, condition, count: errors.length, errors, ...fit };
      }).sort((a, b) => a.variable.localeCompare(b.variable) || a.condition - b.condition);

      this.variableSelections = {};
      VARIABLES.forEach((variable) => {
        const variableGroups = this.groups.filter((group) => group.variable === variable);
        if (variableGroups.length === 0) return;
        const averageBics = Object.fromEntries(CANDIDATE_NAMES.map((name) => [
          name,
          mean(variableGroups.map((group) => group.candidates.find((candidate) => candidate.name === name).bic))
        ]));
        const selectedName = CANDIDATE_NAMES.reduce((best, name) =>
          averageBics[name] < averageBics[best] ? name : best, CANDIDATE_NAMES[0]);
        this.variableSelections[variable] = {
          variable,
          selected: selectedName,
          averageBic: averageBics[selectedName],
          candidates: averageBics,
          conditionCount: variableGroups.length
        };
        variableGroups.forEach((group) => {
          group.conditionBest = group.selected;
          group.selected = group.candidates.find((candidate) => candidate.name === selectedName);
        });
      });
    }

    bracket(variable, value) {
      const groups = this.groups.filter((group) => group.variable === variable)
        .sort((a, b) => a.condition - b.condition);
      if (groups.length === 0) throw new Error(`모델 추정오차 조건 없음: ${variable}`);
      if (value <= groups[0].condition) return { lower: groups[0], upper: groups[0], weightUpper: 0 };
      if (value >= groups[groups.length - 1].condition) {
        return { lower: groups[groups.length - 1], upper: groups[groups.length - 1], weightUpper: 0 };
      }
      for (let index = 0; index < groups.length - 1; index += 1) {
        const lower = groups[index];
        const upper = groups[index + 1];
        if (value >= lower.condition && value <= upper.condition) {
          return {
            lower,
            upper,
            weightUpper: (value - lower.condition) / (upper.condition - lower.condition)
          };
        }
      }
      return { lower: groups[0], upper: groups[0], weightUpper: 0 };
    }

    sample(variable, value, random = Math.random) {
      const bracket = this.bracket(variable, Number(value));
      const chosen = random() < bracket.weightUpper ? bracket.upper : bracket.lower;
      return {
        error: DriveError.sampleDistribution(chosen.selected, random),
        groundTruth: Number(value),
        lowerCondition: bracket.lower.condition,
        upperCondition: bracket.upper.condition,
        weightUpper: bracket.weightUpper,
        sourceCondition: chosen.condition,
        distribution: chosen.selected.name,
        bic: chosen.selected.bic
      };
    }

    summary() {
      return this.groups.map((group) => ({
        variable: group.variable,
        condition: group.condition,
        count: group.count,
        selected: group.selected.name,
        bic: group.selected.bic,
        averageBic: this.variableSelections[group.variable].averageBic,
        candidates: Object.fromEntries(group.candidates.map((candidate) => [candidate.name, candidate.bic])),
        meanError: mean(group.errors),
        stdError: standardDeviation(group.errors)
      }));
    }

    selectionSummary() {
      return Object.values(this.variableSelections).map((selection) => ({
        ...selection,
        candidates: { ...selection.candidates }
      }));
    }
  }

  function generateExampleRows() {
    const random = DriveError.createSeededRandom(0x7a14e57);
    let spare = null;
    const normal = () => {
      if (spare !== null) {
        const value = spare;
        spare = null;
        return value;
      }
      const magnitude = Math.sqrt(-2 * Math.log(Math.max(1e-12, random())));
      const angle = 2 * Math.PI * random();
      spare = magnitude * Math.sin(angle);
      return magnitude * Math.cos(angle);
    };
    const specs = [
      { variable: "forward", condition: 1.8, unit: "m", type: "g", mean: 0.006, sd: 0.012 },
      { variable: "forward", condition: 2.0, unit: "m", type: "t", mean: -0.002, sd: 0.010 },
      { variable: "forward", condition: 2.2, unit: "m", type: "b", mean1: -0.018, mean2: 0.020, sd: 0.004 },
      { variable: "lateral", condition: -0.30, unit: "m", type: "g", mean: 0.006, sd: 0.012 },
      { variable: "lateral", condition: 0.00, unit: "m", type: "t", mean: 0.000, sd: 0.009 },
      { variable: "lateral", condition: 0.30, unit: "m", type: "b", mean1: -0.020, mean2: 0.018, sd: 0.004 },
      { variable: "yaw", condition: -15, unit: "deg", type: "g", mean: 0.20, sd: 0.65 },
      { variable: "yaw", condition: 0, unit: "deg", type: "t", mean: 0.00, sd: 0.45 },
      { variable: "yaw", condition: 15, unit: "deg", type: "b", mean1: -0.80, mean2: 0.70, sd: 0.20 }
    ];
    const rows = [];
    specs.forEach((spec) => {
      for (let trial = 1; trial <= 30; trial += 1) {
        let error;
        if (spec.type === "g") {
          error = spec.mean + spec.sd * normal();
        } else if (spec.type === "b") {
          error = (trial <= 15 ? spec.mean1 : spec.mean2) + spec.sd * normal();
        } else {
          error = spec.mean + spec.sd * normal();
          if (trial === 4) error += spec.variable === "yaw" ? 2.7 : 0.045;
          if (trial === 18) error -= spec.variable === "yaw" ? 2.4 : 0.041;
          if (trial === 27) error += spec.variable === "yaw" ? 1.8 : 0.032;
        }
        rows.push({
          trial,
          variable: spec.variable,
          condition: spec.condition,
          groundTruth: spec.condition,
          estimate: spec.condition + error,
          error,
          unit: spec.unit
        });
      }
    });
    return rows;
  }

  return {
    FittedEstimationErrorModel,
    VARIABLES,
    generateExampleRows,
    parseCsv,
    rowsToCsv
  };
}));
