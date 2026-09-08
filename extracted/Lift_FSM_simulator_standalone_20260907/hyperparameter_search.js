(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.HyperparameterSearch = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const PARAMETER_DEFINITIONS = Object.freeze({
    YAW_TOL_DEG: Object.freeze({ label: "yaw 허용오차", unit: "°", min: 0.05, max: 45 }),
    OFF_TOL_M: Object.freeze({ label: "lateral 허용오차", unit: "m", min: 0.001, max: 1.0 }),
    WIDTH_MIN_FULL: Object.freeze({ label: "전면 검출 폭 threshold", unit: "m", min: 0, max: 1.10 }),
    ALIGN_DIST_M: Object.freeze({ label: "정렬 목표 거리", unit: "m", min: 0.05, max: 5.0 }),
    ALIGN_BAND_M: Object.freeze({ label: "정렬 거리 band", unit: "m", min: 0.001, max: 2.0 }),
    CMD_STABLE_THR: Object.freeze({
      label: "연속 판정 frame 수", unit: "frames", min: 1, max: 120, integer: true
    })
  });

  function parseGridSpec(text) {
    const entries = [];
    const seen = new Set();
    String(text).split(/\r?\n/).forEach((rawLine, lineIndex) => {
      const line = rawLine.replace(/#.*/, "").trim();
      if (!line) return;
      const separator = line.indexOf("=");
      if (separator < 1) throw new Error(`${lineIndex + 1}행은 PARAM=value1,value2 형식이어야 합니다.`);
      const name = line.slice(0, separator).trim().toUpperCase();
      const definition = PARAMETER_DEFINITIONS[name];
      if (!definition) throw new Error(`${lineIndex + 1}행의 지원하지 않는 파라미터: ${name}`);
      if (seen.has(name)) throw new Error(`${name} 파라미터가 중복되었습니다.`);
      const tokens = line.slice(separator + 1).split(",").map((token) => token.trim()).filter(Boolean);
      if (tokens.length === 0) throw new Error(`${name}에 후보값이 없습니다.`);
      const values = [];
      tokens.forEach((token) => {
        const value = Number(token);
        if (!Number.isFinite(value)) throw new Error(`${name}의 후보값 '${token}'은 숫자가 아닙니다.`);
        if (definition.integer && !Number.isInteger(value)) throw new Error(`${name}은 정수여야 합니다: ${token}`);
        if (value < definition.min || value > definition.max) {
          throw new Error(`${name}=${value} 범위 오류 (${definition.min}~${definition.max})`);
        }
        if (!values.includes(value)) values.push(value);
      });
      entries.push({ name, values });
      seen.add(name);
    });
    if (entries.length === 0) throw new Error("검색할 파라미터와 후보값을 한 줄 이상 입력하세요.");
    return entries;
  }

  function combinationCount(entries) {
    return entries.reduce((count, entry) => count * entry.values.length, 1);
  }

  function cartesianProduct(entries, maximum = 500) {
    const count = combinationCount(entries);
    if (count > maximum) throw new Error(`후보 조합 ${count}개가 제한 ${maximum}개를 초과합니다.`);
    let combinations = [{}];
    entries.forEach((entry) => {
      combinations = combinations.flatMap((combination) => entry.values.map((value) => ({
        ...combination,
        [entry.name]: value
      })));
    });
    return combinations;
  }

  function compareEvaluations(left, right) {
    if (left.successRate !== right.successRate) return right.successRate - left.successRate;
    if (left.collisionRate !== right.collisionRate) return left.collisionRate - right.collisionRate;
    if (left.meanTime !== right.meanTime) return left.meanTime - right.meanTime;
    return (left.index ?? 0) - (right.index ?? 0);
  }

  function rankEvaluations(evaluations) {
    return evaluations.map((evaluation) => ({ ...evaluation })).sort(compareEvaluations)
      .map((evaluation, index) => ({ ...evaluation, rank: index + 1 }));
  }

  function formatParameters(parameters) {
    return Object.entries(parameters).map(([name, value]) => `${name}=${value}`).join(", ");
  }

  return {
    PARAMETER_DEFINITIONS,
    parseGridSpec,
    combinationCount,
    cartesianProduct,
    compareEvaluations,
    rankEvaluations,
    formatParameters
  };
}));
