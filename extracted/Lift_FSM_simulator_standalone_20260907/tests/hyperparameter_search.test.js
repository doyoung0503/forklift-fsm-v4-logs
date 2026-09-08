"use strict";

const assert = require("node:assert/strict");
const {
  parseGridSpec, combinationCount, cartesianProduct, rankEvaluations
} = require("../hyperparameter_search.js");

const entries = parseGridSpec(`
# FSM thresholds
YAW_TOL_DEG=1.5,2.0,2.5
OFF_TOL_M=0.08,0.12
CMD_STABLE_THR=3,5
`);
assert.equal(entries.length, 3);
assert.equal(combinationCount(entries), 12);
const combinations = cartesianProduct(entries);
assert.equal(combinations.length, 12);
assert.deepEqual(combinations[0], { YAW_TOL_DEG: 1.5, OFF_TOL_M: 0.08, CMD_STABLE_THR: 3 });
assert.deepEqual(combinations[11], { YAW_TOL_DEG: 2.5, OFF_TOL_M: 0.12, CMD_STABLE_THR: 5 });

assert.throws(() => parseGridSpec("UNKNOWN=1,2"), /지원하지 않는/);
assert.throws(() => parseGridSpec("STOP_SEC=0.8,1.2"), /지원하지 않는/);
assert.throws(() => parseGridSpec("REL_YAW_TARGET_DEG=88,90"), /지원하지 않는/);
assert.throws(() => parseGridSpec("CMD_STABLE_THR=2.5"), /정수/);
assert.throws(() => parseGridSpec("YAW_TOL_DEG=0"), /범위 오류/);
assert.throws(() => cartesianProduct(entries, 10), /제한 10개/);

const ranked = rankEvaluations([
  { index: 0, successRate: 0.90, collisionRate: 0.02, meanTime: 20 },
  { index: 1, successRate: 0.95, collisionRate: 0.10, meanTime: 30 },
  { index: 2, successRate: 0.95, collisionRate: 0.03, meanTime: 31 },
  { index: 3, successRate: 0.95, collisionRate: 0.03, meanTime: 29 }
]);
assert.deepEqual(ranked.map((row) => row.index), [3, 2, 1, 0]);
assert.deepEqual(ranked.map((row) => row.rank), [1, 2, 3, 4]);

console.log("PASS hyperparameter grid parsing, Cartesian expansion, validation, and ranking");
