"""Read the three insertion runs; export sampled states and exact command intervals."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "extracted/depth_cam/rec"
OUT = ROOT / "analysis/insertion_timing_20260906"
OUT.mkdir(parents=True, exist_ok=True)
TAGS = ["164100", "165305", "170410"]


def phase(state):
    if state in {"STARTUP", "PRECHECK"}: return "Startup"
    if state.startswith(("SEARCH", "ACQUIRE", "INITIAL", "FACE")): return "Acquire Pallet"
    if state.startswith("STANDOFF"): return "Set Standoff"
    if state == "STAGING_PLAN": return "Plan Approach"
    if state.startswith(("WAYPOINT", "RECENTER", "FINAL")): return "Execute Approach"
    if state.startswith(("READY", "INSERT", "RECOVER")): return "Insert Forks"
    return state


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f: return list(csv.DictReader(f))


runs = []
for tag in TAGS:
    base = f"forklift_v4_recording_20260906_{tag}"
    events = [json.loads(line) for line in (REC / (base + "_control_seq.jsonl")).read_text(encoding="utf-8").splitlines() if line]
    meta = json.loads((REC / (base + "_meta.json")).read_text(encoding="utf-8"))
    rows = read_csv(REC / (base + "_pallet_state.csv"))
    timing = read_csv(REC / (base + "_inference_timing.csv"))
    t0 = events[0]["t_mono"]
    failure = next(e for e in events if e["phase"] == "failure")
    end = failure["t_mono"]
    transitions = [(t0, "STARTUP")]
    for r in rows:
        t = float(r["t_mono"])
        if t >= end: break
        if r["fsm_state"] != transitions[-1][1]: transitions.append((t, r["fsm_state"]))
    intervals = [dict(start=t-t0, end=n-t0, duration=n-t, state=s, phase=phase(s))
                 for (t,s),(n,_) in zip(transitions, transitions[1:]+[(end,"END")])]
    commands = [e for e in events if e["phase"] == "cmd" and e["t_mono"] <= end]
    cmd_intervals = [dict(start=a["t_mono"]-t0, end=b["t_mono"]-t0,
                          duration=b["t_mono"]-a["t_mono"], command=a["cmd"])
                     for a,b in zip(commands,commands[1:]+[{"t_mono":end}])]
    totals = defaultdict(float)
    for r in intervals: totals[r["phase"]] += r["duration"]
    ends = {e["step"]:e for e in events if e["phase"] == "end"}
    actions = []
    for e in events:
        if e["phase"] != "begin" or e["step"] not in ends: continue
        finish = ends[e["step"]]
        stop = next((c for c in commands if c["cmd"] == "STOP" and e["t_mono"] < c["t_mono"] <= finish["t_mono"]), None)
        actions.append(dict(step=e["step"], kind=e["kind"], start=e["t_mono"]-t0,
                            end=finish["t_mono"]-t0, stop=None if stop is None else stop["t_mono"]-t0,
                            stop_to_end=None if stop is None else finish["t_mono"]-stop["t_mono"],
                            params=e.get("params"), result=finish.get("result")))
    settles = [a["stop_to_end"] for a in actions if a["stop_to_end"] is not None
               and a["kind"] not in {"v4_initial_visibility_sweep", "v4_insert_segment"}]
    inf = [float(r["model_inference_ms"]) for r in timing[1:] if r["model_inference_ms"] and float(r["pose_result_host_mono_ms"] or 0)/1000 <= end]
    cadence = [float(r["camera_input_interval_ms"]) for r in timing[1:] if r["camera_input_interval_ms"] and float(r["pose_result_host_mono_ms"] or 0)/1000 <= end]
    run = dict(tag=tag, start_iso=events[0]["t_iso"], duration=end-t0,
               insertion_start=next(a["start"] for a in actions if a["kind"] == "v4_insert_segment"),
               phase_totals=dict(totals), stop_command_seconds=sum(c["duration"] for c in cmd_intervals if c["command"]=="STOP"),
               intervals=intervals, commands=cmd_intervals, actions=actions,
               settle_count=len(settles), settle_total=sum(settles), guard_sec=meta["v4_stop_brake_guard_sec"],
               inference_median_ms=statistics.median(inf), inference_p95_ms=sorted(inf)[int(.95*(len(inf)-1))],
               cadence_median_ms=statistics.median(cadence), failure=failure["reason"])
    runs.append(run)
    for suffix,data in [("states",intervals),("commands",cmd_intervals)]:
        with (OUT/f"{tag}_{suffix}.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
(OUT/"summary.json").write_text(json.dumps(runs,indent=2,ensure_ascii=False),encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
colors = dict(zip(["Startup","Acquire Pallet","Set Standoff","Plan Approach","Execute Approach","Insert Forks"],
                  ["#a8adb8","#65b9a1","#73a7d5","#e5b35b","#8975c7","#e98172"]))
fig, axes = plt.subplots(3,1,figsize=(16,7),sharex=True)
for ax,r in zip(axes,runs):
    for s in r["intervals"]:
        ax.broken_barh([(s["start"],s["duration"])],(.52,.42),facecolors=colors[s["phase"]])
    for c in r["commands"]:
        ax.broken_barh([(c["start"],c["duration"])],(.12,.23),facecolors="#374151" if c["command"]=="STOP" else "#d5dbe3")
    ax.axvline(r["insertion_start"],color="#c74b43",ls="--",lw=1)
    ax.text(r["insertion_start"]+.8,1.02,f'Insert {r["insertion_start"]:.1f}s',fontsize=9)
    ax.text(r["duration"]+.8,.56,f'FAIL {r["duration"]:.1f}s',fontsize=9)
    ax.set_title(f'{r["start_iso"][11:19]}  |  STOP command {r["stop_command_seconds"]:.1f}s / {r["duration"]:.1f}s',loc="left",fontsize=11)
    ax.set_yticks([.23,.73],["Command","Phase"]);ax.set_ylim(0,1.2);ax.grid(axis="x",alpha=.2)
    for edge in ["top","right","left"]:ax.spines[edge].set_visible(False)
axes[-1].set_xlabel("Seconds since first logged command (separate origin for each run)")
axes[-1].set_xlim(0,166)
fig.legend(handles=[Patch(color=c,label=p) for p,c in colors.items()]+[Patch(color="#374151",label="STOP command")],loc="lower center",ncol=7,fontsize=9)
fig.suptitle("Insertion-run timelines | 2026-09-06",fontsize=16)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/"timeline.png",dpi=160);plt.close(fig)
for r in runs:
    print(r["tag"],{k:round(v,2) for k,v in r["phase_totals"].items()},"settle",r["settle_count"],round(r["settle_total"],3),"guard",r["settle_count"]*r["guard_sec"])
print(OUT)
