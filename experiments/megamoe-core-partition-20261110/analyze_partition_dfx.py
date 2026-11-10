#!/usr/bin/env python3
"""Audit real per-rank DFX traces and draw physical-core scheduling comparisons."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


WAIT_DESC = 0x10000
TRIGGER_DESC = 0x10007
TERMINATE_DESC = 0x10006
COLORS = {"wait": "#D8DEE4", "trigger": "#8C99A6", "Dispatch": "#4598B4",
          "Combine": "#267C76", "up_even": "#D57927", "up_odd": "#3D6DB0",
          "GMM2": "#7060A1", "SwiGLU": "#73A64B", "shared": "#B472A0",
          "epilogue": "#D9B548", "control": "#9A9A9A", "other": "#5C5C5C"}


def merge_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if end < begin:
            raise ValueError("negative trace interval")
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return merged


def union_length(intervals):
    return sum(end - begin for begin, end in merge_intervals(intervals))


def intersection_length(first, second):
    first, second = merge_intervals(first), merge_intervals(second)
    total, i, j = 0.0, 0, 0
    while i < len(first) and j < len(second):
        total += max(0.0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return total


def overlap(first, second):
    value = intersection_length(first, second)
    shorter = min(union_length(first), union_length(second))
    return {"overlap_us": value, "fraction_of_shorter_active_union": value / shorter if shorter else None}


def stage(event):
    return event["args"].get("task_stage", event["name"])


def phase(event):
    desc = event["args"]["desc_id"]
    return "wait" if desc == WAIT_DESC else "trigger" if desc == TRIGGER_DESC else "compute"


def is_communication(event):
    return stage(event).lower() in ("dispatch", "combine")


def is_control(event):
    return event["args"]["desc_id"] == TERMINATE_DESC or "terminate" in stage(event).lower()


def is_routed_up(event):
    return stage(event).lower() in ("gmm1", "upproj", "up_proj") and "owner_id" in event["args"]


def interval(event):
    return event["ts"], event["ts"] + event["dur"]


def forward_events(trace):
    events = [event for event in trace["traceEvents"] if event.get("cat") == "MegaKernelInternal"
              and event.get("args", {}).get("direction") == "forward"]
    if not events:
        raise ValueError("trace contains no forward records")
    invocations = {event["args"]["invocation_id"] for event in events}
    if len(invocations) != 1:
        raise ValueError(f"expected exactly one forward invocation, got {sorted(invocations)}")
    required = {"core_type", "block_id", "task_id", "desc_id", "start_cycle", "end_cycle", "core_entry_cycle"}
    for event in events:
        if not required <= event["args"].keys() or event.get("ph") != "X" or event["dur"] < 0:
            raise ValueError("incomplete or invalid DFX record")
    return events


def audit_trace(trace, rank_record=None):
    events = forward_events(trace)
    compute = [event for event in events if phase(event) == "compute"]
    waits = [event for event in events if phase(event) == "wait"]
    triggers = [event for event in events if phase(event) == "trigger"]
    counts = Counter(event["args"]["task_id"] for event in compute)
    type_counts = Counter(event["args"]["core_type"] for event in compute)
    expected = rank_record["result"]["schedule"]["forward"] if rank_record else None
    expected_counts = ({"AIC": expected["aic_tasks"], "AIV": expected["aiv_tasks"]} if expected else None)
    actual_counts = {kind: type_counts[kind] for kind in ("AIC", "AIV")}
    duplicates = {str(task): count for task, count in counts.items() if count != 1}
    invocation_id = events[0]["args"]["invocation_id"]
    metadata = trace["megaKernelCycleTrace"]
    invocation = next((item for item in metadata.get("invocations", [])
                       if item["invocationId"] == invocation_id), None)
    dropped = invocation["droppedRecordCount"] if invocation else metadata["droppedRecordCount"]
    complete = dropped == 0 and not duplicates and (expected_counts is None or actual_counts == expected_counts)
    ratios = [(event["args"]["end_cycle"] - event["args"]["start_cycle"]) / event["dur"]
              for event in events if event["dur"] > 0]
    frequency = statistics.median(ratios)
    if frequency <= 0 or any(abs(value - frequency) > frequency * 1e-5 for value in ratios):
        raise ValueError("inconsistent cycle-to-microsecond conversion in exported trace")
    entry_cycles = {event["args"]["core_entry_cycle"] for event in events}
    origin = min(event["ts"] for event in events)
    end = max(event["ts"] + event["dur"] for event in events)
    workers = defaultdict(list)
    for event in events:
        args = event["args"]
        workers[f"{args['core_type']}/{args['block_id']}"] .append(event)
    worker_summary = {}
    for worker, items in sorted(workers.items()):
        start = min(event["ts"] for event in items)
        finish = max(event["ts"] + event["dur"] for event in items)
        ordered = sorted(items, key=lambda event: event["ts"])
        worker_compute = [event for event in items if phase(event) == "compute"]
        worker_summary[worker] = {
            "first_record_us": start - origin, "last_record_us": finish - origin,
            "span_us": finish - start,
            "wait_us": sum(event["dur"] for event in items if phase(event) == "wait"),
            "compute_us": sum(event["dur"] for event in worker_compute),
            "trigger_us": sum(event["dur"] for event in items if phase(event) == "trigger"),
            "unrecorded_gaps_us": max(0.0, finish - start - union_length([interval(event) for event in items])),
            "first_record_phase": phase(ordered[0]), "first_record_stage": stage(ordered[0]),
            "first_record_duration_us": ordered[0]["dur"],
            "compute_task_count": len(worker_compute),
            "stages": dict(Counter(stage(event) for event in worker_compute)),
        }
    communication = [event for event in compute if event["args"]["core_type"] == "AIV" and is_communication(event)]
    vector_compute = [event for event in compute if event["args"]["core_type"] == "AIV"
                      and not is_communication(event) and not is_control(event)]
    experts = defaultdict(list)
    expert_waits = defaultdict(list)
    for event in compute:
        if is_routed_up(event):
            experts[event["args"]["owner_id"]].append(event)
    for event in waits:
        if is_routed_up(event):
            expert_waits[event["args"]["owner_id"]].append(event)
    owners = sorted(experts)
    pairs = {f"{first}/{second}": overlap([interval(event) for event in experts[first]],
                                          [interval(event) for event in experts[second]])
             for first, second in zip(owners[::2], owners[1::2])}
    expert_summary = {str(owner): {
        "workers": sorted({event["args"]["block_id"] for event in values}),
        "logical_tasks": sorted(event["args"]["stage_task_index"] for event in values),
        "compute_records": len(values),
        "first_compute_us": min(event["ts"] for event in values) - origin,
        "last_compute_us": max(event["ts"] + event["dur"] for event in values) - origin,
        "active_union_us": union_length([interval(event) for event in values]),
        "maximum_task_wait_us": max((event["dur"] for event in expert_waits[owner]), default=0.0),
        "total_core_wait_us": sum(event["dur"] for event in expert_waits[owner]),
    } for owner, values in experts.items()}
    mode = rank_record.get("mode") if rank_record else None
    role_errors = []
    if mode in ("vector", "both"):
        for event in communication:
            if event["args"]["block_id"] % 2 != 1:
                role_errors.append({"task_id": event["args"]["task_id"], "error": "communication on even AIV"})
        for event in vector_compute:
            if event["args"]["block_id"] % 2 != 0:
                role_errors.append({"task_id": event["args"]["task_id"], "error": "compute on odd AIV"})
    if mode in ("cube", "both") and expected:
        cubes = expected["num_workers"] // 2
        for local, owner in enumerate(owners):
            if local // 2 * 2 + 1 >= len(owners):
                continue
            wanted = list(range(0, cubes // 2)) if local % 2 == 0 else list(range(cubes // 2, cubes))
            observed = expert_summary[str(owner)]["workers"]
            if observed != wanted:
                role_errors.append({"expert": owner, "error": "up projection physical Cube group mismatch",
                                    "expected": wanted, "actual": observed})
    compute_intervals = [interval(event) for event in compute]
    wait_intervals = [interval(event) for event in waits]
    stage_summary = {}
    for name in sorted({stage(event) for event in events}):
        items = [event for event in events if stage(event) == name]
        stage_compute = [event for event in items if phase(event) == "compute"]
        stage_summary[name] = {"compute_records": sum(phase(event) == "compute" for event in items),
                               "compute_core_us": sum(event["dur"] for event in items if phase(event) == "compute"),
                               "wait_core_us": sum(event["dur"] for event in items if phase(event) == "wait"),
                               "maximum_task_wait_us": max((event["dur"] for event in items if phase(event) == "wait"), default=0.0),
                               "first_compute_us": min((event["ts"] for event in stage_compute), default=origin) - origin,
                               "last_compute_us": max((event["ts"] + event["dur"] for event in stage_compute), default=origin) - origin,
                               "active_union_us": union_length([interval(event) for event in stage_compute])}
    return {
        "complete": complete, "schedule_roles_match": not role_errors, "role_errors": role_errors,
        "forward_compute_records": len(compute), "unique_forward_task_ids": len(counts),
        "duplicate_compute_records": len(compute) - len(counts), "duplicate_task_ids": duplicates,
        "expected_forward_task_count_including_terminate": expected["total_tasks"] if expected else None,
        "expected_compute_counts": expected_counts, "actual_compute_counts": actual_counts,
        "terminate_compute_records": sum(is_control(event) for event in compute), "dropped_records": dropped,
        "aic_workers": sorted({event["args"]["block_id"] for event in compute if event["args"]["core_type"] == "AIC"}),
        "aiv_workers": sorted({event["args"]["block_id"] for event in compute if event["args"]["core_type"] == "AIV"}),
        "aiv_stages_by_worker": {key.split("/")[1]: sorted(value["stages"]) for key, value in worker_summary.items() if key.startswith("AIV/")},
        "up_expert_workers": {key: value["workers"] for key, value in expert_summary.items()},
        "up_experts": expert_summary, "up_expert_pair_overlap": pairs,
        "communication_vector_compute_overlap": overlap([interval(event) for event in communication], [interval(event) for event in vector_compute]),
        "timeline_span_us": end - origin,
        "compute_active_union_us": union_length(compute_intervals),
        "wait_union_without_any_recorded_compute_us": union_length(wait_intervals) - intersection_length(wait_intervals, compute_intervals),
        "wait_core_time_us": sum(event["dur"] for event in waits),
        "trigger_core_time_us": sum(event["dur"] for event in triggers),
        "entry_skew_after_ready_us": (max(entry_cycles) - min(entry_cycles)) / frequency,
        "ready_handshake_measured": False, "completion_handshake_measured": False,
        "ready_measurement_limit": "WaitForDeviceReady precedes CycleTraceRecorder.Init; CompleteDeviceReads follows ProcessProfiled. Task dependency waits are measured, entry/completion handshakes are not.",
        "compute_measurement_limit": "Compute records bracket ExecuteComputeKernel, including transport and any in-handler waits. WeightedUnpermute internally waits for local outgoing Combine read completion; this is not a separate WaitDependency record.",
        "cycle_frequency_mhz_from_exported_durations": frequency,
        "workers": worker_summary, "stages": stage_summary,
    }


def color(event):
    if phase(event) != "compute":
        return COLORS[phase(event)]
    name = stage(event)
    if is_routed_up(event):
        return COLORS["up_odd" if event["args"]["owner_id"] % 2 else "up_even"]
    if name.lower().startswith("shared"):
        return COLORS["shared"]
    if name in COLORS:
        return COLORS[name]
    if is_control(event):
        return COLORS["control"]
    return COLORS["epilogue"] if "Unpermute" in name or "Acquire" in name else COLORS["other"]


def draw_pair(baseline, candidate, rank, output, *, zoom=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    cubes = baseline["record"]["result"]["schedule"]["forward"]["num_workers"] // 2
    vectors = 2 * cubes
    lanes = [("AIC", worker) for worker in range(cubes)]
    if not zoom:
        lanes += [("AIV", worker) for worker in range(vectors)]
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    fig, axes = plt.subplots(1, 2, figsize=(19, 7 if zoom else 17), sharex=True, sharey=True)
    max_ms = max(capture["audit"]["timeline_span_us"] / 1000 for capture in (baseline, candidate))
    zoom_limits = []
    for ax, capture, label in zip(axes, (baseline, candidate), ("Baseline: original schedule", "Candidate: AIV split + Cube split")):
        events = forward_events(capture["trace"])
        origin = min(event["ts"] for event in events)
        if zoom:
            owners = sorted(int(owner) for owner in capture["audit"]["up_experts"])
            pair = owners[:2]
            events = [event for event in events if is_routed_up(event) and event["args"]["owner_id"] in pair]
        vertices, colors = [], []
        for event in events:
            args = event["args"]
            lane = (args["core_type"], args["block_id"])
            if lane not in lane_index:
                continue
            y = lane_index[lane]
            start = (event["ts"] - origin) / 1000
            finish = start + event["dur"] / 1000
            vertices.append([(start, y - .36), (finish, y - .36), (finish, y + .36), (start, y + .36)])
            colors.append(color(event))
        ax.add_collection(PolyCollection(vertices, facecolors=colors, edgecolors="none"))
        for worker in range(cubes):
            ax.axhspan(worker - .5, worker + .5, facecolor="#FFF5E9" if worker < cubes // 2 else "#EEF4FC", alpha=.35, zorder=-2)
        ax.axhline(cubes / 2 - .5, color="#A97B45", linewidth=1.0, linestyle="--")
        if not zoom:
            ax.axhline(cubes - .5, color="#3B4650", linewidth=1.2)
        ax.set_yticks(range(len(lanes)), [f"{kind} {worker:02d}" for kind, worker in lanes], fontsize=7)
        ax.tick_params(axis="y", labelleft=True, length=0)
        for tick, (kind, worker) in zip(ax.get_yticklabels(), lanes):
            if kind == "AIV":
                tick.set_color("#267C76" if worker % 2 else "#73A64B")
        ax.set_ylim(len(lanes) - .5, -.8)
        ax.set_xlim(0, max_ms * 1.015)
        ax.set_xlabel("Time since first recorded forward task (ms)", fontsize=10)
        ax.xaxis.grid(True, alpha=.2)
        audit = capture["audit"]
        ax.set_title(f"{label}\nrank {rank}: {len(audit['aic_workers'])} AIC + {len(audit['aiv_workers'])} AIV active; {audit['forward_compute_records']} tasks", fontsize=11)
        if zoom:
            first = min(event["ts"] for event in events if phase(event) == "compute")
            last = max(event["ts"] + event["dur"] for event in events if phase(event) == "compute")
            margin = max((last - first) * .08, 10)
            zoom_limits.append((max(0, (first - origin - margin) / 1000), (last - origin + margin) / 1000))
            metric = audit["up_expert_pair_overlap"][f"{pair[0]}/{pair[1]}"]
            ax.text(.02, .02, f"Expert {pair[0]}/{pair[1]} up overlap: {metric['overlap_us']:.2f} us", transform=ax.transAxes,
                    fontsize=10, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    if zoom:
        axes[0].set_xlim(min(value[0] for value in zoom_limits), max(value[1] for value in zoom_limits))
    handles = [Patch(color=COLORS[key], label=label) for key, label in (
        ("wait", "Dependency wait"), ("trigger", "Event trigger"), ("Dispatch", "Dispatch"), ("Combine", "Combine"),
        ("up_even", "Routed up: even expert"), ("up_odd", "Routed up: odd expert"), ("GMM2", "Routed down"),
        ("SwiGLU", "Activation"), ("shared", "Shared expert"), ("epilogue", "Unpermute / epilogue"))]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9,
               bbox_to_anchor=(.5, .945 if zoom else .965), frameon=False)
    fig.suptitle("MegaMoe forward physical-core timeline" + (" — first routed expert pair" if zoom else ""), fontsize=15, y=.992)
    fig.text(.5, .018 if zoom else .012,
             "Instrumented DFX capture; not module latency. Task execution spans may include internal waits.\n"
             "Ready/completion handshake is outside recorded intervals. Cube background: first / second half.\n"
             "AIV labels: odd = communication group, even = compute group in candidate.",
             ha="center", va="bottom", fontsize=9, color="#525B64")
    fig.subplots_adjust(left=.055, right=.99, top=.80 if zoom else .90,
                        bottom=.17 if zoom else .08, wspace=.12)
    stem = output / f"rank{rank}_{'up_pair_zoom' if zoom else 'forward_timeline'}"
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def load_capture(directory, rank):
    record_path, trace_path = directory / f"rank{rank}.json", directory / f"rank{rank}_trace.json"
    record, trace = json.loads(record_path.read_text()), json.loads(trace_path.read_text())
    if record["phase"] != "profile" or record["rank"] != rank:
        raise ValueError(f"wrong phase/rank in {record_path}")
    audit = audit_trace(trace, record)
    return {"record": record, "trace": trace, "audit": audit,
            "trace_path": str(trace_path), "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", default="0,1,2,3")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    output = {}
    captures = {}
    for rank in (int(value) for value in args.ranks.split(",")):
        baseline, candidate = load_capture(args.baseline, rank), load_capture(args.candidate, rank)
        if baseline["record"]["mode"] != "baseline" or candidate["record"]["mode"] != "both":
            raise ValueError("expected mode baseline versus mode both")
        for name in ("shape", "shared_intermediate_size", "dispatch_mode", "dtype", "training", "all_six_fusion_switches"):
            if baseline["record"]["provenance"][name] != candidate["record"]["provenance"][name]:
                raise ValueError(f"incompatible baseline/candidate setting: {name}")
        captures[rank] = (baseline, candidate)
        output[str(rank)] = {label: {key: capture[key] for key in ("audit", "trace_path", "trace_sha256")}
                             for label, capture in (("baseline", baseline), ("both", candidate))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dfx_analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    invalid = [f"rank{rank}/{mode}" for rank, pair in output.items() for mode, value in pair.items()
               if not value["audit"]["complete"] or not value["audit"]["schedule_roles_match"]]
    if invalid:
        raise SystemExit("Trace completeness or worker-role checks failed: " + ", ".join(invalid))
    if not args.no_plot:
        for rank, (baseline, candidate) in captures.items():
            draw_pair(baseline, candidate, rank, args.output)
            draw_pair(baseline, candidate, rank, args.output, zoom=True)
    print(args.output / "dfx_analysis.json")


if __name__ == "__main__":
    main()
