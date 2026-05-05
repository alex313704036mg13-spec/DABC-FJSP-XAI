"""
Baseline Methods for FJSP (Kacem 4x5)

This script implements baseline scheduling methods for comparison with DABC:
1. Random Search (RS)
2. Shortest Processing Time (SPT) with ECM
3. Most Work Remaining (MWR) with ECM

Main outputs:
1. Random Search 30-run statistics
2. Best / Avg / Worst / Std / SR of Random Search
3. OS, MA, and Cmax of RS, SPT, and MWR
4. Gantt charts of RS, SPT, and MWR

Author: Shih-Chia Yeh
"""

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ==========================================
# 1. Problem Definition: Kacem 4x5 Benchmark
# ==========================================
PROCESSING_TIMES = {
    1: {1: [2, 5, 4, 1, 2], 2: [5, 4, 5, 7, 5], 3: [4, 5, 5, 4, 5]},
    2: {1: [2, 5, 4, 7, 8], 2: [5, 6, 9, 8, 5], 3: [4, 5, 4, 54, 5]},
    3: {1: [9, 8, 6, 7, 9], 2: [6, 1, 2, 5, 4], 3: [2, 5, 4, 2, 4], 4: [4, 5, 2, 1, 5]},
    4: {1: [1, 5, 2, 4, 12], 2: [5, 1, 2, 1, 2]}
}

NUM_JOBS = 4
NUM_MACHINES = 5
JOB_OP_COUNTS = {1: 3, 2: 3, 3: 4, 4: 2}
TOTAL_OPS = sum(JOB_OP_COUNTS.values())


# ==========================================
# 2. Experimental Settings
# ==========================================
BKS = 11
NUM_RUNS = 30
NUM_EVALS = 5000
BASE_SEED = 42


# ==========================================
# 3. Global Operation Order
# ==========================================
GLOBAL_OP_ORDER = []

for j in range(1, NUM_JOBS + 1):
    for k in range(1, JOB_OP_COUNTS[j] + 1):
        GLOBAL_OP_ORDER.append((j, k))

OP_INDEX_MAP = {op: idx for idx, op in enumerate(GLOBAL_OP_ORDER)}


# ==========================================
# 4. Decoder: OS + MA to Schedule and Cmax
# ==========================================
def decode_schedule(os_vector, ma_vector):
    machine_free_time = np.zeros(NUM_MACHINES + 1, dtype=int)
    job_free_time = {j: 0 for j in range(1, NUM_JOBS + 1)}
    job_op_counter = {j: 1 for j in range(1, NUM_JOBS + 1)}

    schedule = []

    for job_id in os_vector:
        op_id = job_op_counter[job_id]

        ma_index = OP_INDEX_MAP[(job_id, op_id)]
        machine_id = ma_vector[ma_index]

        proc_time = PROCESSING_TIMES[job_id][op_id][machine_id - 1]

        start_time = max(job_free_time[job_id], machine_free_time[machine_id])
        end_time = start_time + proc_time

        machine_free_time[machine_id] = end_time
        job_free_time[job_id] = end_time
        job_op_counter[job_id] += 1

        schedule.append({
            "job": job_id,
            "op": op_id,
            "m": machine_id,
            "s": start_time,
            "e": end_time,
            "name": f"O{job_id}{op_id}"
        })

    cmax = max(task["e"] for task in schedule)

    return schedule, cmax


# ==========================================
# 5. ECM: Earliest Completion Machine
# ==========================================
def choose_ecm_machine(job_id, op_id, job_ready_time, machine_free_time):
    best_m = None
    best_completion = float("inf")
    best_proc_time = float("inf")

    times = PROCESSING_TIMES[job_id][op_id]

    for m_id in range(1, NUM_MACHINES + 1):
        proc_time = times[m_id - 1]
        completion = max(job_ready_time, machine_free_time[m_id]) + proc_time

        if (
            completion < best_completion
            or (completion == best_completion and proc_time < best_proc_time)
            or (completion == best_completion and proc_time == best_proc_time and m_id < best_m)
        ):
            best_completion = completion
            best_proc_time = proc_time
            best_m = m_id

    return best_m, best_proc_time, best_completion


# ==========================================
# 6. Remaining Work for MWR Rule
# ==========================================
def calc_remaining_work(job_id, next_op):
    if next_op > JOB_OP_COUNTS[job_id]:
        return -1

    total = 0

    for op_id in range(next_op, JOB_OP_COUNTS[job_id] + 1):
        total += min(PROCESSING_TIMES[job_id][op_id])

    return total


# ==========================================
# 7. Dispatching Rule: SPT / MWR + ECM
# ==========================================
def solve_fjsp_dispatch(rule="SPT"):
    rule = rule.upper()

    if rule not in ["SPT", "MWR"]:
        raise ValueError("rule must be 'SPT' or 'MWR'.")

    job_next_op = {j: 1 for j in range(1, NUM_JOBS + 1)}
    machine_free_time = np.zeros(NUM_MACHINES + 1, dtype=int)
    job_free_time = {j: 0 for j in range(1, NUM_JOBS + 1)}

    schedule = []
    final_ma_mapping = {}

    completed_ops = 0

    while completed_ops < TOTAL_OPS:
        candidates = []

        for j_id in range(1, NUM_JOBS + 1):
            op_id = job_next_op[j_id]

            if op_id > JOB_OP_COUNTS[j_id]:
                continue

            best_m, best_p, best_completion = choose_ecm_machine(
                j_id,
                op_id,
                job_free_time[j_id],
                machine_free_time
            )

            remaining_work = calc_remaining_work(j_id, op_id)

            candidates.append({
                "job": j_id,
                "op": op_id,
                "machine": best_m,
                "proc_time": best_p,
                "completion": best_completion,
                "remaining_work": remaining_work
            })

        if rule == "SPT":
            next_task = min(candidates, key=lambda x: (x["proc_time"], x["job"]))
        else:
            next_task = max(candidates, key=lambda x: (x["remaining_work"], -x["job"]))

        j_id = next_task["job"]
        op_id = next_task["op"]
        m_id = next_task["machine"]

        duration = PROCESSING_TIMES[j_id][op_id][m_id - 1]

        start_time = max(job_free_time[j_id], machine_free_time[m_id])
        end_time = start_time + duration

        machine_free_time[m_id] = end_time
        job_free_time[j_id] = end_time

        schedule.append({
            "job": j_id,
            "op": op_id,
            "m": m_id,
            "s": start_time,
            "e": end_time,
            "name": f"O{j_id}{op_id}"
        })

        final_ma_mapping[(j_id, op_id)] = m_id
        job_next_op[j_id] += 1
        completed_ops += 1

    cmax = max(task["e"] for task in schedule)
    os_vector = [task["job"] for task in schedule]
    ma_vector = [final_ma_mapping[op] for op in GLOBAL_OP_ORDER]

    return schedule, os_vector, ma_vector, cmax


# ==========================================
# 8. Random Search
# ==========================================
def run_single_random_search(n_evals=5000, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    best_cmax = float("inf")
    best_os = None
    best_ma = None
    best_schedule = None

    os_template = []

    for job_id, count in JOB_OP_COUNTS.items():
        os_template.extend([job_id] * count)

    for _ in range(n_evals):
        random_os = copy.deepcopy(os_template)
        random.shuffle(random_os)

        random_ma = [random.randint(1, NUM_MACHINES) for _ in range(TOTAL_OPS)]

        current_schedule, current_cmax = decode_schedule(random_os, random_ma)

        if current_cmax < best_cmax:
            best_cmax = current_cmax
            best_os = random_os
            best_ma = random_ma
            best_schedule = current_schedule

    return best_cmax, best_os, best_ma, best_schedule


def run_random_search(num_runs=30, num_evals=5000, bks=11):
    run_best_results = []

    overall_best_cmax = float("inf")
    overall_best_os = None
    overall_best_ma = None
    overall_best_schedule = None

    print(f"Start Random Search: {num_runs} runs, {num_evals} evaluations per run")

    for run in range(1, num_runs + 1):
        best_cmax, best_os, best_ma, best_schedule = run_single_random_search(
            n_evals=num_evals,
            seed=BASE_SEED + run - 1
        )

        run_best_results.append(best_cmax)

        if best_cmax < overall_best_cmax:
            overall_best_cmax = best_cmax
            overall_best_os = best_os
            overall_best_ma = best_ma
            overall_best_schedule = best_schedule

        print(f"Run {run:2d}: Best Cmax = {best_cmax}")

    run_best_results = np.array(run_best_results)

    avg_cmax = np.mean(run_best_results)
    std_cmax = np.std(run_best_results)
    worst_cmax = np.max(run_best_results)
    sr = np.sum(run_best_results == bks) / num_runs * 100

    print("=" * 70)
    print("Random Search Summary")
    print("=" * 70)
    print(f"Runs                : {num_runs}")
    print(f"Evaluations per run : {num_evals}")
    print(f"Best Cmax           : {overall_best_cmax}")
    print(f"Avg Cmax            : {avg_cmax:.4f}")
    print(f"Worst Cmax          : {worst_cmax}")
    print(f"Std Dev             : {std_cmax:.4f}")
    print(f"Success Rate (SR)   : {sr:.2f}%")
    print("-" * 70)
    print(f"Best OS*            : {overall_best_os}")
    print(f"Best MA*            : {overall_best_ma}")
    print("=" * 70)

    return {
        "method": "RS",
        "best_each_run": run_best_results.tolist(),
        "best_cmax": overall_best_cmax,
        "avg_cmax": avg_cmax,
        "worst_cmax": worst_cmax,
        "std_cmax": std_cmax,
        "sr": sr,
        "best_os": overall_best_os,
        "best_ma": overall_best_ma,
        "best_schedule": overall_best_schedule
    }


# ==========================================
# 9. Gantt Chart
# ==========================================
def plot_gantt(schedule, title):
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        1: "#1f77b4",
        2: "#ff7f0e",
        3: "#2ca02c",
        4: "#d62728"
    }

    max_cmax = 0

    for task in schedule:
        ax.barh(
            task["m"],
            task["e"] - task["s"],
            left=task["s"],
            color=colors[task["job"]],
            edgecolor="black",
            alpha=0.85
        )

        ax.text(
            task["s"] + (task["e"] - task["s"]) / 2,
            task["m"],
            task["name"],
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold"
        )

        max_cmax = max(max_cmax, task["e"])

    ax.set_yticks(range(1, NUM_MACHINES + 1))
    ax.set_yticklabels([f"M{i}" for i in range(1, NUM_MACHINES + 1)])
    ax.xaxis.set_major_locator(MultipleLocator(1))

    ax.set_xlabel("Time")
    ax.set_title(f"{title} (Cmax = {max_cmax})")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# ==========================================
# 10. Main Program
# ==========================================
if __name__ == "__main__":

    # Random Search
    rs_result = run_random_search(
        num_runs=NUM_RUNS,
        num_evals=NUM_EVALS,
        bks=BKS
    )

    print("\n")
    print("=" * 70)
    print("Random Search Best Solution")
    print("=" * 70)
    print(f"OS*  : {rs_result['best_os']}")
    print(f"MA*  : {rs_result['best_ma']}")
    print(f"Cmax : {rs_result['best_cmax']}")
    print("=" * 70)

    plot_gantt(rs_result["best_schedule"], title="Random Search Best Gantt Chart")

    # SPT + ECM
    spt_schedule, spt_os, spt_ma, spt_cmax = solve_fjsp_dispatch(rule="SPT")

    print("\n")
    print("=" * 70)
    print("SPT + ECM Result")
    print("=" * 70)
    print(f"OS*  : {spt_os}")
    print(f"MA*  : {spt_ma}")
    print(f"Cmax : {spt_cmax}")
    print("=" * 70)

    plot_gantt(spt_schedule, title="SPT + ECM Gantt Chart")

    # MWR + ECM
    mwr_schedule, mwr_os, mwr_ma, mwr_cmax = solve_fjsp_dispatch(rule="MWR")

    print("\n")
    print("=" * 70)
    print("MWR + ECM Result")
    print("=" * 70)
    print(f"OS*  : {mwr_os}")
    print(f"MA*  : {mwr_ma}")
    print(f"Cmax : {mwr_cmax}")
    print("=" * 70)

    plot_gantt(mwr_schedule, title="MWR + ECM Gantt Chart")