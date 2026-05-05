"""
DABC for Flexible Job Shop Scheduling Problem (FJSP)

This script implements the Discrete Artificial Bee Colony (DABC)
algorithm for solving the Kacem 4x5 benchmark problem.

Main outputs:
1. Best Cmax of each run
2. Average Cmax and Success Rate (SR)
3. Best OS and MA representation
4. MA-only dataset for XAI analysis
5. Gantt chart of the best solution
6. Convergence curve of the best run
7. Average convergence curve over 30 runs

Author: Shih-Chia Yeh
"""

import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt


# ==========================================
# 1. Problem Definition: Kacem 4x5 Benchmark
# ==========================================
PROCESSING_TIMES = {
    1: {
        1: [2, 5, 4, 1, 2],
        2: [5, 4, 5, 7, 5],
        3: [4, 5, 5, 4, 5],
    },
    2: {
        1: [2, 5, 4, 7, 8],
        2: [5, 6, 9, 8, 5],
        3: [4, 5, 4, 54, 5],
    },
    3: {
        1: [9, 8, 6, 7, 9],
        2: [6, 1, 2, 5, 4],
        3: [2, 5, 4, 2, 4],
        4: [4, 5, 2, 1, 5],
    },
    4: {
        1: [1, 5, 2, 4, 12],
        2: [5, 1, 2, 1, 2],
    }
}

NUM_JOBS = 4
NUM_MACHINES = 5
JOB_OP_COUNTS = {1: 3, 2: 3, 3: 4, 4: 2}
TOTAL_OPS = sum(JOB_OP_COUNTS.values())

GLOBAL_OP_LIST = []
OP_NAMES = []

for j in range(1, NUM_JOBS + 1):
    for k in range(1, JOB_OP_COUNTS[j] + 1):
        GLOBAL_OP_LIST.append((j, k))
        OP_NAMES.append(f"O{j}{k}")

OP_INDEX_MAP = {op: idx for idx, op in enumerate(GLOBAL_OP_LIST)}


# ==========================================
# 2. DABC Parameters
# ==========================================
SN = 50
MAX_ITER = 600
LIMIT = 20
TARGET_SAMPLES = 5000
NUM_RUNS = 30
BKS = 11


# ==========================================
# 3. Solution Representation
# ==========================================
class Solution:
    """
    A scheduling solution represented by OS and MA.

    OS: Operation sequence
    MA: Machine assignment
    """

    def __init__(self, os_seq=None, ma_assign=None):
        if os_seq is None:
            raw_os = []
            for j in range(1, NUM_JOBS + 1):
                raw_os.extend([j] * JOB_OP_COUNTS[j])
            random.shuffle(raw_os)
            self.os = np.array(raw_os, dtype=int)
        else:
            self.os = np.array(os_seq, dtype=int)

        if ma_assign is None:
            self.ma = np.random.randint(1, NUM_MACHINES + 1, size=TOTAL_OPS, dtype=int)
        else:
            self.ma = np.array(ma_assign, dtype=int)

        self.cmax = 0
        self.trial = 0
        self.schedule = []

        self.calculate_cmax()

    def calculate_cmax(self):
        """
        Decode OS + MA and compute Cmax.
        """
        machine_free_time = np.zeros(NUM_MACHINES, dtype=int)
        job_next_free_time = {j: 0 for j in range(1, NUM_JOBS + 1)}
        job_op_counter = {j: 0 for j in range(1, NUM_JOBS + 1)}

        schedule = []

        for job_id in self.os:
            job_op_counter[job_id] += 1
            op_k = job_op_counter[job_id]

            global_op_index = OP_INDEX_MAP[(job_id, op_k)]
            machine_id = int(self.ma[global_op_index])
            m_idx = machine_id - 1

            p_time = PROCESSING_TIMES[job_id][op_k][m_idx]

            start = max(job_next_free_time[job_id], machine_free_time[m_idx])
            end = start + p_time

            machine_free_time[m_idx] = end
            job_next_free_time[job_id] = end

            schedule.append({
                "job": job_id,
                "op": op_k,
                "machine": machine_id,
                "start": start,
                "end": end,
                "name": f"O{job_id}{op_k}"
            })

        self.cmax = max(job_next_free_time.values())
        self.schedule = schedule


# ==========================================
# 4. Neighborhood Operators
# ==========================================
def generate_neighbor(sol):
    """
    Generate a neighboring solution using one of three operators:
    1. OS-swap
    2. OS-insert
    3. MA-reassign
    """
    new_os = sol.os.copy()
    new_ma = sol.ma.copy()

    r = random.random()

    if r < 0.33:
        i, j = random.sample(range(TOTAL_OPS), 2)
        new_os[i], new_os[j] = new_os[j], new_os[i]

    elif r < 0.66:
        i, j = random.sample(range(TOTAL_OPS), 2)
        val = new_os[i]
        new_os = np.delete(new_os, i)
        new_os = np.insert(new_os, j, val)

    else:
        op_index = random.randint(0, TOTAL_OPS - 1)
        cur_m = int(new_ma[op_index])
        choices = list(range(1, NUM_MACHINES + 1))
        choices.remove(cur_m)
        new_ma[op_index] = random.choice(choices)

    return Solution(new_os, new_ma)


# ==========================================
# 5. Dataset Conversion (MA-only)
# ==========================================
def solution_to_record(sol, idx):
    """
    Convert a solution into one data record using MA-only features.
    """
    record = {"ID": idx}

    for i, name in enumerate(OP_NAMES):
        record[name] = int(sol.ma[i])

    record["Cmax"] = int(sol.cmax)
    return record


# ==========================================
# 6. Single DABC Run
# ==========================================
def run_dabc_collect_dataset(run_id=1):
    """
    Execute one DABC run and collect MA-only dataset records.

    Returns:
        data_records: collected scheduling samples
        best_sol: best solution in this run
        convergence_history: best Cmax of each iteration
    """
    print(f"\n========== Run {run_id} Start ==========")
    print(f"Start DABC (SN={SN}, Iter={MAX_ITER}, Target={TARGET_SAMPLES})")

    population = [Solution() for _ in range(SN)]
    best_sol = copy.deepcopy(min(population, key=lambda x: x.cmax))

    data_records = []
    record_id = 0
    collect_done = False

    convergence_history = [best_sol.cmax]

    # Collect initial population
    for sol in population:
        if len(data_records) < TARGET_SAMPLES:
            data_records.append(solution_to_record(sol, record_id))
            record_id += 1
        else:
            collect_done = True
            break

    for it in range(MAX_ITER):

        # Employed bee phase
        for i in range(SN):
            new_sol = generate_neighbor(population[i])

            if len(data_records) < TARGET_SAMPLES:
                data_records.append(solution_to_record(new_sol, record_id))
                record_id += 1

                if len(data_records) == TARGET_SAMPLES and not collect_done:
                    print(f"Reached target samples = {TARGET_SAMPLES}")
                    collect_done = True

            if new_sol.cmax < population[i].cmax:
                population[i] = new_sol
                population[i].trial = 0
            else:
                population[i].trial += 1

        # Fitness and selection probability
        fitness = [1 / (1 + sol.cmax) for sol in population]
        total_fit = sum(fitness)
        probs = [f / total_fit for f in fitness]

        # Onlooker bee phase
        for _ in range(SN):
            idx = np.random.choice(range(SN), p=probs)
            new_sol = generate_neighbor(population[idx])

            if len(data_records) < TARGET_SAMPLES:
                data_records.append(solution_to_record(new_sol, record_id))
                record_id += 1

                if len(data_records) == TARGET_SAMPLES and not collect_done:
                    print(f"Reached target samples = {TARGET_SAMPLES}")
                    collect_done = True

            if new_sol.cmax < population[idx].cmax:
                population[idx] = new_sol
                population[idx].trial = 0
            else:
                population[idx].trial += 1

        # Scout bee phase
        for i in range(SN):
            if population[i].trial >= LIMIT:
                population[i] = Solution()
                population[i].trial = 0

                if len(data_records) < TARGET_SAMPLES:
                    data_records.append(solution_to_record(population[i], record_id))
                    record_id += 1

                    if len(data_records) == TARGET_SAMPLES and not collect_done:
                        print(f"Reached target samples = {TARGET_SAMPLES}")
                        collect_done = True

        # Update best solution
        cur_best = min(population, key=lambda x: x.cmax)

        if cur_best.cmax < best_sol.cmax:
            best_sol = copy.deepcopy(cur_best)

        convergence_history.append(best_sol.cmax)

        if (it + 1) % 10 == 0:
            print(
                f"Iter {it + 1}/{MAX_ITER}  "
                f"Best Cmax: {best_sol.cmax}  "
                f"Collected: {len(data_records)}"
            )

    print(f"Finish Run {run_id}")
    print("Best OS =", best_sol.os)
    print("Best MA =", best_sol.ma)
    print("Best Cmax =", best_sol.cmax)

    return data_records, best_sol, convergence_history


# ==========================================
# 7. Gantt Chart
# ==========================================
def plot_gantt(solution, title="FJSP Gantt Chart (Best Solution)"):
    """
    Plot the Gantt chart of a scheduling solution.
    Each operation is assigned a distinct color.
    """
    tasks = solution.schedule

    by_machine = {m: [] for m in range(1, NUM_MACHINES + 1)}

    for task in tasks:
        by_machine[task["machine"]].append(task)

    for m in by_machine:
        by_machine[m].sort(key=lambda x: x["start"])

    fig, ax = plt.subplots(figsize=(9, 4.6))

    bar_height = 0.58
    y_positions = {m: m for m in range(1, NUM_MACHINES + 1)}

    custom_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#17becf",
        "#bcbd22",
        "#e377c2",
        "#8c564b",
        "#7f7f7f",
        "#4f81bd",
        "#f79646",
    ]

    op_colors = {
        name: custom_colors[i % len(custom_colors)]
        for i, name in enumerate(OP_NAMES)
    }

    for m in range(1, NUM_MACHINES + 1):
        y = y_positions[m]

        for task in by_machine[m]:
            start = task["start"]
            end = task["end"]
            duration = end - start
            op_name = task["name"]

            ax.barh(
                y,
                duration,
                left=start,
                height=bar_height,
                color=op_colors[op_name],
                edgecolor="black",
                linewidth=1.2
            )

            ax.text(
                start + duration / 2,
                y,
                op_name,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    ax.set_yticks([y_positions[m] for m in range(1, NUM_MACHINES + 1)])
    ax.set_yticklabels([f"M{m}" for m in range(1, NUM_MACHINES + 1)], fontsize=11)

    ax.set_xlabel("Time", fontsize=11)
    ax.set_title(f"{title} (Cmax = {solution.cmax})", fontsize=13)
    ax.set_xlim(0, max(task["end"] for task in tasks) + 1)

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()


# ==========================================
# 8. Convergence Curve (Single Run)
# ==========================================
def plot_convergence(history, title="DABC Convergence Curve"):
    """
    Plot convergence curve of a single DABC run.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cmax so far")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


# ==========================================
# 9. Average Convergence Curve
# ==========================================
def plot_avg_convergence(all_histories, title="Average Convergence Curve over 30 Runs"):
    """
    Plot average convergence curve over multiple runs.
    """
    arr = np.array(all_histories)
    avg_history = np.mean(arr, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(avg_history)), avg_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Average Best Cmax so far")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


# ==========================================
# 10. Main Program
# ==========================================
if __name__ == "__main__":
    all_best_cmax = []
    all_histories = []

    overall_best_sol = None
    overall_best_cmax = float("inf")
    overall_dataset = None

    for run in range(1, NUM_RUNS + 1):
        data_records, best_sol, convergence_history = run_dabc_collect_dataset(run_id=run)

        all_best_cmax.append(best_sol.cmax)
        all_histories.append(convergence_history)

        if best_sol.cmax < overall_best_cmax:
            overall_best_cmax = best_sol.cmax
            overall_best_sol = copy.deepcopy(best_sol)
            overall_dataset = copy.deepcopy(data_records)

    avg_cmax = np.mean(all_best_cmax)
    sr = sum(1 for x in all_best_cmax if x == BKS) / NUM_RUNS

    print("\n==============================")
    print("DABC 30 Runs Summary")
    print("==============================")
    print("Best Cmax of each run =", all_best_cmax)
    print(f"Average Best Cmax = {avg_cmax:.4f}")
    print(f"Success Rate (SR) = {sr:.4f} ({sr * 100:.2f}%)")
    print(f"Overall Best Cmax = {overall_best_cmax}")
    print("Overall Best OS =", overall_best_sol.os)
    print("Overall Best MA =", overall_best_sol.ma)

    df = pd.DataFrame(overall_dataset, columns=["ID"] + OP_NAMES + ["Cmax"])
    filename = "dabc_schedule_dataset_kacem4x5_5000.csv"
    df.to_csv(filename, index=False)

    print(f"\nSaved dataset -> {filename}")

    plot_gantt(
        overall_best_sol,
        title="FJSP Gantt Chart (DABC Best) - Kacem4x5"
    )

    best_run_index = int(np.argmin(all_best_cmax))

    plot_convergence(
        all_histories[best_run_index],
        title=f"DABC Convergence Curve (Best Run = {best_run_index + 1})"
    )

    plot_avg_convergence(
        all_histories,
        title="Average Convergence Curve over 30 Runs"
    )