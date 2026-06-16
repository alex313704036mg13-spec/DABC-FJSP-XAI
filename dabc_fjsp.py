"""
Discrete Artificial Bee Colony (DABC) for the Kacem 4x5 FJSP instance.

This script solves the Kacem et al. (2002) 4x5 flexible job shop scheduling
problem using a discrete ABC algorithm. It also collects machine-assignment
(MA-only) samples for later explainability or surrogate-model analysis.

Key settings:
- Objective: minimize makespan (Cmax)
- Encoding: x = (OS, MA)
  - OS: operation sequence represented by job IDs
  - MA: machine assignment stored in global operation order
- Main experiment: SN=50, MaxIter=600, limit=40, Runs=30
"""

import copy
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# 1. Problem data: Kacem et al. (2002), Table 10, Instance I1
# =========================================================
PROCESSING_TIMES: Dict[int, Dict[int, List[int]]] = {
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
    },
}

NUM_JOBS = 4
NUM_MACHINES = 5
JOB_OP_COUNTS = {1: 3, 2: 3, 3: 4, 4: 2}
TOTAL_OPS = sum(JOB_OP_COUNTS.values())

GLOBAL_OP_LIST: List[Tuple[int, int]] = []
OP_NAMES: List[str] = []
for job_id in range(1, NUM_JOBS + 1):
    for op_id in range(1, JOB_OP_COUNTS[job_id] + 1):
        GLOBAL_OP_LIST.append((job_id, op_id))
        OP_NAMES.append(f"O{job_id}{op_id}")

OP_INDEX_MAP = {op: idx for idx, op in enumerate(GLOBAL_OP_LIST)}


# =========================================================
# 2. DABC parameters
# =========================================================
SN = 50
MAX_ITER = 600
LIMIT = 40
TARGET_SAMPLES = 5000
NUM_RUNS = 30
BKS = 11


# =========================================================
# 3. Solution representation
# =========================================================
class Solution:
    """A scheduling solution encoded as x = (OS, MA)."""

    def __init__(self, os_seq=None, ma_assign=None):
        if os_seq is None:
            raw_os = []
            for job_id in range(1, NUM_JOBS + 1):
                raw_os.extend([job_id] * JOB_OP_COUNTS[job_id])
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

    def calculate_cmax(self) -> None:
        """Decode the solution and calculate its makespan."""
        machine_free_time = np.zeros(NUM_MACHINES, dtype=int)
        job_next_free_time = {job_id: 0 for job_id in range(1, NUM_JOBS + 1)}
        job_op_counter = {job_id: 0 for job_id in range(1, NUM_JOBS + 1)}

        schedule = []

        for job_id in self.os:
            job_op_counter[job_id] += 1
            op_id = job_op_counter[job_id]

            global_op_index = OP_INDEX_MAP[(job_id, op_id)]
            machine_id = int(self.ma[global_op_index])
            machine_idx = machine_id - 1

            processing_time = PROCESSING_TIMES[job_id][op_id][machine_idx]

            start = max(job_next_free_time[job_id], machine_free_time[machine_idx])
            end = start + processing_time

            machine_free_time[machine_idx] = end
            job_next_free_time[job_id] = end

            schedule.append(
                {
                    "job": job_id,
                    "op": op_id,
                    "machine": machine_id,
                    "start": start,
                    "end": end,
                    "name": f"O{job_id}{op_id}",
                }
            )

        self.cmax = max(job_next_free_time.values())
        self.schedule = schedule


# =========================================================
# 4. Feasibility check
# =========================================================
def is_feasible(sol: Solution) -> bool:
    """
    Check whether a solution belongs to the feasible solution set.

    For the Kacem 4x5 instance, all operations can be processed by all five
    machines, so the MA range check is sufficient. This function is retained
    to make the DABC structure extendable to FJSP instances with stricter
    machine-eligibility constraints.
    """
    if len(sol.os) != TOTAL_OPS:
        return False

    for job_id in range(1, NUM_JOBS + 1):
        if np.sum(sol.os == job_id) != JOB_OP_COUNTS[job_id]:
            return False

    if np.any(sol.ma < 1) or np.any(sol.ma > NUM_MACHINES):
        return False

    return True


def create_feasible_solution() -> Solution:
    """Generate a feasible solution; reject and regenerate if needed."""
    cont = True
    while cont:
        new_sol = Solution()
        if is_feasible(new_sol):
            cont = False
        else:
            # Reject the infeasible solution and regenerate.
            continue
    return new_sol


# =========================================================
# 5. Neighborhood operations
# =========================================================
def generate_neighbor(sol: Solution) -> Solution:
    """Generate a neighboring solution by OS-swap, OS-insert, or MA-reassign."""
    new_os = sol.os.copy()
    new_ma = sol.ma.copy()

    r = random.random()

    if r < 0.33:
        idx1, idx2 = random.sample(range(TOTAL_OPS), 2)
        new_os[idx1], new_os[idx2] = new_os[idx2], new_os[idx1]

    elif r < 0.66:
        idx1, idx2 = random.sample(range(TOTAL_OPS), 2)
        value = new_os[idx1]
        new_os = np.delete(new_os, idx1)
        new_os = np.insert(new_os, idx2, value)

    else:
        op_index = random.randint(0, TOTAL_OPS - 1)
        current_machine = int(new_ma[op_index])
        machine_choices = list(range(1, NUM_MACHINES + 1))
        machine_choices.remove(current_machine)
        new_ma[op_index] = random.choice(machine_choices)

    return Solution(new_os, new_ma)


# =========================================================
# 6. Roulette-wheel food source selection
# =========================================================
def roulette_select(probs: List[float]) -> int:
    """
    Select a food source using roulette-wheel selection.

    This implementation explicitly follows the rule-based decision logic:
    scan food-source index i, accumulate probability, and select x_i when
    the random number falls in the corresponding cumulative probability range.
    """
    r = random.random()
    i = 0
    cumulative_prob = 0.0
    chosen_idx = None
    cont = True

    while cont:
        cumulative_prob += probs[i]

        if r <= cumulative_prob:
            chosen_idx = i
            cont = False
        else:
            i += 1

            if i >= SN:
                cont = False
                break

    if chosen_idx is None:
        chosen_idx = SN - 1

    return chosen_idx


# =========================================================
# 7. Convert solution to dataset record (MA-only)
# =========================================================
def solution_to_record(sol: Solution, record_id: int) -> Dict[str, int]:
    record = {"ID": record_id}
    for idx, name in enumerate(OP_NAMES):
        record[name] = int(sol.ma[idx])
    record["Cmax"] = int(sol.cmax)
    return record


# =========================================================
# 8. Single DABC run and data collection
# =========================================================
def run_dabc_collect_dataset(run_id: int = 1):
    print(f"\n========== Run {run_id} Start ==========")
    print(f"Start DABC (SN={SN}, Iter={MAX_ITER}, Target={TARGET_SAMPLES})")

    population = [create_feasible_solution() for _ in range(SN)]
    best_sol = copy.deepcopy(min(population, key=lambda x: x.cmax))

    data_records = []
    record_id = 0
    collect_done = False

    convergence_history = [best_sol.cmax]

    for sol in population:
        if len(data_records) < TARGET_SAMPLES:
            data_records.append(solution_to_record(sol, record_id))
            record_id += 1
        else:
            collect_done = True
            break

    for it in range(MAX_ITER):
        # Employed bees
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
        probs = [fit / total_fit for fit in fitness]

        # Onlooker bees
        for _ in range(SN):
            chosen_idx = roulette_select(probs)
            new_sol = generate_neighbor(population[chosen_idx])

            if len(data_records) < TARGET_SAMPLES:
                data_records.append(solution_to_record(new_sol, record_id))
                record_id += 1
                if len(data_records) == TARGET_SAMPLES and not collect_done:
                    print(f"Reached target samples = {TARGET_SAMPLES}")
                    collect_done = True

            if new_sol.cmax < population[chosen_idx].cmax:
                population[chosen_idx] = new_sol
                population[chosen_idx].trial = 0
            else:
                population[chosen_idx].trial += 1

        # Scout bees
        for i in range(SN):
            if population[i].trial >= LIMIT:
                population[i] = create_feasible_solution()
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


# =========================================================
# 9. Plot functions
# =========================================================
def plot_gantt(solution: Solution, title: str = "FJSP Gantt Chart (Best Solution)") -> None:
    tasks = solution.schedule

    by_machine = {machine_id: [] for machine_id in range(1, NUM_MACHINES + 1)}
    for task in tasks:
        by_machine[task["machine"]].append(task)

    for machine_id in by_machine:
        by_machine[machine_id].sort(key=lambda x: x["start"])

    fig, ax = plt.subplots(figsize=(9, 4.6))

    bar_height = 0.58
    y_positions = {machine_id: machine_id for machine_id in range(1, NUM_MACHINES + 1)}

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
    op_colors = {name: custom_colors[idx % len(custom_colors)] for idx, name in enumerate(OP_NAMES)}

    for machine_id in range(1, NUM_MACHINES + 1):
        y = y_positions[machine_id]
        for task in by_machine[machine_id]:
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
                linewidth=1.2,
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


def plot_convergence(history: List[int], title: str = "DABC Convergence Curve") -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cmax")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_avg_convergence(all_histories: List[List[int]], title: str = "Average Convergence Curve over 30 Runs") -> None:
    arr = np.array(all_histories)
    avg_history = np.mean(arr, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(avg_history)), avg_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Average Cmax")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


# =========================================================
# 10. Main program: 30 runs, average Cmax, and success rate
# =========================================================
def first_reach_iteration(history: List[int], target: int):
    for iteration, value in enumerate(history):
        if value <= target:
            return iteration
    return float("inf")


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
    sr = sum(1 for value in all_best_cmax if value == BKS) / NUM_RUNS

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

    plot_gantt(overall_best_sol, title="FJSP Gantt Chart (DABC Best) - Kacem4x5")

    reach_iters = [first_reach_iteration(history, BKS) for history in all_histories]

    if min(reach_iters) < float("inf"):
        best_run_index = int(np.argmin(reach_iters))
        curve_title = f"DABC Convergence Curve (Best Run = {best_run_index + 1})"
    else:
        best_run_index = int(np.argmin(all_best_cmax))
        curve_title = f"DABC Convergence Curve (Best Final Run = Run {best_run_index + 1})"

    plot_convergence(all_histories[best_run_index], title=curve_title)
    plot_avg_convergence(all_histories, title="Average Convergence Curve over 30 Runs")
