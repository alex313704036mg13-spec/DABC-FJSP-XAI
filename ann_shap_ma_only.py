"""
DABC for Kacem 4x5 FJSP (MA-only) with ANN Surrogate Model and SHAP Analysis.

Workflow:
1. Solve Kacem 4x5 using DABC.
2. Collect machine-assignment (MA-only) scheduling samples.
3. Train an ANN surrogate model for Cmax prediction.
4. Use SHAP to explain machine assignment importance.

Author: Shih-Chia Yeh
"""

import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

import shap

# ==========================================
# 1. 題目數據定義：Kacem et al. (2002) Table 10, Instance I1 (Kacem4x5)
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
# 2. DABC 參數
# ==========================================
SN = 50
MAX_ITER = 600
LIMIT = 40
TARGET_SAMPLES = 5000

# ==========================================
# 3. Solution 類別
# ==========================================
class Solution:
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

# =========================================================
# 可行性檢查
# =========================================================
def is_feasible(sol):

    # OS 長度檢查
    if len(sol.os) != TOTAL_OPS:
        return False

    # 每個工件出現次數檢查
    for job_id in range(1, NUM_JOBS + 1):
        if np.sum(sol.os == job_id) != JOB_OP_COUNTS[job_id]:
            return False

    # MA 範圍檢查
    if np.any(sol.ma < 1) or np.any(sol.ma > NUM_MACHINES):
        return False

    return True
# ==========================================
# 4. 鄰域操作
# ==========================================
def generate_neighbor(sol):
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
# 5. 將解轉成資料列（MA-only）
# ==========================================
def solution_to_record(sol, idx):
    record = {"ID": idx}

    for (j, k), idx_op in OP_INDEX_MAP.items():
        op_name = f"O{j}{k}"
        record[f"MA_{op_name}"] = int(sol.ma[idx_op])

    record["Cmax"] = int(sol.cmax)

    return record


# ==========================================
# 6. DABC 主流程 + 蒐集資料
# ==========================================
def run_dabc_collect_dataset():
    print(f"Start DABC (SN={SN}, Iter={MAX_ITER}, Target={TARGET_SAMPLES})")

    population = [Solution() for _ in range(SN)]
    best_sol = copy.deepcopy(min(population, key=lambda x: x.cmax))

    data_records = []
    record_id = 0
    collect_done = False

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

        # Fitness & probability
        fitness = [1 / (1 + sol.cmax) for sol in population]
        total_fit = sum(fitness)
        probs = [f / total_fit for f in fitness]

        # Onlooker bees
        for _ in range(SN):

            r = random.random()
            i = 0
            cumulative_prob = 0.0
            chosen_idx = None
            CONT = 1

            while CONT == 1:

                cumulative_prob += probs[i]

                if r <= cumulative_prob:
                    chosen_idx = i
                    CONT = 0

                else:
                    i += 1

                    if i >= SN:
                        CONT = 0
                        break

            if chosen_idx is None:
                chosen_idx = SN - 1

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

                CONT = 1

                while CONT == 1:

                    new_sol = Solution()

                    if is_feasible(new_sol):
                        population[i] = new_sol
                        population[i].trial = 0
                        CONT = 0

                    else:
                        # Reject new_sol，重新生成
                        CONT = 1

                if len(data_records) < TARGET_SAMPLES:
                    data_records.append(solution_to_record(population[i], record_id))
                    record_id += 1

                    if len(data_records) == TARGET_SAMPLES and not collect_done:
                        print(f"Reached target samples = {TARGET_SAMPLES}")
                        collect_done = True

        # 更新最佳解
        cur_best = min(population, key=lambda x: x.cmax)
        if cur_best.cmax < best_sol.cmax:
            best_sol = copy.deepcopy(cur_best)

        if (it + 1) % 10 == 0:
            print(f"Iter {it+1}/{MAX_ITER}  Best Cmax: {best_sol.cmax}  Collected: {len(data_records)}")

    print("Finish DABC")
    return data_records, best_sol


# ==========================================
# 7. 畫甘特圖
# ==========================================
def plot_gantt(solution, title="FJSP Gantt Chart (Best Solution)"):
    tasks = solution.schedule

    by_machine = {m: [] for m in range(1, NUM_MACHINES + 1)}
    for t in tasks:
        by_machine[t["machine"]].append(t)

    for m in by_machine:
        by_machine[m].sort(key=lambda x: x["start"])

    fig, ax = plt.subplots()

    bar_height = 0.6
    y_positions = {m: m for m in range(1, NUM_MACHINES + 1)}

    for m in range(1, NUM_MACHINES + 1):
        y = y_positions[m]
        for t in by_machine[m]:
            start = t["start"]
            end = t["end"]
            duration = end - start
            ax.barh(y, duration, left=start, height=bar_height)
            ax.text(start + duration / 2, y, t["name"], ha="center", va="center", fontsize=9)

    ax.set_yticks([y_positions[m] for m in range(1, NUM_MACHINES + 1)])
    ax.set_yticklabels([f"M{m}" for m in range(1, NUM_MACHINES + 1)])
    ax.set_xlabel("Time")
    ax.set_title(f"{title}  (Cmax = {solution.cmax})")
    ax.set_xlim(0, max(t["end"] for t in tasks) + 1)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()


# ==========================================
# 8. 主程式：MA-only dataset + ANN + SHAP
# ==========================================
if __name__ == "__main__":

    # -----------------------------
    # Part A：DABC 產生 MA-only dataset
    # -----------------------------
    data_records, best_sol = run_dabc_collect_dataset()

    ma_columns = [f"MA_{name}" for name in OP_NAMES]
    all_columns = ["ID"] + ma_columns + ["Cmax"]

    df = pd.DataFrame(data_records, columns=all_columns)

    print("\nDataset Preview (MA-only)")
    print(df.head())
    print(f"\nTotal rows = {len(df)}")

    filename = "dabc_schedule_dataset_kacem4x5_5000_MA_only.csv"
    df.to_csv(filename, index=False)
    print(f"\nSaved dataset -> {filename}")

    print("\nBest Solution (1-based)")
    print("OS =", best_sol.os)
    print("MA =", best_sol.ma)
    print("Cmax =", best_sol.cmax)

    plot_gantt(best_sol, title="FJSP Gantt Chart (DABC Best) - Kacem4x5")

    # -----------------------------
    # Part B：ANN + SHAP
    # -----------------------------
    df.columns = df.columns.str.strip()

    ma_cols = [c for c in df.columns if c.startswith("MA_")]
    feature_cols = ma_cols

    X = df[feature_cols].astype(float).values
    y = df["Cmax"].astype(float).values

    print("Loaded from in-memory dataframe")
    print("Columns =", df.columns.tolist())
    print("X shape =", X.shape, "| y shape =", y.shape)
    print("MA features =", ma_cols)
    print("Total features =", feature_cols)

    if len(feature_cols) == 0:
        raise ValueError("找不到特徵欄位，請檢查 DataFrame 是否包含 MA_ 開頭之欄位。")

    # =========================================================
    # 2) 訓練集 / 測試集切分
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================
    # 3) Z-score 標準化
    # =========================================================
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # =========================================================
    # 4) 建立並訓練 ANN 代理模型
    # =========================================================
    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        Input(shape=(X_train_s.shape[1],)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=500,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # =========================================================
    # 5) 模型評估（Train / Test）
    # =========================================================
    y_train_pred = model.predict(X_train_s, verbose=0).reshape(-1)
    y_test_pred = model.predict(X_test_s, verbose=0).reshape(-1)

    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print("\n================ Model Performance ================")
    print("\nTrain:")
    print("R2   =", train_r2)
    print("RMSE =", train_rmse)
    print("MAE  =", train_mae)
    print("MAPE =", train_mape, "%")

    print("\nTest:")
    print("R2   =", test_r2)
    print("RMSE =", test_rmse)
    print("MAE  =", test_mae)
    print("MAPE =", test_mape, "%")

    # =========================================================
    # 6) 預測值與實際值比較圖
    # =========================================================
    n_plot = min(100, len(y_test))
    y_test_plot = y_test[:n_plot]
    y_test_pred_plot = y_test_pred[:n_plot]

    plot_df = pd.DataFrame({
        "Actual": y_test_plot,
        "Predicted": y_test_pred_plot
    }).sort_values(by="Actual").reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        plot_df.index + 1,
        plot_df["Actual"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Actual Cmax"
    )
    plt.plot(
        plot_df.index + 1,
        plot_df["Predicted"],
        marker="s",
        markersize=4,
        linewidth=1.5,
        label="Predicted Cmax",
        alpha=0.8
    )

    plt.xlabel("Sorted Sample Index (Smallest to Largest Cmax)", fontsize=11)
    plt.ylabel("Cmax", fontsize=11)
    plt.title("Comparison of Actual and Predicted Cmax (MA-only)", fontsize=13, fontweight="bold")
    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("kacem4x5_ma_only_cmax_comparison_sorted.png", dpi=300)
    plt.show()

    # =========================================================
    # 7) SHAP 解釋
    # =========================================================
    bg_size = min(100, X_train_s.shape[0])
    plot_size = min(200, X_test_s.shape[0])

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(X_train_s.shape[0], bg_size, replace=False)
    plot_idx = rng.choice(X_test_s.shape[0], plot_size, replace=False)

    X_bg = X_train_s[bg_idx]
    X_plot = X_test_s[plot_idx]
    X_plot_df = pd.DataFrame(X_plot, columns=feature_cols)
    y_plot_actual = y_test[plot_idx]

    try:
        explainer = shap.DeepExplainer(model, X_bg)
        sv = explainer.shap_values(X_plot)
        shap_values = sv[0] if isinstance(sv, list) else sv

        ev = explainer.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            ev = ev[0]
        try:
            base_value = float(ev)
        except Exception:
            base_value = float(ev.numpy())

        print("[SHAP] DeepExplainer ok")

    except Exception as e:
        print("[SHAP] DeepExplainer failed -> GradientExplainer fallback")
        print("Reason:", e)

        explainer = shap.GradientExplainer(model, X_bg)
        sv = explainer.shap_values(X_plot)
        shap_values = sv[0] if isinstance(sv, list) else sv

        if hasattr(explainer, "expected_value"):
            ev = explainer.expected_value
            if isinstance(ev, (list, tuple, np.ndarray)):
                ev = ev[0]
            try:
                base_value = float(ev)
            except Exception:
                try:
                    base_value = float(ev.numpy())
                except Exception:
                    base_value = float(np.mean(model.predict(X_bg, verbose=0)))
        else:
            base_value = float(np.mean(model.predict(X_bg, verbose=0)))

    # 統一 SHAP 輸出形狀
    shap_arr = np.array(shap_values)
    print("SHAP raw shape:", shap_arr.shape)

    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 1:
        shap_arr = shap_arr[..., 0]

    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)

    print("SHAP final shape:", shap_arr.shape)

    shap_df = pd.DataFrame(shap_arr, columns=feature_cols)
    mean_abs_shap = shap_df.abs().mean(axis=0)
    ma_importance = mean_abs_shap[ma_cols].sort_values(ascending=True)

    # =========================================================
    # 8) 圖1：MA importance
    # =========================================================
    plt.figure(figsize=(10, 6))
    plt.barh(ma_importance.index, ma_importance.values)
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("MA Features")
    plt.title("Aggregated SHAP Importance at Operation Level(MA-only)")
    plt.tight_layout()
    plt.savefig("kacem4x5_ma_only_shap_1_ma_importance.png", dpi=300)
    plt.show()
    print("Saved: kacem4x5_ma_only_shap_1_ma_importance.png")


    # =========================================================
    # 9) 圖2：MA-only beeswarm（單色版）
    plt.figure()
    shap.summary_plot(
        shap_arr,  # 只傳 SHAP values
        feature_names=feature_cols,  # 保留名稱
        show=False
    )

    plt.title("SHAP Beeswarm Plot (Operation Level,MA-only)")
    plt.tight_layout()
    plt.savefig("kacem4x5_ma_only_shap_2_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved: kacem4x5_ma_only_shap_2_beeswarm.png")

    # =========================================================
    # 10) 圖3：MA-only waterfall
    # =========================================================
    y_plot_pred = model.predict(X_plot, verbose=0).reshape(-1)

    single_idx = int(
        np.argmin(
            np.abs(y_plot_pred - y_plot_actual)
        )
    )

    exp = shap.Explanation(
        values=shap_df.iloc[single_idx].values,
        base_values=base_value,
        data=None,
        feature_names=feature_cols
    )

    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=12)
    plt.title("SHAP Waterfall (MA-only), Representative Solution)")
    plt.tight_layout()
    plt.savefig("kacem4x5_ma_only_shap_3_waterfall.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved: kacem4x5_ma_only_shap_3_waterfall.png")

    # =========================================================
    # 11) 輸出重要度表
    # =========================================================
    ma_rank_df = ma_importance.sort_values(ascending=False).reset_index()
    ma_rank_df.columns = ["Feature", "MeanAbsSHAP"]

    print("\n=== MA-only Importance Ranking ===")
    print(ma_rank_df)

    ma_rank_df.to_csv("shap_ma_only_importance.csv", index=False)
    print("Saved: shap_ma_only_importance.csv")

    # =========================================================
    # 12) 完成
    # =========================================================
    print("\nDone.")
    print("\nTrain:")
    print("R2   =", train_r2)
    print("RMSE =", train_rmse)
    print("MAE  =", train_mae)
    print("MAPE =", train_mape, "%")

    print("\nTest:")
    print("R2   =", test_r2)
    print("RMSE =", test_rmse)
    print("MAE  =", test_mae)
    print("MAPE =", test_mape, "%")
