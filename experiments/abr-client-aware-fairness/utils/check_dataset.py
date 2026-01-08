import os
import pandas as pd
import numpy as np

# === Configurações esperadas ===
NUM_CLIENTS = 20

# DEFAULT DATASET_DIR
# ./multithread/20-clients/trace-lte-driving
# ./multithread/20-clients/trace-lte-short

# ./multiprocess/20-clients/trace-lte-driving
# ./multiprocess/20-clients/trace-lte-short


# DATASET_DIR = "./multithread"
# DATASET_DIR = "./multithread/20-clients/trace-lte-driving"
# DATASET_DIR = "./multithread/20-clients/trace-lte-short"
# DATASET_DIR = "./multiprocess/20-clients/trace-lte-driving"
DATASET_DIR = "./multiprocess/20-clients/trace-lte-short"


# Pega apenas o nome do diretório (sem ./)
dir_name = os.path.basename(os.path.normpath(DATASET_DIR))

REPORT_FILE = f"{DATASET_DIR}/validation_report-{dir_name}.csv"

# Colunas obrigatórias mínimas
REQUIRED_COLUMNS = [
    "client_id", "abr_mode",
    "avg_buffer_level", "avg_latency", "avg_throughput",
    "throughput_samples", "latency_samples", "buffer_level_samples",
    "qfs", "tf_qoe", "bfi", "saf"
]

def longest_ones_streak(values):
    """Retorna o comprimento máximo de sequência consecutiva de 1.0"""
    max_streak, current = 0, 0
    for val in values:
        if val == 1.0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak

def validate_csv(file_path, consecutive_threshold=5):
    issues = []
    summary = {}
    max_streak = 0

    try:
        df = pd.read_csv(file_path)

        # === Estatísticas de client_id ===
        client_ids = sorted(df["client_id"].unique()) if "client_id" in df.columns else []
        client_count = len(client_ids)
        expected_ids = list(range(1, NUM_CLIENTS + 1))
        missing_ids = [cid for cid in expected_ids if cid not in client_ids]

        summary.update({
            "total_rows": len(df),
            "client_count": client_count,
            "client_ids": client_ids,
            "missing_ids": missing_ids if missing_ids else "None",
            "expected_ids": expected_ids
        })

        # 1. Colunas obrigatórias
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                issues.append(f"❌ Missing column: {col}")

        # 2. Número de clientes
        if client_count != NUM_CLIENTS:
            issues.append(f"⚠️ Expected {NUM_CLIENTS} clients, found {client_count}")

        # 3. Fairness entre 0 e 1
        for col in ["qfs", "tf_qoe", "saf", "bfi"]:
            if col in df.columns:
                invalid = df[(df[col] < 0) | (df[col] > 1)]
                if not invalid.empty:
                    issues.append(f"⚠️ Column {col} has values outside [0,1]")

        # 4. Valores inválidos
        invalid_values = df.isin([np.nan, np.inf, -np.inf]).any()
        for col, has_invalid in invalid_values.items():
            if has_invalid:
                issues.append(f"⚠️ Column {col} has NaN or Inf")

        # 5. Métricas que não podem ser negativas
        for col in ["avg_buffer_level", "avg_latency", "avg_throughput"]:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"⚠️ Column {col} has negative values")

        # 6. Colunas de samples não vazias
        sample_cols = [c for c in df.columns if c.endswith("_samples")]
        for col in sample_cols:
            if df[col].isnull().any() or (df[col].astype(str).str.len() == 0).any():
                issues.append(f"⚠️ Column {col} has empty entries")

        # 7. Grupo inválido (0 ou desconhecido)
        if "group_client" in df.columns:
            group_values = df["group_client"].astype(str).unique().tolist()
            summary["group_client_values"] = group_values
            if "0" in group_values:
                issues.append(f"⚠️ Invalid group '0' detected in group_client column")
        else:
            summary["group_client_values"] = "Not found"

        # 8. Checar sequência anômala de BFI == 1
        if "bfi" in df.columns:
            bfi_vals = df["bfi"].round(6).fillna(0).values
            max_streak = longest_ones_streak(bfi_vals)
            if max_streak >= consecutive_threshold:
                algo = df["abr_mode"].iloc[0] if "abr_mode" in df.columns else "UNKNOWN"
                issues.append(
                    f"⚠️ Long sequence of BFI=1 detected (max streak={max_streak}) "
                    f"in algorithm {algo}"
                )

    except Exception as e:
        issues.append(f"❌ Error reading file: {e}")
        summary.update({
            "total_rows": "Error",
            "client_count": "Error",
            "client_ids": "Error",
            "missing_ids": "Error",
            "expected_ids": list(range(1, NUM_CLIENTS + 1)),
            "group_client_values": "Error"
        })

    # Sempre incluir no resumo, mesmo se der erro
    summary["max_bfi_ones_streak"] = max_streak if isinstance(max_streak, (int, float)) else "Error"

    return issues, summary

def main():
    report = []
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(DATASET_DIR, file)
            issues, summary = validate_csv(file_path)
            status = "✅ OK" if not issues else "⚠️ Issues"

            entry = {
                "file": file,
                "status": status,
                "issues": "; ".join(issues) if issues else "None"
            }
            entry.update(summary)
            report.append(entry)

            print(f"[{status}] {file}")
            if issues:
                for issue in issues:
                    print("   -", issue)
            else:
                print("   ✓ All checks passed.")
            print("   ↪ group_client values:", summary.get("group_client_values"))
            print("   ↪ max_bfi_ones_streak:", summary.get("max_bfi_ones_streak"))

    pd.DataFrame(report).to_csv(REPORT_FILE, index=False)
    print(f"\n📄 Validation report saved to: {REPORT_FILE}")

if __name__ == "__main__":
    main()
