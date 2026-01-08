import os
import pandas as pd
import numpy as np
import re

# === CONFIGURAÇÃO: altere o diretório aqui ===
# DATASET_DIR = "./multithread/20-clients/trace-lte-driving"
# DATASET_DIR = "./multithread/20-clients/trace-lte-short"
# DATASET_DIR = "./multiprocess/20-clients/trace-lte-driving"
DATASET_DIR = "./multiprocess/20-clients/trace-lte-short"

FAIRNESS_COLS = ["bfi", "tf_qoe", "qfs", "saf"]

def extract_algorithm_from_filename(fname):
    m = re.search(r"simulation-ABR-([^-]+)-group", fname)
    return m.group(1) if m else "UNKNOWN"

def summarize_file(file_path):
    try:
        df = pd.read_csv(file_path)

        algo = df["abr_mode"].iloc[0] if "abr_mode" in df.columns else extract_algorithm_from_filename(os.path.basename(file_path))
        client_count = df["client_id"].nunique() if "client_id" in df.columns else np.nan
        total_rows = len(df)

        summary = {
            "file": os.path.basename(file_path),
            "algorithm": algo,
            "client_count": client_count,
            "total_rows": total_rows,
        }

        # métricas fairness: média, desvio padrão e nº de valores válidos
        for col in FAIRNESS_COLS:
            if col in df.columns:
                summary[f"{col}_mean"] = df[col].mean(skipna=True)
                summary[f"{col}_std"] = df[col].std(skipna=True)
                summary[f"{col}_count"] = df[col].count()
            else:
                summary[f"{col}_mean"] = np.nan
                summary[f"{col}_std"] = np.nan
                summary[f"{col}_count"] = 0

        return summary

    except Exception as e:
        return {
            "file": os.path.basename(file_path),
            "algorithm": "ERROR",
            "client_count": "Error",
            "total_rows": "Error",
            "error": str(e)
        }

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Diretório {DATASET_DIR} não existe.")
        return

    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")]
    if not files:
        print(f"⚠️ Nenhum CSV encontrado em {DATASET_DIR}")
        return

    summaries = [summarize_file(os.path.join(DATASET_DIR, f)) for f in files]
    df_summary = pd.DataFrame(summaries)

    # 🔑 Normalizar nome do diretório para compor o nome do arquivo
    clean_name = DATASET_DIR.strip("./").replace("/", "-")

    # Criar subdiretório de relatórios
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    report_file = os.path.join(reports_dir, f"fairness_summary-{clean_name}.csv")

    df_summary.to_csv(report_file, index=False)
    print(f"\n📄 Fairness summary salvo em: {report_file}")
    print(df_summary)

if __name__ == "__main__":
    main()
