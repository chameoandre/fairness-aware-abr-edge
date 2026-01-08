import os
import pandas as pd

# === CONFIGURAÇÃO: altere para "./multiprocess" ou "./multithread"

DATASET_DIR = "./multithread/20-clients/trace-lte-driving"
# DATASET_DIR = "./multithread/20-clients/trace-lte-short"
# DATASET_DIR = "./multiprocess/20-clients/trace-lte-driving"
# DATASET_DIR = "./multiprocess/20-clients/trace-lte-short"


# DATASET_DIR = "./multithread"
# DATASET_DIR = "./multiprocess"

def fix_group_client():
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".csv"):
            path = os.path.join(DATASET_DIR, file)
            try:
                df = pd.read_csv(path)

                if "group_client" not in df.columns:
                    print(f"⏭️ Ignorado (sem coluna group_client): {file}")
                    continue

                original_values = df["group_client"].unique().tolist()

                # Decide o grupo com base no nome do arquivo (trace)
                # if "trace-lte-short" in file.lower():
                #     df["group_client"] = df["group_client"].replace({0: "G1", "0": "G1"})
                #     group_target = "G1"
                # elif "trace-lte-driving" in file.lower():
                #     df["group_client"] = df["group_client"].replace({0: "G2", "0": "G2"})
                #     group_target = "G2"
                if "trace-lte-short" in file.lower():
                    df["group_client"] = "G1"
                    group_target = "G1"
                elif "trace-lte-driving" in file.lower():
                    df["group_client"] = "G2"
                    group_target = "G2"
                else:
                    print(f"⚠️ Trace não reconhecida em {file}, valores não alterados.")
                    continue

                df.to_csv(path, index=False)
                print(f"✅ Corrigido {file} | {original_values} → {group_target}")

            except Exception as e:
                print(f"❌ Erro ao processar {file}: {e}")

if __name__ == "__main__":
    fix_group_client()
