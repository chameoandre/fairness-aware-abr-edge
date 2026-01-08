import os

directory = "./simulations/"
all_files = os.listdir(directory)

print("📂 **Arquivos encontrados no diretório:**")
for idx, file in enumerate(all_files, start=1):
    print(f"{idx}: {file}")