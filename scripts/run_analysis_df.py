from src.onet_project.make_analysis_file import build_analysis_df
import pandas as pd

df_task_dwa = pd.read_csv("../data/raw/Tasks to DWAs.txt", sep="\t", dtype=str, na_filter=False)
df_task_desc = pd.read_csv("../data/raw/Task Statements.txt", sep="\t", dtype=str, na_filter=False)
df_dwa_desc = pd.read_csv("../data/raw/DWA Reference.txt", sep="\t", dtype=str, na_filter=False)
df_occ = pd.read_csv("../data/raw/Occupation Data.txt", sep="\t", dtype=str, na_filter=False)

analysis = build_analysis_df(df_task_dwa, df_occ, df_dwa_desc, df_task_desc)

analysis.to_parquet("../data/processed/analysis_df.parquet", index=False)

print("Rows:", len(analysis), "| DWAs:", analysis['dwa_id'].nunique(), "| Tasks:", analysis['task_id'].nunique())
print(analysis.columns)
print(analysis['task_text'].sample(3))

