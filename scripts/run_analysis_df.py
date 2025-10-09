from src.onet_project.make_analysis_file import build_analysis_df, dup_task_text, dedup_task_text
import pandas as pd

df_task_dwa = pd.read_csv("../data/raw/Tasks to DWAs.txt", sep="\t", dtype=str, na_filter=False)
df_task_desc = pd.read_csv("../data/raw/Task Statements.txt", sep="\t", dtype=str, na_filter=False)
df_dwa_desc = pd.read_csv("../data/raw/DWA Reference.txt", sep="\t", dtype=str, na_filter=False)
df_occ = pd.read_csv("../data/raw/Occupation Data.txt", sep="\t", dtype=str, na_filter=False)

analysis = build_analysis_df(df_task_dwa, df_occ, df_dwa_desc, df_task_desc)
print("Rows:", len(analysis), "| DWAs:", analysis['dwa_id'].nunique(), "| Tasks:", analysis['task_id'].nunique())

# check that dwa id and dwa title are 1:1
tmp = analysis[['dwa_id','dwa_title']].copy()
id_to_titles = tmp.groupby('dwa_id',observed=True)["dwa_title"].nunique()
ids_with_multi_titles = id_to_titles[id_to_titles > 1]
title_to_ids = tmp.groupby("dwa_title",observed=True)['dwa_id'].nunique()
titles_with_multi_ids = title_to_ids[title_to_ids > 1]
is_one_to_one = ids_with_multi_titles.empty and titles_with_multi_ids.empty
print(f"One-to-one mapping for DWA ID to Title: {is_one_to_one}")

#check out issue where multiple task texts are rolling up to multiple task ids
grouped_task_text = dup_task_text(analysis)
print("How many unique texts are pure (only 1 DWA)?")
print(grouped_task_text["pure"].value_counts())
print("\nShare pure:", grouped_task_text["pure"].mean())
print("\nTexts that map to multiple DWAs (top 5 by frequency):")
print(grouped_task_text.loc[grouped_task_text["n_dwas"] > 1]
                .sort_values(["n_dwas"], ascending=[False])
                .head(5))

analysis_deduped = dedup_task_text(analysis)

analysis_deduped.to_parquet("../data/processed/analysis_df.parquet", index=False)

print("Rows:", len(analysis_deduped), "| DWAs:", analysis_deduped['dwa_id'].nunique(), "| Tasks:", analysis_deduped['task_text'].nunique())
print(analysis_deduped.columns)
print(analysis_deduped['task_text'].sample(3))

#keep in mind that soc_code, occ_title, task_id are incomplete due to removal of dups