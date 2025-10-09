from src.onet_project.eda import summarize_basics, plot_hist, top_ngrams
import matplotlib.pyplot as plt
import pandas as pd

#read in a summarize previously created parquet file
df = pd.read_parquet("../data/processed/analysis_df.parquet", engine="pyarrow")

summary = summarize_basics(df)
print("Rows:", summary["n_rows"], "| DWAs:", summary["n_dwas"], "| Tasks:", summary["n_tasks"])
n_with_1 = (summary["tasks_per_dwa"] == 1).sum()
n_under_5 = (summary["tasks_per_dwa"] < 5).sum()
n_under_10 = (summary["tasks_per_dwa"] < 10).sum()

print(f"\nDWAs with less than 10 tasks: {int(n_under_10)}")
print(f"DWAs with less than 5 tasks: {int(n_under_5)}")
print(f"DWAs with only 1 task: {int(n_with_1)}")

print("\n# of DWAs per Task:", summary["dwas_per_task_hist"])

#create some plots
fig1 = plot_hist(df["task_len"], "Histogram of Task Word Count")
fig1.savefig("../artifacts/figures/task_len_hist.png", dpi=150)
plt.close(fig1)
fig2 = plot_hist(summary["tasks_per_dwa"], "Histogram of Tasks per DWA")
fig2.savefig("../artifacts/figures/tasks_per_dwa_hist.png", dpi=150)
plt.close(fig2)

x = summary["dwas_per_task_hist"].index.astype(int)
y = summary["dwas_per_task_hist"].values

fig3, ax3 = plt.subplots(figsize=(7,4))
ax3.bar(x, y, width=0.9)
ax3.set_xlabel("# of DWAs per task")
ax3.set_ylabel("# of tasks")
ax3.set_title("Number of DWAs that Each Task Maps To")
ax3.set_xticks(x)
# If counts are very skewed, a log y-scale can help:
# ax2.set_yscale("log")
ax3.grid(axis="y", alpha=0.3)
fig3.tight_layout()
fig3.savefig("../artifacts/figures/dwas_per_task_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig3)

#print some tasks with small word count
short_tasks = df[df["task_len"]<6]["task_text"]
print("\n# of rows where task <6 words",len(short_tasks))
print(short_tasks.sample(10, random_state=6740))

#export a sample
df[["occ_title", "dwa_title", "task_text"]].sample(5, random_state=6740).to_csv("../artifacts/reports/sample_tasks.csv", index=False)

#top ngrams
top_terms = top_ngrams(df=df,topk=7)
top_terms.to_csv("../artifacts/reports/top_ngrams_by_dwa.csv", index=False)