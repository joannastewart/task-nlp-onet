import pandas as pd
import matplotlib.pyplot as plt

#load file with distances calculated for each dwa centroid by embedding type
PATH = "../data/processed/analysis_with_distances.parquet"
df = pd.read_parquet(PATH)
#print(df.head())
#print(df.columns)

#create some flags
high_z = 1.5
disagree_z = 1.5
small_dwa = 7
small = df['tasks_per_dwa'] <= small_dwa
df['small']=small
df['flag_tfidf'] = df['z_tfidf']>=high_z
df['flag_sbert'] = df['z_sbert']>=high_z
df['flag_both'] = (df['flag_tfidf'] & df['flag_sbert'])
df['flag_either'] = (df['flag_tfidf'] | df['flag_sbert'])
df['flag_disagree'] = (df['z_tfidf']-df['z_sbert']).abs()>=disagree_z
df['mismatch_either'] = (df['mismatch_tfidf'] | df['mismatch_sbert'])

#basic summary and cross tabs
counts = {
    "n_rows": len(df),
    "tasks_in_small_dwas": int(df["small"].sum()),
    "n_dwa": df["dwa_id"].nunique(),
    "flag_tfidf": int(df["flag_tfidf"].sum()),
    "flag_sbert": int(df["flag_sbert"].sum()),
    "flag_both": int(df["flag_both"].sum()),
    "flag_either": int(df["flag_either"].sum()),
    "flag_disagree": int(df["flag_disagree"].sum()),
    "mismatch_sbert":int(df["mismatch_sbert"].sum()),
    "mismatch_tfidf":int(df["mismatch_tfidf"].sum()),
    "mismatch_either":int(df["mismatch_either"].sum())
}
print(pd.Series(counts))

print("\nflag cross-tab (TF-IDF vs SBERT):")
print(pd.crosstab(df["flag_tfidf"], df["flag_sbert"], margins=True))

print("\nflag cross-tab excluding small DWAs (TF-IDF vs SBERT):")
print(pd.crosstab(df.loc[~small, "flag_tfidf"],df.loc[~small, "flag_sbert"],margins=True))

print("\nMismatch Top DWA (TF-IDF vs SBERT):")
print(pd.crosstab(df["mismatch_tfidf"], df["mismatch_sbert"], margins=True))

dwa_summary = (
    df.groupby("dwa_id",observed=True)
      .agg(
          dwa_title=("dwa_title","first"),
          n=("task_id","size"),
          small_flag=("small","sum"),
          tfidf_flag=("flag_tfidf","sum"),
          sbert_flag=("flag_sbert","sum"),
          either_flag = ("flag_either","sum"),
          both_flag=("flag_both","sum"),
          disagree=("flag_disagree","sum"),
          mean_dist_tfidf=("dist_tfidf","mean"),
          mean_dist_sbert=("dist_sbert","mean"),
          p95_tfidf=("dist_tfidf", lambda s: s.quantile(0.95)),
          p95_sbert=("dist_sbert", lambda s: s.quantile(0.95)),
          mismatch_tfidf=("mismatch_tfidf","sum"),
          mismatch_sbert=("mismatch_sbert","sum"),
          mismatch_either=("mismatch_either","sum")
      )
      .assign(
          rate_tfidf=lambda d: d.tfidf_flag / d.n,
          rate_sbert=lambda d: d.sbert_flag / d.n,
          rate_both=lambda d: d.both_flag / d.n,
          rate_either = lambda d: d.either_flag / d.n,
          rate_disagree=lambda d: d.disagree / d.n,
          rate_mismatch_tfidf=lambda d: d.mismatch_tfidf / d.n,
          rate_mismatch_sbert=lambda d: d.mismatch_sbert / d.n,
          rate_mismatch_either = lambda d: d.mismatch_either / d.n,
      )
      .sort_values("rate_either", ascending=False)
)

dwa_summary.to_csv("../artifacts/reports/dwa_distance_summary.csv")

#get some sample outliers
cols = ["dwa_id","dwa_title","tasks_per_dwa","occ_title","task_id","task_text","dist_sbert","z_sbert","dist_tfidf","z_tfidf","closest_dwa_tfidf","closest_dwatitle_tfidf","closest_dwa_sbert","closest_dwatitle_sbert"]

#outliers for either model; output all the tasks not just the outlier
flagged_dwas = df.loc[df["flag_either"], "dwa_id"].unique()
out = (
        df[df["dwa_id"].isin(flagged_dwas)]
          .sort_values(["dwa_id","flag_both","flag_either","z_sbert","z_tfidf"], ascending=[True, False, False, False, False])
          [cols + ["flag_tfidf","flag_sbert","flag_either","flag_both","mismatch_tfidf","mismatch_sbert","mismatch_either"]]
    )
out.to_csv("../artifacts/reports/outliers_distance.csv", index=False)

#histogram of distance for tfidf and sbert at the dwa level
#all_means = np.concatenate([dwa_summary["mean_dist_tfidf"].values,dwa_summary["mean_dist_sbert"].values])
bins = 30
plt.figure(figsize=(7, 4))
plt.hist(dwa_summary["mean_dist_tfidf"],bins=bins,alpha=0.5,label="TF-IDF mean distance")

plt.hist(dwa_summary["mean_dist_sbert"],bins=bins,alpha=0.5,label="SBERT mean distance",)

plt.xlabel("Mean task–DWA centroid distance")
plt.ylabel("Number of DWAs")
plt.title("Distribution of mean distances by embedding type")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../artifacts/figures/dwa_mean_distance_hist_tfidf_vs_sbert.png", dpi=300)
plt.close()