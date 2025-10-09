import pandas as pd

#load file with distances calculated for each dwa centroid by embedding type
PATH = "../data/processed/analysis_with_distances.parquet"
df = pd.read_parquet(PATH)
#print(df.head())
#print(df.columns)

#create some flags
high_z = 2.5
disagree_z = 1.5
small_dwa = 7
high_perc = 0.95 #may work better for really small clusters, just looks at furthest point
small = df['tasks_per_dwa'] <= small_dwa
df['small']=small
df['flag_tfidf'] = df['z_tfidf']>=high_z
df['flag_sbert'] = df['z_sbert']>=high_z
#df.loc[small, "flag_tfidf"] = df.loc[small, "pct_tfidf"].ge(high_perc) #decided not to flag furthest task in small dwa
#df.loc[small, "flag_sbert"] = df.loc[small, "pct_sbert"].ge(high_perc)
df['flag_both'] = (df['flag_tfidf'] & df['flag_sbert'])
df['flag_disagree'] = (df['z_tfidf']-df['z_sbert']).abs()>=disagree_z

#basic summary and cross tabs
counts = {
    "n_rows": len(df),
    "tasks_in_small_dwas": int(df["small"].sum()),
    "n_dwa": df["dwa_id"].nunique(),
    "flag_tfidf": int(df["flag_tfidf"].sum()),
    "flag_sbert": int(df["flag_sbert"].sum()),
    "flag_both": int(df["flag_both"].sum()),
    "flag_disagree": int(df["flag_disagree"].sum()),
}
print(pd.Series(counts))

print("\nflag cross-tab (TF-IDF vs SBERT):")
print(pd.crosstab(df["flag_tfidf"], df["flag_sbert"], margins=True))

print("\nflag cross-tab excluding small DWAs (TF-IDF vs SBERT):")
print(pd.crosstab(df.loc[~small, "flag_tfidf"],df.loc[~small, "flag_sbert"],margins=True))

dwa_summary = (
    df.groupby("dwa_id",observed=True)
      .agg(
          n=("task_id","size"),
          small_flag=("small","sum"),
          tfidf_flag=("flag_tfidf","sum"),
          sbert_flag=("flag_sbert","sum"),
          both_flag=("flag_both","sum"),
          disagree=("flag_disagree","sum"),
          mean_dist_tfidf=("dist_tfidf","mean"),
          mean_dist_sbert=("dist_sbert","mean"),
          p95_tfidf=("dist_tfidf", lambda s: s.quantile(0.95)),
          p95_sbert=("dist_sbert", lambda s: s.quantile(0.95)),
      )
      .assign(
          rate_tfidf=lambda d: d.tfidf_flag / d.n,
          rate_sbert=lambda d: d.sbert_flag / d.n,
          rate_both=lambda d: d.both_flag / d.n,
          rate_disagree=lambda d: d.disagree / d.n,
      )
      .sort_values("rate_both", ascending=False)
)

dwa_summary.to_csv("../artifacts/reports/dwa_distance_summary.csv")

#get some sample outliers
cols = ["task_id","dwa_id","dwa_title","tasks_per_dwa","task_text","dist_sbert","z_sbert","dist_tfidf","z_tfidf"]

#top k when both models agree; output all the tasks not just the outlier
flagged_dwas = df.loc[df["flag_both"], "dwa_id"].unique()
out = (
        df[df["dwa_id"].isin(flagged_dwas)]
          .sort_values(["dwa_id","flag_both","z_sbert","z_tfidf"], ascending=[True, False, False, False])
          [cols + ["flag_tfidf","flag_sbert","flag_both"]]
    )
out.to_csv("../artifacts/reports/outliers_both.csv", index=False)

#why are there multiple task ids per task text???