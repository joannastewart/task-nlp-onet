import pandas as pd
import re

def _tidy_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (out.columns
                   .str.strip()
                   .str.lower()
                   .str.replace(r'[^a-z0-9]+', '_', regex=True))
    return out

def build_analysis_df(df_task_dwa, df_occ, df_dwa_desc, df_task_desc):

    # tidy col names first
    t = _tidy_cols(df_task_dwa)
    occ = _tidy_cols(df_occ)
    dwa = _tidy_cols(df_dwa_desc)
    task = _tidy_cols(df_task_desc)

    # rename occupation columns
    occ = occ.rename(columns={
        'o_net_soc_code': 'soc_code',
        'title': 'occ_title',
        'description': 'occ_description'
    })
    #rename and dedup columns
    t = t.rename(columns={'o_net_soc_code': 'soc_code'})
    occ = occ[['soc_code','occ_title','occ_description']].drop_duplicates('soc_code')
    dwa = dwa[['dwa_id','dwa_title']].drop_duplicates('dwa_id')
    task = task.rename(columns={'task': 'task_text'})[['task_id','task_text']].drop_duplicates('task_id')

    #left join where task to dwa file is base
    out = (t
           .merge(occ, on='soc_code', how='left', validate='many_to_one')
           .merge(dwa, on='dwa_id',   how='left', validate='many_to_one')
           .merge(task, on='task_id', how='left', validate='many_to_one'))

    out = out.drop_duplicates(['task_id','dwa_id'])
    #standardize text in task
    out["task_text"]= (
        out["task_text"].fillna("")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    #create task length column that has word count
    out['task_len'] = out['task_text'].fillna('').str.split().str.len()


    #convert to category (smaller, faster)
    for c in ['soc_code','task_id','dwa_id','occ_title','dwa_title']:
        if c in out.columns:
            out[c] = out[c].astype('category')

    analysis_cols = ['soc_code','occ_title','task_id','task_text',
                     'task_len','dwa_id','dwa_title']
    analysis_df = out[[c for c in analysis_cols if c in out.columns]].copy()

    return analysis_df

def dup_task_text(df: pd.DataFrame):

    grp_text = df.groupby("task_text",observed=True).agg(
        n_rows=("task_id", "size"),
        n_task_ids=("task_id", "nunique"),
        n_dwas=("dwa_id", "nunique"),
        dwas=("dwa_id", lambda s: tuple(sorted(pd.unique(s)))),
    )
    grp_text["pure"] = (grp_text["n_dwas"] == 1)
    return grp_text

#limit one task text per dwa
def dedup_task_text(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dwa_id", "task_text" ]
    rep = (
        df.reset_index(names="row_i")
        .sort_values(keys + ["task_id"])
        .drop_duplicates(keys, keep="first")
    )
    return rep