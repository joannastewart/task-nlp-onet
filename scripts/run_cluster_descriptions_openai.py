from openai import OpenAI
import pandas as pd

in_path = "../data/processed/analysis_task_level_with_clusters.parquet"
out_path = "../artifacts/reports/task_cluster_descriptions.csv"
out2 ="../data/processed/analysis_task_level_with_clusters_descriptions.parquet"
out3 = "../artifacts/reports/tasks_with_cluster_descriptions.csv"

#note the openai API key is saved as environment variable for security purposes
client = OpenAI()

#models = client.models.list()
#for m in models.data:
#    print(m.id)

#load
df = pd.read_parquet(in_path)

#group by cluster, limit to certain number of texts randomly, text as a list
max_tasks = 20 #limit inputs
random_state = 6740
grouped = (df.groupby("task_cluster_id")["task_text"].apply(lambda s: s.sample(n=min(max_tasks, len(s)),random_state=random_state).tolist()))

#first come up with the system message to kick off the session; pulled from ONET documenation
SYSTEM_MESSAGE = """
You are an analyst with expertise in occupations.
You work for O*NET with the Department of Labor. 
Your job is to write standardized “Detailed Work Activity”-style summaries. 
The summary is based on sample task statements provided.

Follow these rules:
- Begin with a present-tense action verb (e.g., Fit, Install, Devise).
- Use only one action verb per summary.
- The summary should be clear about the activity. You should be able to visualize the task. “Harvest vegetables” is clear; “Grow things” is not.
- Do NOT include a subject (no “Workers" or "They").
- Use clear, general language describing the main work action and its object.
- The summary will be less specific than individual task statements.
- Keep it to ONE sentence, about 10–20 words.
- Use as few nouns as possible while still being specific.“Equipment” is too general, “laser guided drones” is probably too precise.
- Avoid adverbs, examples in parentheses, or multiple action verbs joined with “and” unless they are tightly related.
- Do not mention specific job titles or occupations.
"""
#user prompt will give the specific tasks per cluster
USER_PROMPT_TEMPLATE = """
You are given example work tasks that all belong to the same semantic cluster.
Write ONE O*NET-style Detailed Work Activity-style summary that best represents this cluster.
Apply the rules given in the system message.

Example tasks:
{examples}

Now write the single summary statement:
"""

records=[]
for cluster_id, sample_tasks in grouped.items():
    examples_str = "\n".join(f"- {t}" for t in sample_tasks)
    user_content = USER_PROMPT_TEMPLATE.format(examples=examples_str)
    print(f"Generating summary for cluster {cluster_id} (using {len(sample_tasks)} tasks)...")

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": SYSTEM_MESSAGE},{"role": "user", "content": user_content}],
        temperature=0.2,
    )

    summary = resp.choices[0].message.content.strip()
    print(f"  → {summary}")

    records.append({"task_cluster_id": cluster_id,"cluster_summary": summary})

summaries_df = pd.DataFrame(records).sort_values("task_cluster_id")
summaries_df.to_csv(out_path, index=False)

#merge back into original df
task_with_summaries = df.merge(
    summaries_df,
    on="task_cluster_id",
    how="left"
)
task_with_summaries.to_parquet(out2, index=False)
task_with_summaries.to_csv(out3, index=False)
