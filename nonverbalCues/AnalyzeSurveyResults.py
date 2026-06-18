import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

constructs = {
    "Immersion": [(11, 21), (12, 22), (13, 23), (14, 24)],
    "Presence": [(15, 25), (16, 26), (17, 27)],
    "Understanding": [(18, 28), (19, 29), (110, 210), (111, 211)],
    "Naturalness": [(112, 212), (113, 213)],
    "Nonverbal": [(114, 214), (115, 215), (116, 216), (117, 217)],
}

yesno_constructs = {
    "VR Experience": ["Q01", "Q02", "Q03"],
    "Full-Body Tracking (FBT) Experience": ["Q04", "Q05", "Q06"],
    "AI NPC Interaction Experience": ["Q07", "Q08", "Q09"]
}

q3_constructs = {
    "Limitation": ["Q31", "Q32", "Q33"],
    "Nonverbal cues awarnes": ["Q34", "Q35", "Q36"],
    "Interaction Comparison": ["Q37", "Q38", "Q39"],
}

survey_answer_df = pd.read_excel("Survey_Answer_Data_Formated.xlsx")

plot_data = []

for construct, pairs in constructs.items():
    for q1, q2 in pairs:
        for val in survey_answer_df[f"Q{q1}"]:
            plot_data.append({
                "Construct": construct,
                "Condition": "Phase 1",
                "Question": f"Q{q1}",
                "Score": val
            })
        for val in survey_answer_df[f"Q{q2}"]:
            plot_data.append({
                "Construct": construct,
                "Condition": "Phase 2",
                "Question": f"Q{q2}",
                "Score": val
            })

plot_df = pd.DataFrame(plot_data)

plt.figure(figsize=(15, 7))
sns.boxplot(
    data=plot_df,
    x="Construct",
    y="Score",
    hue="Condition"
)
plt.title("Comparison of Answers in Phase 1 vs Phase 2 by Theme")
plt.ylabel("Score")
plt.xlabel("Theme")
plt.legend(title="Questions", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.savefig(f"survey_plots/Comparison_of_Answers_in_Phase_1_vs_Phase_2_by_theme.png", format="png", dpi=1200)
plt.show()

plot_df["Pair"] = "Question" + "_" + plot_df["Question"].str[2:]

plt.figure(figsize=(14, 8))
sns.boxplot(
    data=plot_df,
    x="Pair",
    y="Score",
    hue="Condition"
)
plt.xticks(rotation=45)
plt.xlabel("Pair of Answers")
plt.title("Comparison between each answer in both Phases (Phase 1 vs Phase 2)")
plt.legend(title="Questions", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.savefig(f"survey_plots/Comparison_between_each_answer_in_both_Phases.png", format="png", dpi=1200)
plt.show()

for construct in plot_df["Construct"].unique():
    subset = plot_df[plot_df["Construct"] == construct].copy()

    subset["Pair"] = "Question_" + subset["Question"].str[2:]

    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=subset,
        x="Pair",
        y="Score",
        hue="Condition"
    )
    plt.title(f"Experiment Comparison for {construct}")
    plt.xlabel("Answers")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Questions", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"survey_plots/Experiment_Comparison_for_{construct}.png", format="png", dpi=1200)
    plt.show()

for theme, cols in yesno_constructs.items():
    theme_data = []

    for col in cols:
        counts = survey_answer_df[col].value_counts()

        for answer, count in counts.items():
            theme_data.append({
                "Question": col,
                "Answer": answer,
                "Count": count
            })

    theme_df = pd.DataFrame(theme_data)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=theme_df,
        x="Question",
        y="Count",
        hue="Answer"
    )

    plt.title(f"{theme} (Yes/No Responses)")
    plt.ylabel("Count")
    plt.xlabel("Question")
    plt.legend(title="Resonse", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.savefig(f"survey_plots/{theme}_Yes_No_Responses.png", format="png", dpi=1200)
    plt.show()

# Q31-Q39 Likert-style questions (excluding Q310)
q3_cols = [f"Q3{i}" for i in range(1, 10)]
q3_plot_data = []

for theme, cols in q3_constructs.items():
    for col in cols:
        question_num = int(col[2:])

        for score in survey_answer_df[col]:
            q3_plot_data.append({
                "Theme": theme,
                "Question": f"Q3.{question_num}",
                "Score": score
            })

q3_plot_df = pd.DataFrame(q3_plot_data)

for theme in q3_plot_df["Theme"].unique():
    subset = q3_plot_df[q3_plot_df["Theme"] == theme]

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=subset,
        x="Question",
        y="Score"
    )
    sns.stripplot(
        data=subset,
        x="Question",
        y="Score",
        color="black",
        alpha=0.35
    )
    plt.title(f"{theme} Responses")
    plt.xlabel("Question")
    plt.ylabel("Score")
    plt.savefig(f"survey_plots/{theme}_Responses.png", format="png", dpi=1200)
    plt.show()

# Q310 Pie chart
q310_counts = survey_answer_df["Q310"].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(
    q310_counts,
    labels=q310_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Q3.10 Preference (First vs Second Phase)")
plt.savefig(f"survey_plots/Q3_10_Preference_First_vs_Second_Phase.png", format="png", dpi=1200)
plt.show()

# Age
survey_answer_df["Age"].value_counts().sort_index().plot(kind="bar", figsize=(8, 4))
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.savefig(f"survey_plots/Age_Distribution.png", format="png", dpi=1200)
plt.show()

# Gender
survey_answer_df["Gender"].value_counts().plot(kind="bar", figsize=(5, 4))
plt.title("Gender Distribution")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.savefig(f"survey_plots/Gender_Distribution.png", format="png", dpi=1200)
plt.show()
