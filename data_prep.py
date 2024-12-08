import pandas as pd
import numpy as np

df = pd.read_csv('Impact_of_Remote_Work_on_Mental_Health.csv')
#df.dropna(inplace=True)
del df['Physical_Activity']
del df['Mental_Health_Condition']
df.Employee_ID = df.Employee_ID.apply(lambda x: int(x[-4:]))
df.set_index('Employee_ID', inplace=True)

df_factors = df.copy()

def transform_stress(x):
    if x == "Low":
        return 1
    if x == "Medium":
        return 2
    if x == "High":
        return 3

df_factors["Stress_Level_Rating"] = df.Stress_Level.apply(transform_stress)

def transform_satisfaction(x):
    if x == "Unsatisfied":
        return 1
    if x == "Neutral":
        return 2
    if x == "Satisfied":
        return 3
df_factors["Satisfaction_with_Remote_Work_Rating"] = df.Satisfaction_with_Remote_Work.apply(transform_satisfaction)

def transform_sleep(x):
    if x == "Poor":
        return 1
    if x == "Average":
        return 2
    if x == "Good":
        return 3
df_factors["Sleep_Quality_Rating"] = df.Sleep_Quality.apply(transform_sleep)

ratings_for_regions = dict()
for region in df_factors.Region.unique():
    stress = round(df_factors[df_factors.Region == region].Stress_Level_Rating.mean(), 2)
    satisfaction = round(df_factors[df_factors.Region == region].Satisfaction_with_Remote_Work_Rating.mean(), 2)
    sleep = round(df_factors[df_factors.Region == region].Sleep_Quality_Rating.mean(), 2)
    support = round(df_factors[df_factors.Region == region].Company_Support_for_Remote_Work.mean(), 2)
    isolation = round(df_factors[df_factors.Region == region].Social_Isolation_Rating.mean(), 2)
    balance = round(df_factors[df_factors.Region == region].Work_Life_Balance_Rating.mean(), 2)
    ratings_for_regions[region] = [stress, satisfaction, sleep, support, isolation, balance]

df_regs = pd.DataFrame(np.array(list(ratings_for_regions.values())), columns=df.Region.unique())

df_factors_add = df_factors.copy()
df_factors_add['Ratio_of_Virtual_Meetings_to_Hours_Worked_Per_Week'] = df_factors.Number_of_Virtual_Meetings / df_factors.Hours_Worked_Per_Week

