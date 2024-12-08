import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from data_prep import df, df_regs, df_factors, df_factors_add
import json
import requests
import seaborn as sns
import numpy as np
import pandas as pd


URL = 'http://127.0.0.1:8000/{}'

st.header("Impact of Remote Work on Mental Health")

st.text("In this report, we'll look at how remote working affects people from different parts of the world.")

st.text("My dataset includes both demographic data (e.g., age, region, experience level) and work-related factors (e.g., hours worked, remote work ratio, company support for remote work). It also includes mental health indicators, such as stress levels, sleep quality, social isolation, and work satisfaction.")

st.subheader("Data cleanup", divider="violet")

st.text("First, I checked that all the data types were correct and checked for empty values. I had empty values in the Mental_Health_Condition and Physical_Activity columns, but the number of these values in both columns exceeded 1000, and I only have 5000 rows. Since I didn't plan to use the data from these columns anywhere, I just deleted these columns.")

st.text("I also had an Employee_ID column where the data was written in this format: EMP0001. I decided to make this column an index column, because I analyze employees. To do this, I had to convert the data from this format to a numeric format and make it an index column.")
st.write(df.head())
st.write(df.dtypes)
st.write(df.isna().sum())

st.subheader("Descriptive statistics", divider="orange")
st.text("To start familiarizing myself with the dataset, I want to output statistics (mean, median and standard deviation of the fields) for three numerical columns.")
st.text("One of the most important and understandable at first glance columns is the Age column, so let's derive these statistics for it.")
st.text(f'Mean: {round(df.Age.mean(), 2)}')
st.text(f'Median: {round(df.Age.median(), 2)}')
st.text(f'Standard deviation: {round(df.Age.std(), 2)}')

st.text("It's also worth analyzing the Years_of_Experience column.")
st.text(f'Mean: {round(df.Years_of_Experience.mean(), 2)}')
st.text(f'Median: {round(df.Years_of_Experience.median(), 2)}')
st.text(f'Standard deviation: {round(df.Years_of_Experience.std(), 2)}')

st.text("The last column I would like to see statistics on is the Hours_Worked_Per_Week column to understand the occupancy of the proposed employees for analysis.")
st.text(f'Mean: {round(df.Hours_Worked_Per_Week.mean(), 2)}')
st.text(f'Median: {round(df.Hours_Worked_Per_Week.median(), 2)}')
st.text(f'Standard deviation: {round(df.Hours_Worked_Per_Week.std(), 2)}')

st.subheader("Simple plots", divider="red")
st.text("For starters, I was interested in how the employees in my dataset were distributed by age, and I also wanted to check that they were roughly evenly distributed so that I didn't predominantly have only one age group to analyze the data.")

# Assuming df is your DataFrame and has a column 'Age'
plt.hist(df.Age)
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.grid(True)

# Use Streamlit to display the matplotlib figure
st.pyplot(plt)

st.text("The histogram shows that the number of people of different age groups was about the same. Only people from about 22 to 26 years old were the most numerous, and people from about 37 to 41 years old were the least numerous. But this difference is not very big.")

st.text("Next, I was interested in learning how people in my dataset know how to find work-life balance.")
# Assuming df is your DataFrame and has a column 'Work_Life_Balance_Rating'
plt.pie(df.Work_Life_Balance_Rating.value_counts(),
        labels=df.Work_Life_Balance_Rating.value_counts().index,
        autopct='%1.1f%%')

# Add legend to the pie chart
plt.legend(title="Work Life Balance Rating:",
           bbox_to_anchor=(1, 0.5), loc="center right",
           fontsize=10, bbox_transform=plt.gcf().transFigure)
plt.title("Work Life Balance")
plt.xlabel("")
plt.ylabel("Number of people in percentage")
# Use Streamlit to display the matplotlib figure
st.pyplot(plt)

st.text("From the pie chart we can see that these employees are again evenly distributed in terms of having a life/work balance. People who manage to do so in an average way are the most numerous. Interestingly, the people who manage 2 out of 5 are the least, but not the people who manage 5 out of 5.")

st.text("In the last simple plot, I wanted to understand how being busy at work during the week affects a person's social isolation.")
# Assuming df is your DataFrame and has the columns 'Social_Isolation_Rating' and 'Hours_Worked_Per_Week'
plt.figure(figsize=(8, 6))  # Optional: Set the figure size (adjust width and height as needed)

# Create the Seaborn boxplot
sns.boxplot(data=df, x="Social_Isolation_Rating", y="Hours_Worked_Per_Week")

# Use Streamlit to display the Seaborn figure
st.pyplot(plt)

st.text("The box plot shows that employees who are most socially isolated by median do not work the most per week. Although those who are socially isolated 4 out of 5 still work the most hours per week according to the median. It is interesting to note that the median weekly hours worked by the people with the least social isolation and the most are the same, and are the smallest in this plot.")

st.subheader("Complex plots (detailed overview)", divider="orange")

# Assuming df is already imported from data_prep
# fig = px.sunburst(...) # Your Plotly figure code

fig = px.sunburst(data_frame=df, path=["Stress_Level", "Sleep_Quality"], color="Work_Life_Balance_Rating", color_continuous_scale="rdbu")
# Display the Plotly chart
st.plotly_chart(fig)

st.text("A sunburst plot, demonstrates hierarchical relationships in data using a circular diagram where each level of hierarchy is represented by a concentric ring. The outer ring shows the distribution in sleep quality across different segments, and in the inner ring each segment is subdivided by stress level . The color gradient (on the “rdbu” scale) adds an additional level of insight by showing how work life balance assessment changes with different combinations of stress level and sleep quality. The graph shows that certain combinations of stress level and sleep quality are more likely to be associated with higher or lower work life balance scores. For example, we see that people with the highest stress levels, but with good sleep quality, have the best work-life balance between comparisons. And employees with average stress levels but good sleep quality show the worst work-life balance.")

st.text("The next plot will require a bit of preparation. I want to show the average of the ratings of the six factors (Stress Level Rating, Satisfaction with Remote Work Rating, Sleep Quality Rating, Company Support for Remote Work, Social Isolation Rating, Work Life Balance Rating) by part of the world.")

st.text("To make this plot we need to make an additional dataset based on our dataset, where the column names are the parts of the world, and the data in the columns are the averages for each factor.")

st.write(df_regs.head())

st.write("Now let's make a set of simple graphs for each part of the world by factors.")
# Set up the figure and axes
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Labels and colors for the bars
labels = ["Stress Level Rating", "Satisfaction with Remote Work Rating", "Sleep Quality Rating", "Company Support for Remote Work", "Social Isolation Rating", "Work Life Balance Rating"]

colors = ['chocolate', 'lightcoral', 'indianred', 'tomato', 'brown', 'darksalmon', 'sandybrown']

x = 0
y = 0

# Loop through regions
for region in df.Region.unique():
    bars = axs[x % 3, y % 2].barh(labels, df_regs[region], color=colors[y], height=0.5)
    axs[x % 3, y % 2].set_title(region)
    axs[x % 3, y % 2].set_xlabel('Rating')
    axs[x % 3, y % 2].set_ylabel('Factor')

    offset = 0.2
    for bar in bars:
        axs[x % 3, y % 2].text(bar.get_width() + 0.05,  # Offset to the right of the bars
                               bar.get_y() + bar.get_height() / 2,  # Center the text vertically
                               f'{bar.get_width():.2f}',  # Value with 2 decimal points
                               va='center', ha='left', fontsize=10)

    axs[x % 3, y % 2].set_xlim(0, max(df_regs[region]) * 1.12)

    y += 1
    if y % 2 != 0:
        x += 1

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot using Streamlit
st.pyplot(fig)

st.write("Now let's combine all the simple plots into one complex plot.")

labels = ["Stress Level Rating", "Satisfaction with Remote Work", "Sleep Quality Rating",
          "Company Support for Remote Work", "Social Isolation Rating",
          "Work Life Balance Rating"]
colors = ['chocolate', 'lightcoral', 'indianred', 'tomato', 'brown', 'darksalmon', 'sandybrown']

regions = df["Region"].unique()  # Get unique regions from your data
x = np.arange(len(labels))  # Positions for the groups
width = 0.8 / len(regions)  # Adjust bar width based on the number of regions

# Create the plot
fig, ax = plt.subplots(figsize=(15, 8))

# Plot bars for each region
for i, region in enumerate(regions):
    region_data = df_regs[region]
    ax.bar(
        x + i * width,
        region_data,
        width,
        label=region,
        color=colors[i % len(colors)]
    )

# Add labels, title, and legend
ax.set_xticks(x + width * (len(regions) - 1) / 2)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_title("Comparison of Ratings Across Regions")
ax.set_xlabel("Factors")
ax.set_ylabel("Average (Mean) Rating")
ax.legend(title="Regions", bbox_to_anchor=(1.05, 1), loc='upper left')

# Add value annotations
for i, region in enumerate(regions):
    region_data = df_regs[region]
    for j, value in enumerate(region_data):
        ax.text(
            x[j] + i * width,
            value + 0.05,
            f"{value:.2f}",
            ha="center", va="bottom", fontsize=8
        )

# Adjust layout
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

st.write("This complex bar graph shows that all factors are higher in South America, Oceania and Africa. It can also be concluded that the best sleeping employees are in South America. And the lowest support for remote work is in Asia. In the same way we can draw conclusions about other factors.")

st.subheader("Data transformation", divider="violet")

st.text("I wanted to see correlations between different factors like stress level, sleep quality, satisfaction and social isolation with remote work to see if there was any relationship between them. To do this, I needed to create three new columns from Stress_Level, Satisfaction_with_Remote_Work and Sleep_Quality to translate the text value of the columns into a numeric value so that I could plot the correlations. So I got three new columns: Stress_Level_Rating, Satisfaction_with_Remote_Work_Rating, Sleep_Quality_Rating.")

st.write(df_factors.head())
st.write(df_factors[["Sleep_Quality_Rating", "Satisfaction_with_Remote_Work_Rating", "Stress_Level_Rating"]].head())


corr_matrix = df_factors[["Sleep_Quality_Rating", "Satisfaction_with_Remote_Work_Rating",
                  "Stress_Level_Rating", "Social_Isolation_Rating"]].corr()


# Set the figure size (optional, adjust width and height as needed)
plt.figure(figsize=(10, 8))

# Create the heatmap with annotations on all cells
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Use Streamlit to display the Seaborn heatmap
st.pyplot(plt)

st.write("This plot says that none of these factors are almost independent of each other.")

st.write("In addition, I became interested in the ratio of virtual meetings to hours worked per week, so I made a separate column with their ratio.")

st.write(df_factors_add.head())
st.write(df_factors_add['Ratio_of_Virtual_Meetings_to_Hours_Worked_Per_Week'].head())

st.subheader("A hypothesis check", divider="orange")

st.text("My hypothesis is that if a person is Asian, has average stress levels and sleeps well, they have a fairly low work-life balance (around two).")

st.text("This hypothesis I derived by looking at both complex graphs from . We will also check it with calculations.")

st.text(f'The average life and work balance rating if you filter the dataset so that the Region column is Asia, the Stress_Level column is Medium, and the Sleep_Quality column is good is {round(df[(df["Region"] == "Asia") & (df["Stress_Level"] == "Medium") & (df["Sleep_Quality"] == "Good")].Work_Life_Balance_Rating.mean(), 2)}.')

st.text('We got around 3, not 2, so we were almost right.')

st.subheader("FastAPI", divider="red")
st.write("Unfortunately the forms only work when you run the application locally on your computer, as you also need to run the back.py file with FastAPI on your computer for it to deploy to the local host.")
st.write("(!Make sure the FastApi host is running for the forms to work!)")
st.write("This is a form with which you can get the average value of some factor by part of the world (so that you don't have to search for values on the plot suggested earlier) using GET method.")
# Display form
with st.form("my_form"):
    st.write("Inside the form")
    part = st.selectbox(
        "choose your part of the world",
        options=df.Region.unique()  # or replace with df['experience_level'].unique().tolist()
    )
    factor = st.selectbox(
        "choose the factor",
        options=labels  # or replace with df['experience_level'].unique().tolist()
    )
    # Add a submit button
    submitted = st.form_submit_button("Submit")

    # Display results after form submission
    # if submitted:
    #     st.write(f"Remote ratio: {remote}%")
    #     st.write(f"Currency ratio: {currency}")
    #     st.write(f"Experience: {experience}")
    #     st.write(f"Company size: {company_size}")

    if submitted:
        st.write("button has been pressed")
        request_data = {
            'region': part,
            'factor': factor,
        }
        response = requests.get(URL.format('part_world'), params=request_data)
        data = json.loads(response.content)

        st.metric(
            label="Mean value", value=f'{data["result"]:.2f}',
        )




# Define your API URL
URL = "http://127.0.0.1:8000/{}"  # Replace with your actual FastAPI endpoint URL

st.text("With this form, you can add your row to the dataset we have already processed.")
# Form for user input
with st.form("post_form"):
    st.write("Inside the form")

    # Collect user input
    job = st.text_input("Job Role")
    industry = st.text_input("Industry")
    age = st.number_input("Age", min_value=0, max_value=200)
    years = st.number_input("Years of Experience", min_value=0, max_value=200)
    hours = st.number_input("Hours Worked Per Week", min_value=1, max_value=168)
    meets = st.number_input("Number of Virtual Meetings", min_value=0, max_value=200)
    gender = st.selectbox("Gender", options=df['Gender'].unique().tolist())
    location = st.selectbox("Work Location", options=df['Work_Location'].unique().tolist())
    product = st.selectbox("Productivity Change", options=df['Productivity_Change'].unique().tolist())
    access = st.selectbox("Access to Mental Health Resources",
                          options=df['Access_to_Mental_Health_Resources'].unique().tolist())
    satisf_work = st.selectbox("Satisfaction with Remote Work",
                               options=df['Satisfaction_with_Remote_Work'].unique().tolist())
    region = st.selectbox("Region", options=df['Region'].unique().tolist())
    balance = st.slider("Work Life Balance", step=1, max_value=7, min_value=1)
    isol = st.slider("Social Isolation", step=1, max_value=5, min_value=1)
    sleep = st.select_slider("Sleep Quality", options=["Poor", "Average", "Good"])
    stress = st.select_slider("Stress Level", options=["Low", "Medium", "High"])
    support = st.select_slider("Company Support for Remote Work", options=[1, 2, 3])

    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write("Button has been pressed")

        # Prepare request data
        request_data = {
            "job": job,
            "industry": industry,
            "age": age,
            "years": years,
            "hours": hours,
            "meets": meets,
            "gender": gender,
            "location": location,
            "product": product,
            "access": access,
            "satisf_work": satisf_work,
            "region": region,
            "balance": balance,
            "isol": isol,
            "sleep": sleep,
            "stress": stress,
            "support": support,
        }

        try:
            # Send POST request to FastAPI
            response = requests.post(URL.format('newrow'), json=request_data)
            response.raise_for_status()  # Raise an error for HTTP issues

            # Handle the response
            data = json.loads(response.content)
            new_df = pd.DataFrame.from_dict(data["received_data"])
            st.write(new_df.tail())
            df_factors_add = new_df.copy()
            st.write("Response received successfully!")


        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
