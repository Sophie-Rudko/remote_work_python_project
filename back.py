from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from data_prep import df_factors, transform_stress, transform_satisfaction, transform_sleep, df_factors_add
#import front
#from front import add_new_row
#from model import predict_salary, reg, enc
#Field, =Field(int)

app = FastAPI(docs_url='/')


@app.put("/echo")
async def root(message: str):
    return {'echo': message}


@app.get('/part_world')
async def get_world_mean(region: str, factor: str): #checking the data type
    ans: float = round(df_factors[df_factors.Region == region]['_'.join(factor.split())].mean(), 2)
    return {
        "result": ans
    }


# Define the data model for the request
class NewRowRequest(BaseModel):
    job: str
    industry: str
    age: int
    years: int
    hours: int
    meets: int
    gender: str
    location: str
    product: str
    access: str
    satisf_work: str
    region: str
    balance: int
    isol: int
    sleep: str
    stress: str
    support: int

# Define the POST method
@app.post("/newrow")
async def new_row(data: NewRowRequest):
    job = str(data.job)
    industry = str(data.industry)
    age = int(data.age)
    years = int(data.years)
    hours = int(data.hours)
    meets = int(data.meets)
    gender = str(data.gender)
    location = str(data.location)
    product = str(data.product)
    access = str(data.access)
    satisf_work = str(data.satisf_work)
    region = str(data.region)
    balance = int(data.balance)
    isol = int(data.isol)
    sleep = str(data.sleep)
    stress = str(data.stress)
    support = int(data.support) # Convert the Pydantic model to a dictionary
    new_row = {
        "Age": age,
        "Gender": gender,
        "Job_Role": job,
        "Industry": industry,
        "Years_of_Experience": years,
        "Work_Location": location,
        "Hours_Worked_Per_Week": hours,
        "Number_of_Virtual_Meetings": meets,
        "Work_Life_Balance_Rating": balance,
        "Stress_Level": stress,
        "Access_to_Mental_Health_Resources": access,
        "Productivity_Change": product,
        "Social_Isolation_Rating": isol,
        "Satisfaction_with_Remote_Work": satisf_work,
        "Company_Support_for_Remote_Work": support,
        "Sleep_Quality": sleep,
        "Region": region,
        "Stress_Level_Rating": transform_stress(stress),
        "Satisfaction_with_Remote_Work_Rating": transform_satisfaction(satisf_work),
        "Sleep_Quality_Rating": transform_sleep(sleep),
        "Ratio_of_Virtual_Meetings_to_Hours_Worked_Per_Week": meets / hours
    }
    # Setting the new Employee_ID for the new row
    new_employee_id = df_factors.index.max() + 1 if not df_factors.empty else 1

    # Adding the new row to the DataFrame
    df_factors_add.loc[new_employee_id] = new_row

    return {
        "received_data": df_factors_add.to_dict(),
    }
