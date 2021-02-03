import pickle
import numpy as np
from pydantic import BaseModel
from configs import config
from fastapi import FastAPI
import lightgbm

#Instance of FastAPI class
app = FastAPI()


# Load pickle files for features,encoder and model
enc = pickle.load(open(config.enc_pickle, 'rb'))
features = pickle.load(open(config.feature_pickle, 'rb'))
clf = pickle.load(open(config.mod_pickle, 'rb'))

# Declare Input Data-Structure
class Data(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: float
    average_montly_hours: float
    time_spend_company: float
    Work_accident: float
    promotion_last_5years: float
    sales: str
    salary: str

    class Config:
        schema_extra = {
            "example": {
                "satisfaction_level": 0.38,
                "last_evaluation": 0.53,
                "number_project": 2,
                "average_montly_hours": 157,
                "time_spend_company": 3,
                "Work_accident": 0,
                "promotion_last_5years": 0,
                "sales": "support",
                "salary": "low"
            }
        }
        
        
@app.post("/predict")
def predict_attrition(data: Data):
    
    # Extract data in correct order
    data_dict = data.dict()
    to_predict = [data_dict[feature] for feature in features]
     
    # Apply one-hot encoding
    encoded_features = list(enc.transform(np.array(to_predict[-2:]).reshape(1, -1))[0])
    to_predict = np.array(to_predict[:-2] + encoded_features)
    
    # Create and return prediction
    prediction = clf.predict(to_predict.reshape(1, -1))
    
    return {"prediction": int(prediction[0])}