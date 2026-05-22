#import
import pandas as pd
from sklearn.linear_model import LinearRegression

#read the data
data = pd.read_csv(r"C:\flutter_projects\INTERNSHIP\class2\homework\updated_power_data.csv")
x= data[["Wind_Speed","Blade_Angle","Rotor_Speed"]]
y=data[["Power_Output"]]

#train
model=LinearRegression()
model.fit(x,y)


print("the regression coefficients are:", model.coef_)
print("the model intercept is:", model.intercept_)

#prediction
print("Predicted Power Output",model.predict([[9,11,100]]))
