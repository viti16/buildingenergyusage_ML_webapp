
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from catboost import CatBoostRegressor


st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>Site energy usage prediction</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Enter data for all columns below</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: red;'>31 columns with 6 categorical</h1>", unsafe_allow_html=True)


datacolumns=['State_Factor', 'building_class', 'facility_type', 'Year_Factor',
       'floor_area', 'year_built', 'energy_star_rating', 'ELEVATION',
       'cooling_degree_days', 'heating_degree_days', 'avg_temp', 'id',
       'site_eui', 'summonth_min', 'winmonth_min', 'autmonth_min',
       'sprmonth_min', 'summonth_max', 'winmonth_max', 'autmonth_max',
       'sprmonth_max', 'summonth_avg', 'winmonth_avg', 'autmonth_avg',
       'sprmonth_avg', 'heatingdays', 'coolingdays', 'buildingvolume']


st.markdown('##')
alldata=pd.DataFrame(columns=datacolumns)
alldata.loc[0]=np.NaN

alldata['cooling_degree_days'].loc[0]=1202.250446
alldata['heating_degree_days'].loc[0]=4324.957390
alldata['avg_temp'].loc[0]=56.176705	
alldata['summonth_min'].loc[0]=44.705941
alldata['winmonth_min'].loc[0]=10.882625	
alldata['autmonth_min'].loc[0]=38.125942	
alldata['sprmonth_min'].loc[0]=11.447325	
alldata['summonth_max'].loc[0]=96.070027	
alldata['winmonth_max'].loc[0]=71.510052	
alldata['autmonth_max'].loc[0]=94.473316	
alldata['sprmonth_max'].loc[0]=82.674340	
alldata['summonth_avg'].loc[0]=70.468603	
alldata['winmonth_avg'].loc[0]=41.353057	
alldata['autmonth_avg'].loc[0]=67.891111	
alldata['sprmonth_avg'].loc[0]=44.593664	
alldata['heatingdays'].loc[0]=97.050490
alldata['coolingdays'].loc[0]=71.967739


cols=st.columns(4)
colname=datacolumns[4:8]
for i in range(4):
    alldata[colname[i]].loc[0]=cols[i].number_input(str(colname[i])) 

st.markdown('##')
cols=st.columns(2)
alldata['building_class'].loc[0] = cols[0].radio('building_class', ['Residential','Commercial'])

alldata['Year_Factor'].loc[0] = cols[1].select_slider('Year_Factor',[1,2,3,4,5,6,7])

st.markdown('##')
cols=st.columns(2)
alldata['State_Factor'].loc[0]=cols[0].selectbox('State_Factor', ['State_1','State_2','State_4','State_6','State_8','State_10','State_11'])

alldata['facility_type'].loc[0]=cols[1].selectbox('facility_type',['Grocery_store_or_food_market',
       'Warehouse_Distribution_or_Shipping_center',
       'Retail_Enclosed_mall', 'Education_Other_classroom',
       'Warehouse_Nonrefrigerated', 'Warehouse_Selfstorage',
       'Office_Uncategorized', 'Data_Center', 'Commercial_Other',
       'Mixed_Use_Predominantly_Commercial',
       'Office_Medical_non_diagnostic', 'Education_College_or_university',
       'Industrial', 'Laboratory',
       'Public_Assembly_Entertainment_culture',
       'Retail_Vehicle_dealership_showroom', 'Retail_Uncategorized',
       'Lodging_Hotel', 'Retail_Strip_shopping_mall',
       'Education_Uncategorized', 'Health_Care_Inpatient',
       'Public_Assembly_Drama_theater', 'Public_Assembly_Social_meeting',
       'Religious_worship', 'Mixed_Use_Commercial_and_Residential',
       'Office_Bank_or_other_financial', 'Parking_Garage',
       'Commercial_Unknown', 'Service_Vehicle_service_repair_shop',
       'Service_Drycleaning_or_Laundry', 'Public_Assembly_Recreation',
       'Service_Uncategorized', 'Warehouse_Refrigerated',
       'Food_Service_Uncategorized', 'Health_Care_Uncategorized',
       'Food_Service_Other', 'Public_Assembly_Movie_Theater',
       'Food_Service_Restaurant_or_cafeteria', 'Food_Sales',
       'Public_Assembly_Uncategorized', 'Nursing_Home',
       'Health_Care_Outpatient_Clinic', 'Education_Preschool_or_daycare',
       '5plus_Unit_Building', 'Multifamily_Uncategorized',
       'Lodging_Dormitory_or_fraternity_sorority',
       'Public_Assembly_Library', 'Public_Safety_Uncategorized',
       'Public_Safety_Fire_or_police_station', 'Office_Mixed_use',
       'Public_Assembly_Other', 'Public_Safety_Penitentiary',
       'Health_Care_Outpatient_Uncategorized', 'Lodging_Other',
       'Mixed_Use_Predominantly_Residential', 'Public_Safety_Courthouse',
       'Public_Assembly_Stadium', 'Lodging_Uncategorized',
       '2to4_Unit_Building', 'Warehouse_Uncategorized'])

alldata['buildingvolume']=alldata['ELEVATION']*alldata['floor_area']
alldata['buildingvolume']=alldata['ELEVATION']*alldata['floor_area']

st.markdown('##')
st.markdown('##')
st.markdown("<h1 style='text-align: center;'>Dataframe from entered data</h1>", unsafe_allow_html=True)
cols=st.columns(3)
cols[1].write(alldata[datacolumns[:8]])


st.markdown('#')
st.markdown('#')

categorycols=['State_Factor', 'building_class', 'facility_type']
result=0.0
filename = 'Model/catboostenergypredictormodel.sav'
catbb = CatBoostRegressor(cat_features=categorycols)
catbb.load_model(filename)
result=catbb.predict(alldata)



cols=st.columns(3)
if cols[1].button('Predict EUI'):
#     catbb.load_model(filename)
     result=catbb.predict(alldata)
     cols[1].write(result)
