# Put code in here
import pandas as pd 

# Step 1: Load all three CSV files
crash_df = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes_20250330.csv')
weather_df = pd.read_csv('NYC_Weather_2016_2022.csv')
traffic_df = pd.read_csv('Automated_Traffic_Volume_Counts_20250330.csv')

#-------
# Step 2: Clean crash_df (1st dataset)
crash_df = crash_df.drop_duplicates()
crash_df = crash_df.dropna(subset=['CRASH DATE', 'CRASH TIME'])
crash_df['timestamp'] = crash_df['CRASH DATE'] + ' ' + crash_df['CRASH TIME']
crash_df['timestamp'] = pd.to_datetime(crash_df['timestamp'], errors='coerce')
crash_df = crash_df[crash_df['timestamp'].dt.year == 2021]
crash_df['timestamp'] = crash_df['timestamp'].dt.floor('h')


#-------
# Step 3: Clean weather_df (2nd data set)
weather_df = weather_df.drop_duplicates()
weather_df = weather_df.dropna(subset=['time'])
weather_df['timestamp'] = pd.to_datetime(weather_df['time'], errors='coerce')
weather_df = weather_df[weather_df['timestamp'].dt.year == 2021]            #select only 2021
weather_df['timestamp'] = weather_df['timestamp'].dt.floor('h')

#-------
# Step 4: Clean traffic_df (3rd Dataset)
traffic_df = traffic_df.drop_duplicates()
traffic_df = traffic_df.dropna(subset=['Yr', 'M', 'D', 'HH'])  # Ensure column names match
traffic_df['timestamp'] = pd.to_datetime({
    'year': traffic_df['Yr'],
    'month': traffic_df['M'],
    'day': traffic_df['D'],
    'hour': traffic_df['HH']
}, errors='coerce')

#traffic_df['timestamp'] = pd.to_datetime(traffic_df[['Yr', 'M', 'D', 'HH']], errors='coerce')  # Corrected column names
traffic_df = traffic_df[traffic_df['timestamp'].dt.year == 2021]


#-------
# Step 5: Group traffic data by timestamp and sum the volume
# This gives us total number of vehicles each hour across all locations
traffic_grouped = traffic_df.groupby('timestamp').sum().reset_index()
#traffic_grouped = traffic_df.groupby('timestamp')['Vol'].sum().reset_index()

#-------
# Step 6: Merge the datasets together one by one
# First merge crash and weather
merged_df = pd.merge(crash_df, weather_df, on='timestamp', how='inner')
# Then merge with traffic volume
merged_df = pd.merge(merged_df, traffic_grouped[['timestamp', 'Vol']], on='timestamp', how='inner')

# Step 7: Create weather condition variable
def classify_weather(row):
    if row['rain (mm)'] > 0.5:
        return 'Rainy'
    elif row['windspeed_10m (km/h)'] > 60:
        return 'Windy'
    else:
        return 'Clear'
merged_df['weather_condition'] = merged_df.apply(classify_weather, axis=1)

# Step 8: Save result to CSV file
merged_df.to_csv('merged_final_dataset.csv', index=False)

# Each row in the merged dataset represents one hour in NYC, combining weather, traffic, and accident data for analysis

import matplotlib.pyplot as plt
import numpy as np

# make bar graph for accident frequency per weather type
def barc(df):
    y = {}
    for i,x in df.iterrows():
        if x['weather_condition'] not in y:
            y[x['weather_condition']] = 1
        else:
            y[x['weather_condition']]+=1
    keys = list(y.keys())
    values = list(y.values())
    plt.bar(keys, values)
    plt.title('accidents per weather type')
    plt.xlabel("Weather Type")
    plt.ylabel("Accidents")
    plt.show()

# make line graph of accident frequency per traffic volume
def line(df):

    x = df['Vol'].unique()
    y = dict.fromkeys(list(x), 0)

    for i,k in df.iterrows():
        k['Vol'] += 1

    plt.plot(list(y.keys()),list(y.values()))
    plt.title('Accidents per Traffic Volume')
    plt.xlabel("Traffic Volume")
    plt.ylabel("Accidents")
    plt.show()

# make scatter plot of temperature and frequency of accidents
def scatter(df):
    x = df['temperature_2m (°C)'].unique()
    y = dict.fromkeys(list(x), 0)

    for i, k in df.iterrows():
        k['temperature_2m (°C)'] += 1
    xv = list(y.keys())
    yv = list(y.values())
    plt.scatter(xv,yv)
    plt.title('Accidents as different Temperatures')
    plt.xlabel("Temperature")
    plt.ylabel("Accidents")
    plt.show()

# make histogram of people injured (also returns a total as well in case we want it)
def histogram(df):
    plt.hist(df['NUMBER OF PERSONS INJURED'])
    plt.title('People injured Per Accident')
    plt.show()
    total = df['NUMBER OF PERSONS INJURED'].sum() + df['NUMBER OF PERSONS KILLED'].sum()
    return total

# makes boxplot of different traffic volumes per weather type
def boxplot(df):
    types = list(df['weather_condition'].unique())
    y = dict.fromkeys(types, [])
    for i,x in df.iterrows():
        y[x['weather_condition']].append(x['Vol'])
    plt.boxplot(list(y.values()), labels=list(y.keys()))
    plt.title('Traffic Volume Per Weather Type')
    plt.xlabel("Weather Type")
    plt.ylabel("Traffic Volume")
    plt.show()

# calls each one
barc(merged_df)
line(merged_df)
scatter(merged_df)
histogram(merged_df)
boxplot(merged_df)

