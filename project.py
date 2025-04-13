# Put code in here
import pandas as pd 

# Step 1: Load all three CSV files
crash_df = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes_20250330.csv')
weather_df = pd.read_csv('NYC_Weather_2016_2022.csv')
traffic_df = pd.read_csv('Automated_Traffic_Volume_Counts_20250330.csv')

# Step 2: Create ONE timestamp column in the 1st dataset crash_df
# Combine CRASH DATE and CRASH TIME into one column, then convert to datetime format
crash_df['timestamp'] = crash_df['CRASH DATE'] + ' ' + crash_df['CRASH TIME']
crash_df['timestamp'] = pd.to_datetime(crash_df['timestamp'], errors='coerce')
# Keep only rows from the year 2021
crash_df = crash_df[crash_df['timestamp'].dt.year == 2021]
# Round to the nearest hour to help with merging
crash_df['timestamp'] = crash_df['timestamp'].dt.floor('h')

# Step 3: Clean the 2nd dataset weather_df
# Convert 'time' column to datetime and rename it to 'timestamp' for merging
weather_df['timestamp'] = pd.to_datetime(weather_df['time'], errors='coerce')
weather_df = weather_df[weather_df['timestamp'].dt.year == 2021]
weather_df['timestamp'] = weather_df['timestamp'].dt.floor('h')

# Step 4: Clean the 3rd dataset traffic_df
# Build a datetime column from year, month, day, hour columns
traffic_df['timestamp'] = pd.to_datetime(traffic_df[['Yr', 'MM', 'DD', 'HH']], errors='coerce')
traffic_df = traffic_df[traffic_df['timestamp'].dt.year == 2021]

# Step 5: Group traffic data by timestamp and sum the volume
# This gives us total number of vehicles each hour across all locations
traffic_grouped = traffic_df.groupby('timestamp').sum().reset_index()

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
