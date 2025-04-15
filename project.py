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
    elif row['windspeed_10m (km/h)'] > 30:
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
    wind = weather_df.loc[(weather_df['windspeed_10m (km/h)'] > 30) & (weather_df['timestamp'].dt.year == 2021)]['time'].count().astype(int)
    rain = weather_df.loc[(weather_df['rain (mm)'] > 0.5) & (weather_df['timestamp'].dt.year == 2021)]['time'].count().astype(int)
    clear = weather_df.loc[(weather_df['windspeed_10m (km/h)'] <= 30) & (weather_df['timestamp'].dt.year == 2021) & (weather_df['rain (mm)'] <= 0.5)]['time'].count().astype(int)

    print(wind, rain, clear)
    y = {}
    for i,x in df.iterrows():
        if x['weather_condition'] not in y:
            y[x['weather_condition']] = 1
        else:
            y[x['weather_condition']]+=1
    x1 = y['Rainy']/rain
    x2 = y['Clear']/clear
    x3 = y['Windy']/wind
    keys = list(y.keys())
    data2 = dict.fromkeys(keys)
    data2['Rainy'] = x1
    data2['Clear'] = x2
    data2['Windy'] = x3
    values = list(y.values())
    plt.subplot(1,2,1)
    plt.bar(keys, values)
    plt.title('Accidents Per Weather Type')
    plt.xlabel("Weather Type")
    plt.ylabel("Accidents")

    plt.subplot(1,2,2)
    plt.bar(keys, list(data2.values()))
    plt.title('Proportional Version')
    plt.xlabel("Weather Type")
    plt.ylabel("Accident Per Day of Weather Type")
    plt.tight_layout()
    plt.show()

# make line graph of accident frequency per traffic volume
def line(df):

    x = df['Vol'].unique()
    x.sort()
    y = dict.fromkeys(list(x), 0)

    for i,k in df.iterrows():
        y[k['Vol']] += 1

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
        y[k['temperature_2m (°C)']] += 1
    xv = list(y.keys())
    yv = list(y.values())
    plt.scatter(xv,yv)
    plt.title('Accidents as different Temperatures')
    plt.xlabel("Temperature")
    plt.ylabel("Accidents")
    plt.show()

# make histogram of people injured (also returns a total as well in case we want it)
def histogram(df):

    df['combine'] = df['NUMBER OF PERSONS INJURED'] + df['NUMBER OF PERSONS KILLED']
    high = df.loc[df['combine'] > 5]
    low = df.loc[df['combine'] <= 5]
    plt.subplot(1,2,1)
    plt.hist(low['combine'])
    plt.title('Injured/Killed Per Accident')

    plt.subplot(1,2,2)
    plt.hist(high['combine'])
    plt.title('Injured/Killed Per Accident > 5')
    plt.show()
    total = df['NUMBER OF PERSONS INJURED'].sum() + df['NUMBER OF PERSONS KILLED'].sum()
    return total

# makes boxplot of different traffic volumes per weather type
def boxplot(df):
    types = list(df['weather_condition'].unique())
    y = {}
    for i,x in df.iterrows():
        if x['weather_condition'] not in y:
            y[x['weather_condition']] = []
            y[x['weather_condition']].append(x['Vol'])
        else:
            y[x['weather_condition']].append(x['Vol'])
    plt.boxplot(list(y.values()), labels=list(y.keys()))
    plt.title('Traffic Volume Per Weather Type')
    plt.xlabel("Weather Type")
    plt.ylabel("Traffic Volume")
    plt.show()


def plot_monthly_traffic_volume():
    # Prepare the data
    traffic_df['day_of_month'] = traffic_df['timestamp'].dt.day
    traffic_df['month'] = traffic_df['timestamp'].dt.month

    # Group by month and day
    monthly_daily_traffic = traffic_df.groupby(['month', 'day_of_month'])['Vol'].sum().unstack(level=0)

    # Create the plot
    plt.figure(figsize=(14, 8))

    # New temperature-based colors (Dec→Jan = coldest, Jul→Aug = hottest)
    month_colors = {
        1: '#0a3d6b',  # January (Dark Blue - Coldest)
        2: '#1a5b92',  # February (Deep Blue)
        3: '#3a7cb8',  # March (Medium Blue)
        4: '#5d9bd4',  # April (Light Blue)
        5: '#a5d5f8',  # May (Pale Blue - Cool)
        6: '#ffcc99',  # June (Peach - Warming)
        7: '#ff9966',  # July (Light Orange - Warm)
        8: '#ff3300',  # August (Dark Red - Hottest)
        9: '#ff6600',  # September (Bright Orange - Cooling)
        10: '#ff9933',  # October (Light Orange - Mild)
        11: '#3a7cb8',  # November (Medium Blue - Cold)
        12: '#0a3d6b'  # December (Dark Blue - Coldest)
    }
    # Month names for legend
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Plot each month
    for month in range(1, 13):
        month_data = monthly_daily_traffic[month]
        plt.plot(month_data.index, month_data.values,
                 color=month_colors[month],
                 label=month_names[month - 1],
                 marker='o', markersize=4, linewidth=2)
    # Customize plot
    plt.title('Daily Traffic Volume by Month (2021) - Colored by Temperature', fontsize=16)
    plt.xlabel('Day of Month', fontsize=14)
    plt.ylabel('Total Traffic Volume', fontsize=14)
    plt.xticks(range(1, 32))
    plt.grid(True, linestyle='--', alpha=0.7)
    #plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add temperature colorbar (optional)
    from matplotlib.colors import LinearSegmentedColormap
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list("temp_colors", [month_colors[1], month_colors[8]]),
        norm=plt.Normalize(vmin=1, vmax=12)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Temperature Trend')
    cbar.set_ticks([1, 6, 12])
    cbar.set_ticklabels(['Cold (Dec/Jan)', 'Mild (Jun)', 'Hot (Aug)'])

    plt.tight_layout()
    plt.show()

def plot_monthly_traffic_volume_outliers():
    # Prepare the data
    traffic_df['day_of_month'] = traffic_df['timestamp'].dt.day
    traffic_df['month'] = traffic_df['timestamp'].dt.month

    # Filter outliers (replace values > 70,000 with median of values <= 70,000)
    median_under_70k = traffic_df[traffic_df['Vol'] <= 70000]['Vol'].median()
    traffic_df['Vol_filtered'] = traffic_df['Vol'].where(traffic_df['Vol'] <= 70000, median_under_70k)

    # Group by month and day using the filtered data
    monthly_daily_traffic = traffic_df.groupby(['month', 'day_of_month'])['Vol_filtered'].sum().unstack(level=0)

    # Create the plot
    plt.figure(figsize=(14, 8))

    # New temperature-based colors (Dec→Jan = coldest, Jul→Aug = hottest)
    month_colors = {
        1: '#0a3d6b',  # January (Dark Blue - Coldest)
        2: '#1a5b92',  # February (Deep Blue)
        3: '#3a7cb8',  # March (Medium Blue)
        4: '#5d9bd4',  # April (Light Blue)
        5: '#a5d5f8',  # May (Pale Blue - Cool)
        6: '#ffcc99',  # June (Peach - Warming)
        7: '#ff9966',  # July (Light Orange - Warm)
        8: '#ff3300',  # August (Dark Red - Hottest)
        9: '#ff6600',  # September (Bright Orange - Cooling)
        10: '#ff9933',  # October (Light Orange - Mild)
        11: '#3a7cb8',  # November (Medium Blue - Cold)
        12: '#0a3d6b'  # December (Dark Blue - Coldest)
    }
    # Month names for legend
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Plot each month
    for month in range(1, 13):
        month_data = monthly_daily_traffic[month]
        plt.plot(month_data.index, month_data.values,
                 color=month_colors[month],
                 label=month_names[month - 1],
                 marker='o', markersize=4, linewidth=2)
    # Customize plot
    plt.title('Daily Traffic Volume by Month (2021) - Colored by Temperature\n(Outliers >70,000 replaced with median)', fontsize=16)
    plt.xlabel('Day of Month', fontsize=14)
    plt.ylabel('Total Traffic Volume', fontsize=14)
    plt.xticks(range(1, 32))
    plt.grid(True, linestyle='--', alpha=0.7)
    #plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add temperature colorbar (optional)
    from matplotlib.colors import LinearSegmentedColormap
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list("temp_colors", [month_colors[1], month_colors[8]]),
        norm=plt.Normalize(vmin=1, vmax=12)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Temperature Trend')
    cbar.set_ticks([1, 6, 12])
    cbar.set_ticklabels(['Cold (Dec/Jan)', 'Mild (Jun)', 'Hot (Aug)'])

    plt.tight_layout()
    plt.show()

plot_monthly_traffic_volume_outliers()


# calls each one
#barc(merged_df)
#line(merged_df)
#scatter(merged_df)
#histogram(merged_df)
#boxplot(merged_df)

