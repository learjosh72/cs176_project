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

import matplotlib.pyplot as plt


def plot_monthly_traffic_volume_v0():
    # Prepare the data - group traffic data by month and day
    traffic_df['day_of_month'] = traffic_df['timestamp'].dt.day
    traffic_df['month'] = traffic_df['timestamp'].dt.month

    # Group by month and day, then sum the volume
    monthly_daily_traffic = traffic_df.groupby(['month', 'day_of_month'])['Vol'].sum().unstack(level=0)

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Plot each month as a separate line
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for month in months:
        # Get the data for this month
        month_data = monthly_daily_traffic[month]

        # Plot the line
        plt.plot(month_data.index, month_data.values,
                 label=month_names[month - 1],
                 marker='o', markersize=4, linewidth=2)

    # Customize the plot
    plt.title('Daily Traffic Volume by Month (2021)', fontsize=16)
    plt.xlabel('Day of Month', fontsize=14)
    plt.ylabel('Total Traffic Volume', fontsize=14)
    plt.xticks(range(1, 32))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_monthly_traffic_volume():
    # Prepare the data
    traffic_df['day_of_month'] = traffic_df['timestamp'].dt.day
    traffic_df['month'] = traffic_df['timestamp'].dt.month
    traffic_df['date'] = traffic_df['timestamp'].dt.date

    # First calculate daily sums
    daily_traffic = traffic_df.groupby(['month', 'day_of_month', 'date'])['Vol'].sum().reset_index()

    # Calculate monthly medians of daily sums (only considering days <= 70,000)
    monthly_medians = daily_traffic[daily_traffic['Vol'] <= 70000].groupby('month')['Vol'].median()

    # Calculate overall median (for NaN replacement)
    overall_median = daily_traffic[daily_traffic['Vol'] <= 70000]['Vol'].median()

    # Create a dictionary for faster lookup
    month_median_dict = monthly_medians.to_dict()

    # Replace outliers in daily sums
    def replace_outliers(row):
        if row['Vol'] > 70000:
            # Use monthly median if available, otherwise overall median
            return month_median_dict.get(row['month'], overall_median)
        return row['Vol']

    daily_traffic['Vol_filtered'] = daily_traffic.apply(replace_outliers, axis=1)

    # Now group by month and day_of_month using the filtered daily sums
    monthly_daily_traffic = daily_traffic.groupby(['month', 'day_of_month'])['Vol_filtered'].sum().unstack(level=0)

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
    plt.title(
        'Daily Traffic Volume by Month (2021) - Colored by Temperature\n(Daily totals >70,000 replaced with monthly median)',
        fontsize=16)
    plt.xlabel('Day of Month', fontsize=14)
    plt.ylabel('Total Traffic Volume', fontsize=14)
    plt.xticks(range(1, 32))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add temperature colorbar
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

def fatal_acc():
    # Create new dataframe with accidents where (killed + injured) > 5
    severe_accidents_df = merged_df[
        (merged_df['NUMBER OF PERSONS KILLED'] + merged_df['NUMBER OF PERSONS INJURED']) > 5
        ].copy()

    # Extract date from timestamp for merging with daily traffic
    severe_accidents_df['date'] = severe_accidents_df['timestamp'].dt.date

    # Calculate daily traffic volume (sum of all hours in each day)
    daily_traffic = traffic_df.groupby(traffic_df['timestamp'].dt.date)['Vol'].sum().reset_index()
    daily_traffic.columns = ['date', 'daily_traffic_volume']

    # Merge daily traffic volume with severe accidents
    severe_accidents_df = pd.merge(severe_accidents_df, daily_traffic, on='date', how='left')

    # Create combined killed+injured column
    severe_accidents_df['total_casualties'] = (
            severe_accidents_df['NUMBER OF PERSONS KILLED'] +
            severe_accidents_df['NUMBER OF PERSONS INJURED']
    )

    # Select and rename important columns
    columns_to_keep = [
        'COLLISION_ID',
        'timestamp',
        'daily_traffic_volume',
        'weather_condition',
        'temperature_2m (°C)',
        'rain (mm)',
        'windspeed_10m (km/h)',
        'NUMBER OF PERSONS INJURED',
        'NUMBER OF PERSONS KILLED',
        'total_casualties',
        'NUMBER OF PEDESTRIANS INJURED',
        'NUMBER OF PEDESTRIANS KILLED',
        'NUMBER OF CYCLIST INJURED',
        'NUMBER OF CYCLIST KILLED',
        'NUMBER OF MOTORIST INJURED',
        'NUMBER OF MOTORIST KILLED',
        'VEHICLE TYPE CODE 1',
        'VEHICLE TYPE CODE 2'
    ]

    # Create final dataframe
    severe_accidents_final = severe_accidents_df[columns_to_keep].copy()

    # Rename columns for clarity
    severe_accidents_final = severe_accidents_final.rename(columns={
        'timestamp': 'crash_datetime',
        'temperature_2m (°C)': 'temperature_celsius',
        'rain (mm)': 'rainfall_mm',
        'windspeed_10m (km/h)': 'wind_speed_kmh'
    })

    # Add separate date and time columns
    severe_accidents_final['crash_date'] = severe_accidents_final['crash_datetime'].dt.date
    severe_accidents_final['crash_time'] = severe_accidents_final['crash_datetime'].dt.time

    # Reorder columns
    column_order = [
        'COLLISION_ID',
        'crash_datetime',
        'crash_date',
        'crash_time',
        'daily_traffic_volume',
        'weather_condition',
        'temperature_celsius',
        'rainfall_mm',
        'wind_speed_kmh',
        'NUMBER OF PERSONS INJURED',
        'NUMBER OF PERSONS KILLED',
        'total_casualties',
        'NUMBER OF PEDESTRIANS INJURED',
        'NUMBER OF PEDESTRIANS KILLED',
        'NUMBER OF CYCLIST INJURED',
        'NUMBER OF CYCLIST KILLED',
        'NUMBER OF MOTORIST INJURED',
        'NUMBER OF MOTORIST KILLED',
        'VEHICLE TYPE CODE 1',
        'VEHICLE TYPE CODE 2'
    ]
    severe_accidents_final = severe_accidents_final[column_order]
    return severe_accidents_final

def plot_weather_distribution(severe_accidents_df):
    """
    Plot the distribution of weather conditions during severe accidents
    """
    # Count weather occurrences
    weather_counts = severe_accidents_df['weather_condition'].value_counts()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Create bar plot
    bars = plt.bar(weather_counts.index, weather_counts.values,
                   color=['skyblue', 'lightcoral', 'lightgreen'])

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.title('Weather Conditions During Severe Accidents (>5 casualties)')
    plt.xlabel('Weather Condition')
    plt.ylabel('Number of Accidents')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_casualties_vs_weather(severe_accidents_df):
    """
    Plot boxplot of casualties by weather condition using only plt
    """
    # Prepare data for boxplot
    weather_types = severe_accidents_df['weather_condition'].unique()
    data = [severe_accidents_df[severe_accidents_df['weather_condition'] == wt]['total_casualties']
            for wt in weather_types]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Create boxplot using plt
    positions = range(1, len(weather_types) + 1)
    box = plt.boxplot(data, positions=positions, patch_artist=True,
                      labels=weather_types)

    # Color boxes
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Distribution of Casualties by Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Total Casualties (Killed + Injured)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_weather_hourly_pattern(severe_accidents_df):
    """
    Plot hourly distribution of accidents by weather condition using only plt
    """
    # Extract hour from timestamp
    severe_accidents_df['hour'] = severe_accidents_df['crash_datetime'].dt.hour

    # Get unique weather conditions
    weather_types = severe_accidents_df['weather_condition'].unique()

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot line for each weather type
    colors = ['blue', 'red', 'green']
    for wt, color in zip(weather_types, colors):
        hourly_counts = severe_accidents_df[severe_accidents_df['weather_condition'] == wt] \
            .groupby('hour').size()
        plt.plot(hourly_counts.index, hourly_counts.values,
                 label=wt, color=color, marker='o')

    plt.title('Hourly Distribution of Severe Accidents by Weather Condition')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Accidents')
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


# First get your severe accidents data
severe_accidents = fatal_acc()

# Then generate the plots
plot_weather_distribution(severe_accidents)
plot_casualties_vs_weather(severe_accidents)
plot_weather_hourly_pattern(severe_accidents)

# calls each one
barc(merged_df)
line(merged_df)
scatter(merged_df)
histogram(merged_df)
boxplot(merged_df)

plot_monthly_traffic_volume_v0()
plot_monthly_traffic_volume()


def pivot_table(df):
    df['month'] = df['timestamp'].dt.month
    pivot = df.pivot_table(index='month', values='Vol', columns=['weather_condition'], aggfunc='mean')
    pivot = round(pivot, 2)
    print(pivot)
    pivot.to_csv("Pivoted_data.csv", index=False)

pivot_table(merged_df)
