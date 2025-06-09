import sqlite3
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import timedelta
import scipy.stats
from scipy import interpolate
from scipy import stats
import metpy
from metpy.units import units
from scipy.interpolate import CubicSpline
import sys
warnings.filterwarnings("ignore")

# ======================== Part I: Read data and transfer into the dataframe required ========================
# Transfer data into proper format data list
def preprocess_weatherdata(data):

    lines = data.splitlines()
    lines = lines[1:]  # 

    data_values = " ".join(lines).split()

    data_values = list(map(float, data_values))

    return(data_values)


def turn_into_daily(values,needtype):
    step = 24
    data = [values[i:i+step] for i in range(0,len(values),step)]
    data_daily = []
    for lis in data:
        if needtype == "mean":
            data_daily.append(np.mean(lis))
        elif needtype == "max":
            data_daily.append(np.max(lis))
        elif needtype == "min":
            data_daily.append(np.min(lis))
    return(data_daily)

# W/m2 into MJ/m2
def turn_into_daily_radi(values):
    step = 24
    data = [values[i:i+step] for i in range(0,len(values),step)]
    data_daily = []
    for lis in data:
        data_daily.append(np.sum(lis)*60*60/1000000) #
    return(data_daily)


def calculate_Vapor_pressure(temp, AH):

    temperature = temp * units.degC  # 
    absolute_humidity = AH * units.g / units.kg  # 

    R = 461.5  # 

    temperature_K = temperature.to(units.K)  # 

    vapor_pressure = (absolute_humidity.magnitude / 1000) * R * temperature_K.magnitude
    return(vapor_pressure)


def parse_years_from_ID(start_ID, end_ID):
    if start_ID > 10000 and end_ID > 10000:

        start_year = 2000 + int(str(start_ID)[-3:])
        end_year = 2000 + int(str(end_ID)[-3:])
    else:

        start_year = start_ID
        end_year = end_ID
    return start_year, end_year



def Generate_monthly_data(data, monthnum):
    total_data = pd.DataFrame(data)

    monthly_data = {}

    monthly_stats = {}
    start_day = 0  # 
    for month, days in monthnum.items():
        end_day = start_day + days
        month_data = total_data.iloc[start_day:end_day, :]  # 
        monthly_data[month] = month_data
        start_day = end_day  # 

    for month,data in monthly_data.items():
        mean_all = data.values.flatten().mean()
        std_all = data.values.flatten().std()
        monthly_stats[month] = {
            'mean': mean_all,
            'std': std_all
        }
    monthly_std_data = {}
    for month, data in monthly_data.items():
        mean_value = monthly_stats[month]['mean']
        std_value = monthly_stats[month]['std']
        print(mean_value,std_value)
        standardized_data = (data - mean_value) / std_value
        
        monthly_std_data[month] = standardized_data
    
    return monthly_data, monthly_stats, monthly_std_data

def extract_weather_data(df_bj):
    # DBT
    temp_total_list = []
    for data in df_bj["t"].values:
        temp_list = preprocess_weatherdata(data)
        temp_total_list = temp_total_list + temp_list
    # RH
    humid_total_list = []
    for data in df_bj["d"].values:
        humid_list = preprocess_weatherdata(data)
        humid_total_list = humid_total_list + humid_list

    # GHI  & DHI
    rtotal_total_list = []
    for data in df_bj["r_total"].values:
        rtotal_list = preprocess_weatherdata(data)
        rtotal_total_list = rtotal_total_list + rtotal_list
    rscatter_total_list = []
    for data in df_bj["r_scatter"].values:
        rscatter_list = preprocess_weatherdata(data)
        rscatter_total_list = rscatter_total_list + rscatter_list
        
    # WS
    ws_total_list = []
    for data in df_bj["ws"].values:
        ws_list = preprocess_weatherdata(data)
        ws_total_list = ws_total_list + ws_list
        
    # TSKY & GST
    tsky_total_list = []
    for data in df_bj["t_sky"].values:
        tsky_list = preprocess_weatherdata(data)
        tsky_total_list = tsky_total_list + tsky_list
    tground_total_list = []
    for data in df_bj["t_ground"].values:
        tground_list = preprocess_weatherdata(data)
        tground_total_list = tground_total_list + tground_list

    # WD
    wd_total_list = []
    for data in df_bj["wd"].values:
        wd_list = preprocess_weatherdata(data)
        wd_total_list = wd_total_list + wd_list

    # Pressure
    b_total_list = []
    for data in df_bj["b"].values:
        b_list = preprocess_weatherdata(data)
        b_total_list = b_total_list + b_list
    
    df_bj_final = pd.DataFrame({"t":temp_total_list, "d": humid_total_list, "r_total": rtotal_total_list, 
                                "r_scatter": rscatter_total_list, "t_sky": tsky_total_list, "t_ground": tground_total_list, 
                                "ws":ws_total_list, "wd": wd_total_list, "b": b_total_list })
    # VP
    df_bj_final["Vapro_pre"] = df_bj_final.apply(lambda x: calculate_Vapor_pressure(x["t"], x["d"]), axis = 1)
    return(df_bj_final)


# ======================== 第二节：准备CDF以及分层抽样函数，基于日值挑选RMY ========================
def compute_cdf(data):
    """
    CDF function
    
    input:
    - data: DataFrame (365 x 20)
    
    return:
    - sorted_data: 1d-value
    - cdf
    """
    flattened_data = data.values.flatten()  # 
    sorted_data = np.sort(flattened_data)  # 
    cdf = np.linspace(0, 1, len(sorted_data))  # 
    return sorted_data, cdf

def find_closest_values(cdf, sorted_data, probabilities, min_stats=None, max_stats = None):

    first_idx = 0
    last_idx = len(cdf) - 1
    probabilities_now = probabilities[1:-1]
    selected_indices = [first_idx] + list(np.searchsorted(cdf, probabilities_now, side='right')) + [last_idx]
    selected_values = sorted_data[selected_indices]  # 

    if min_stats is not None:
        min_index = np.abs(sorted_data - min_stats).argmin()
        min_cdf_value = cdf[min_index]  # 

        prob_index = np.abs(probabilities - min_cdf_value).argmin()

        selected_values[prob_index] = min_stats
    
    if max_stats is not None:
        max_index = np.abs(sorted_data - max_stats).argmin()
        max_cdf_value = cdf[max_index]  # 
        prob_index = np.abs(probabilities - max_cdf_value).argmin()
        original_stats = selected_values[prob_index]
        selected_values[prob_index] = max_stats
        print(f"Replace max stats, originally selected stats is {original_stats}, now is replaced with {max_stats}")

    return selected_values

def select_representative_data(cdf_values, sorted_data, probabilities, total_days=31, min_stats=None):
    """
    :param cdf_values: 
    :param sorted_data: 
    :param total_days: 
    :return: 
    """

    if len(probabilities) > 2:
        first_idx = 0
        last_idx = len(cdf_values) - 1
        probabilities_now = probabilities[1:-1]
        selected_indices = [first_idx] + list(np.searchsorted(cdf_values, probabilities_now, side='right')) + [last_idx]

    elif len(probabilities) == 2:
        probabilities_now = np.linspace(0.25, 0.75, 2)
        selected_indices = list(np.searchsorted(cdf_values, probabilities_now, side='right'))

    elif len(probabilities) == 1:
        mid_idx = np.searchsorted(cdf_values, 0.5, side='right')
        print("Current day is only 1 day and the selected value is: ", sorted_data[mid_idx])
        return (sorted_data[mid_idx], np.repeat(sorted_data[mid_idx], total_days))

    selected_indices = np.clip(selected_indices, 0, len(sorted_data) - 1)
    selected_data = sorted_data[selected_indices]

    if min_stats is not None and len(probabilities) > 2:
        selected_data[0] = min_stats

    selected_cdf_values = cdf_values[selected_indices]
    cdf_intervals = np.diff(np.insert(selected_cdf_values, 0, 0))

    counts = np.round(cdf_intervals * (total_days - 2)).astype(int)
    counts[0] = 1
    counts[-1] = 1
    print("Occurrence times before adjustment:", counts)

    diff = total_days - np.sum(counts)

    if len(probabilities) == 3:
        counts[1] += diff
        diff = 0 

    elif len(probabilities) == 2:
        counts[0] += diff // 2
        counts[1] += diff - (diff // 2)
        diff = 0  

    elif len(probabilities) > 3:
        while diff != 0:
            mid = (len(counts) - 1) // 2  
            left, right = mid, mid + 1 if len(counts) % 2 == 0 else mid

            if diff > 0:
                while diff > 0 and left >= 1 and right <= len(counts) - 2:
                    counts[left] += 1
                    diff -= 1
                    if diff > 0:
                        counts[right] += 1
                        diff -= 1
                    left -= 1
                    right += 1

            elif diff < 0:
                while diff < 0 and left >= 1 and right <= len(counts) - 2:
                    if counts[left] > 1:
                        counts[left] -= 1
                        diff += 1
                    if diff < 0 and counts[right] > 1:
                        counts[right] -= 1
                        diff += 1
                    left -= 1
                    right += 1

    print("Occurrence times after adjustment:", counts)

    final_data = np.repeat(selected_data, counts)
    return selected_data, final_data

def find_original_indices(data, selected_values, start_year=2031):
    """
    - data: DataFrame (365 x 20)
    - selected_values: 
    - start_year: 
    
    return:
    - indices: (day, year) 
    """
    indices = []
    data_array = data.values  # 

    for val in selected_values:
        diff = np.abs(data_array - val)
        row, col = np.unravel_index(np.argmin(diff), data_array.shape)

        day = row + 1  # 
        year = start_year + col  # 
        indices.append((day, year))

    return indices


def reorder_tmy(df_tmy, smooth = False):
    df_tmy['year_month_day'] = df_tmy["year"].astype(str).str.zfill(4) + "-" + df_tmy['month'].astype(str).str.zfill(2) + '-' + df_tmy['day'].astype(str).str.zfill(2)

    daily_avg_temp = df_tmy.groupby('year_month_day')['t'].mean().reset_index()
    daily_avg_temp.columns = ['year_month_day', 'avg_temp']


    df_tmy = pd.merge(df_tmy, daily_avg_temp, on='year_month_day', how='left')
    df_tmy['month_day'] = df_tmy['month'].astype(str).str.zfill(2) + '-' + df_tmy['day'].astype(str).str.zfill(2)

    df_sorted = df_tmy.sort_values(by=['month_day', 'avg_temp', 'hour']).reset_index(drop=True)
    

    total_hours = len(df_sorted)
    if smooth is True:
        for i in range(0, total_hours - 24, 24):  # 
            transition_points = df_sorted.iloc[i:i + 48]

            y_known = np.concatenate([transition_points['t'].values[21:23], transition_points['t'].values[25:27]])
            x_known = np.array([0, 1, 4, 5])  # 

            # cubic spline
            x = np.linspace(0, 5, 6)  # 
            cubic_spline = CubicSpline(x_known, y_known)
            y_new = cubic_spline(x)

            df_sorted.loc[i + 23, 't'] = y_new[2]  # 
            df_sorted.loc[i + 24, 't'] = y_new[3]  # 
    else:
        pass
    return df_sorted


def reorder_tmy_new(df):
    df_other = df[df["month"] < 8]

    # 
    df_reversed_list = []
    for month in range(8, 13):  # 
        df_now = df[df["month"] == month].copy()
        print(f"Processing month {month}, total rows: {len(df_now)}")
        def reverse_24h_blocks(df):
            """
            按照 24 小时为一个块倒序排列
            """
            num_rows = len(df)
            reversed_df = pd.concat([df.iloc[i:i+24] for i in range(num_rows - 24, -1, -24)], ignore_index=True)
            return reversed_df
        df_now = reverse_24h_blocks(df_now)
        df_reversed_list.append(df_now)

    df_reversed = pd.concat(df_reversed_list, axis=0).reset_index(drop=True)
    df_final = pd.concat([df_other, df_reversed]).reset_index(drop=True)

    return df_final


# ======================== Third Part: Main  ========================
def main():
    # Read cmd
    print(sys.argv)
    if len(sys.argv) != 6:
        print("Usage: python your_script.py <your climate database. db> <cityname> <start_year> <end_year> <RMY_save_path.xlsx>")
        print("Cityname & start_year & end_year should be in the climate database.")
        sys.exit(1)
    
    db_path = sys.argv[1]
    cityname = sys.argv[2]
    start_ID = int(sys.argv[3])
    end_ID = int(sys.argv[4])
    # Target save file path
    output_excel_savename = sys.argv[4]
    
    # db_path = "ClimateData-fivecities-6.db"
    # db_path = "Typical_cities_AMY.db"
    conn = sqlite3.connect(db_path)
    # Query
    query = "SELECT * FROM station_lib WHERE city = ?;"
    # print(query)
    df_city = pd.read_sql_query(query, conn, params=(cityname,))
    # ERA5 database
    df_city = df_city[(df_city["type"] >= start_ID) & (df_city["type"] <= end_ID)].reset_index(drop=True)
    # Close connection
    conn.close()
    df_weather_data_final = extract_weather_data(df_city)
    # Year_list = turn_into_daily(Data_Feature["Year"].values, "mean")
    print(df_weather_data_final.head())
    
    # Daily statistics
    AVE_temp_daily = turn_into_daily(df_weather_data_final["t"].values, "mean")
    Min_temp_daily = turn_into_daily(df_weather_data_final["t"].values, "min")
    Max_temp_daily = turn_into_daily(df_weather_data_final["t"].values,"max")
    AVE_VP_daily = turn_into_daily(df_weather_data_final["Vapro_pre"].values, "mean")
    Sum_Radi_daily = turn_into_daily_radi(df_weather_data_final["r_total"].values)
    AVE_GST_daily = turn_into_daily(df_weather_data_final["t_ground"].values, "mean")
    AVE_WS_daily = turn_into_daily(df_weather_data_final["ws"].values, "mean")
    n_days = 365
    total_years = int(len(AVE_temp_daily) / n_days)
    # Daily avg DBT
    AVE_temp_daily = np.array(AVE_temp_daily).reshape(n_days,total_years,order='F') #把数据变形一下
    temp_data = pd.DataFrame(AVE_temp_daily)
    # Daily max DBT
    Max_temp_daily = np.array(Max_temp_daily).reshape(n_days,total_years,order='F') #把数据变形一下
    temp_max_data = pd.DataFrame(Max_temp_daily)
    # Daily min DBT
    Min_temp_daily = np.array(Min_temp_daily).reshape(n_days,total_years,order='F') #把数据变形一下
    temp_min_data = pd.DataFrame(Min_temp_daily)
    # Daily VP
    AVE_VP_daily = np.array(AVE_VP_daily).reshape(n_days,total_years, order = 'F') #把数据变形一下
    VP_daily_data = pd.DataFrame(AVE_VP_daily)
    # Daily GHI
    Sum_Radi_daily = np.array(Sum_Radi_daily).reshape(n_days,total_years, order = 'F') #把数据变形一下
    Radi_daily_data = pd.DataFrame(Sum_Radi_daily)
    # Daily GST
    AVE_GST_daily = np.array(AVE_GST_daily).reshape(n_days,total_years, order = 'F') #把数据变形一下
    Tground_daily_data = pd.DataFrame(AVE_GST_daily)
    # Daily WS
    AVE_WS_daily = np.array(AVE_WS_daily).reshape(n_days,total_years, order = 'F') #把数据变形一下
    WS_daily_data = pd.DataFrame(AVE_WS_daily)
    # Days of each month
    monthnum = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    # Mean daily temperatures: normalised to the mean.
    monthly_temp_data, monthly_temp_stats, monthly_std_temp_data = Generate_monthly_data(temp_data,monthnum)
    # Maximum daily temperatures: normalised to mean values
    monthly_temp_max_data, monthly_max_stats, monthly_std_temp_max_data = Generate_monthly_data(temp_max_data, monthnum)
    # Minimum daily temperatures: normalised to the mean value.
    monthly_temp_min_data, monthly_min_stats, monthly_std_temp_min_data = Generate_monthly_data(temp_min_data,monthnum)
    # Daily mean vapour pressure: normalised to the mean value
    monthly_VP_data, monthly_VP_stats, monthly_std_VP_data = Generate_monthly_data(VP_daily_data,monthnum)
    # Total daily radiation
    monthly_Radi_data, monthly_Radi_stats, monthly_std_Radi_data = Generate_monthly_data(Radi_daily_data, monthnum)
    # Daily mean ground surface temperature
    monthly_Tg_data, monthly_Tg_stats, monthly_std_Tg_data = Generate_monthly_data(Tground_daily_data,monthnum)
    # Daily mean wind speed
    monthly_ws_data, monthly_ws_stats, monthly_std_ws_data = Generate_monthly_data(WS_daily_data,monthnum)
    
    # Start from the start year
    start_year, end_year = parse_years_from_ID(start_ID, end_ID)
    print("Start year: %d, End year: %d" %(start_year, end_year))
    # Generate dates (skip 2/29)
    all_dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='D')
    all_dates = all_dates[~((all_dates.month == 2) & (all_dates.day == 29))]  # 去掉闰年的 2 月 29 日
    # Enlarge time index
    time_index = all_dates.repeat(24)  # 每天 24 小时
    # Add time index
    if len(time_index) > len(df_weather_data_final):
        time_index = time_index[:len(df_weather_data_final)]
    # Dataframe date time information
    df_weather_data_final['year'] = time_index.year
    df_weather_data_final['month'] = time_index.month
    df_weather_data_final['day'] = time_index.day
    df_weather_data_final['hour'] = list(range(24)) * (len(df_weather_data_final) // 24)
    # Calculate the number of the day
    df_weather_data_final['day_of_year'] = (df_weather_data_final.index // 24) % 365 + 1
    
    # RMY selection parameters: Mean DBT, Minimum DBT, Maximum DBT, Mean VP, Total GHI, Mean Ground Temperature, Mean WS
    weights_list = [2/16, 1/16, 1/16, 2/16, 8/16, 1/16, 1/16]
    select_days_list = [31,]
    df_selected_year_list = []
    for select_days in select_days_list:          
        Selected_data_list = []
        for month_now in list(range(1,13)):
            print(month_now)
            # Dataframe merge
            dataframes = [monthly_std_temp_data[month_now], monthly_std_temp_max_data[month_now], monthly_std_temp_min_data[month_now],
                        monthly_std_VP_data[month_now], monthly_std_Radi_data[month_now], monthly_std_Tg_data[month_now], monthly_std_ws_data[month_now]]
            weighted_sum = pd.DataFrame(0, index=dataframes[0].index, columns=dataframes[0].columns)
            # Find the minimum 
            min_value = monthly_std_temp_min_data[month_now].min().min()  # 
            min_col = monthly_std_temp_min_data[month_now].min().idxmin()  # 
            min_row = monthly_std_temp_min_data[month_now][min_col].idxmin()  # 
            print(f"Current month: {month_now}, Temperature Minimum Value Located at Row: {min_row}, Column: {min_col}")
            #print(f"Located at Row: {min_row}, Column: {min_col}")
            # Weighted dataframe
            for df, weight in zip(dataframes, weights_list):
                weighted_sum += df * weight
            
            min_stats = weighted_sum.loc[min_row, min_col]
            # Calculate CDF
            sorted_data, cdf = compute_cdf(weighted_sum)
            n_days_now = monthnum[month_now]
            # if selected_days is 31，it means every month needs to select as the days it contains.
            if select_days == 31:
                select_days_now = n_days_now
            else:
                select_days_now = select_days
            # Generate probabilities
            probabilities = np.linspace(0, 1, select_days_now)
            # Find the closest value
            if select_days_now == n_days_now:
                final_selected_values = find_closest_values(cdf, sorted_data, probabilities, min_stats=min_stats)
            elif select_days_now>=3 and select_days_now < n_days_now:
                selected_values, final_selected_values = select_representative_data(cdf, sorted_data, probabilities, total_days=n_days_now, min_stats=min_stats)
            else:
                selected_values, final_selected_values = select_representative_data(cdf, sorted_data, probabilities, total_days=n_days_now)
            # Find the corresponding (Year, month, day)
            original_indices = find_original_indices(weighted_sum, final_selected_values, start_year=start_year)
            # Find the original 24 hours from total dataset
            data_current_month = []
            # Show the selected results
            for i, (val, (day, year)) in enumerate(zip(final_selected_values, original_indices)):
                data_24 = df_weather_data_final[(df_weather_data_final["year"] == year)&(df_weather_data_final["month"] == month_now) & (df_weather_data_final["day"] == day)].reset_index(drop = True)
                data_current_month.append(data_24)
                if select_days_now == n_days_now:
                    print(f"Target {i + 1}: Value = {val:.4f}, Date = Day {day}, Month = {month_now}, Year {year}")
            data_current_merged = pd.concat(data_current_month, axis = 0).reset_index(drop = True)
            Selected_data_list.append(data_current_merged)
        df_selected_year = pd.concat(Selected_data_list, axis = 0)
        df_selected_year.reset_index(drop = True, inplace = True)
        df_selected_year_list.append(df_selected_year)
    
    # Write into different sheets of the excel
    folder_path = "ERA5_RMY/"+"RMY_"+str(start_ID) + "_" + str(end_ID) + "_" +cityname
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    output_excel_path = os.path.join(folder_path, output_excel_savename)
    # If already exists, it will delete and create a new
    if os.path.exists(output_excel_path):
        os.remove(output_excel_path)

    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        for df_now, sheet_name in zip(df_selected_year_list, select_days_list):
            print(df_now.head())
            print("Selected length of RMY:", len(df_now))
            if sheet_name == 31:
                #print(df_now.head())
                df_sorted = reorder_tmy(df_now, smooth=False)
            else:
                df_sorted = reorder_tmy_new(df_now)
            generated_hourly_DBT = df_sorted.iloc[:,0].values
            fig, ax = plt.subplots(figsize=(8, 5.5), dpi = 150)
            plt.plot(generated_hourly_DBT, color = "dodgerblue", alpha = 0.8)
            plt.xlabel("Time(h)")
            plt.ylabel("Dry-bulb temperature (\u2103)")
            plt.savefig(folder_path+"/" + "RMY_"+ cityname+ "_" + str(sheet_name)+".png")
            #plt.show()
            df_sorted.to_excel(writer, sheet_name=str(sheet_name), index=False)
            
    print(f"Data is successfully written into {output_excel_path}")
    

if __name__ == "__main__":
    main()