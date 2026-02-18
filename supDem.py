from retry_requests import retry
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import sys, os



def getWeather(lat=48.8534, lon=2.3488,
               solarPanel=1.6 , panelEf=.12):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "hourly": [
            "temperature_2m", 
            "wind_speed_10m", 
            "global_tilted_irradiance", 
                "cloud_cover"],
                }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
    hourly_global_tilted_irradiance = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()


    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["global_tilted_irradiance"] = hourly_global_tilted_irradiance/2
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["tilted_irrad_cover"]=((1-(hourly_data["cloud_cover"]/100))*hourly_data["global_tilted_irradiance"])
    hourly_data["kW_per_panel"]=hourly_data["tilted_irrad_cover"]*solarPanel*panelEf/1000
    hourly_data["kw_turbine"] = 0.01328 * (.25) * ((hourly_data["wind_speed_10m"]/3.6*2.23694)**3)/1000
    hourly_dataframe = pd.DataFrame(data = hourly_data)

    df_solar=pd.DataFrame()
    df_solar["dateFull"]=pd.date_range(hourly_dataframe["date"].min(),hourly_dataframe["date"].max(), freq='30min')
    df_solar["date"]=pd.to_datetime(df_solar.dateFull).dt.strftime('%Y-%m-%d %H:00:00')
    df_solar["date"]=pd.to_datetime(df_solar["date"])
    
    df_to_join=hourly_dataframe[["date","temperature_2m","wind_speed_10m","global_tilted_irradiance", "cloud_cover",
                                 "kW_per_panel","kw_turbine"]]
    df_to_join["date"]=pd.to_datetime(df_to_join["date"]).dt.tz_localize(None)
    df_solar=df_solar.merge(df_to_join,how='left', left_on='date', right_on='date')
 
    return  df_solar



def SupDem(directory_path = 'data', demand_data_path ='data/consumptionParis.csv', save=True):
    """
    Parameters
    
        directory_path
            Path to the directory where data is kept
        consCsvPath
            Path to the consumption data
        save
            Boolean to save the data from supply and demand
    Returns
    -------
        s_others_yearly
            Solar supply
        e_others_yearly
            Wind supply
        d_others_yearly
            Demand
    """
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            print("Please set 'directory_path' to the location of your CSV files.")
            directory_path = '.' # Fallback to current directory
    except NameError:
        print("Setting 'directory_path' to the current directory.")
        directory_path = '.'

    try:
        dfEnergy = getWeather()
        # Create a proper datetime index
        dfEnergy['datetime'] = pd.to_datetime(dfEnergy['dateFull'])
        dfEnergy.set_index('datetime', inplace=True)

        # --- SCALING IMPROVEMENT ---
        # Increased the number of panels for a more realistic community supply
        num_panels_community = round((87836 - 7800) * 0.001) * 400 # Changed from 100 to 400
        print(f"Adjusted number of community panels to: {num_panels_community}")

        # Correct scaling to kW
        s_others_yearly = dfEnergy["kW_per_panel"] * num_panels_community
        e_others_yearly = dfEnergy["kw_turbine"] * num_panels_community
    except FileNotFoundError:
        print(f"Error: An error occured. Please check the path.")
        s_others_yearly = None

    
    try:
        df_demand = pd.read_csv(demand_data_path, delimiter=";")
        # Filter for the year 2023
        df_demand["Date"] = pd.to_datetime(df_demand["Date"], format='mixed')
        df_demand = df_demand[df_demand.Date.dt.year == 2023].copy()

        # Create a proper datetime index
        df_demand['datetime'] = pd.to_datetime(df_demand['Date'].astype(str) + ' ' + df_demand['Heures'])
        df_demand.set_index('datetime', inplace=True)

        # --- PLOTTING FIX ---
        # Sort the index to ensure chronological order for correct plotting
        df_demand.sort_index(inplace=True)

        # Define scaling factor for the community
        buildings_in_community = 1000
        avg_people_per_building = 4
        community_population = buildings_in_community * avg_people_per_building
        grand_paris_population = 12000000
        scaling_factor = community_population / grand_paris_population

        # Calculate community demand in kW
        d_others_yearly = df_demand["Consommation(MW)"] * scaling_factor * 1000

    except FileNotFoundError:
        print(f"Error: Could not find '{os.path.basename(demand_data_path)}'. Please check the path.")
        d_others_yearly = None
    # --- 3. Save the generated data to CSV files ---
    if save==True:
        if s_others_yearly is not None:

            (s_others_yearly+e_others_yearly).to_csv( 'data/s_others_yearly.csv', header=['supply_kw'])
            print("\nSaved s_others_yearly.csv")

        if d_others_yearly is not None:
            d_others_yearly.to_csv('data/d_others_yearly.csv', header=['demand_kw'])
            print("Saved d_others_yearly.csv")
    
    return s_others_yearly,e_others_yearly, d_others_yearly