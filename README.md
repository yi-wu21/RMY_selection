# RMY_selection
A new representative meteorological year (RMY) selection method based on daily statistics distribution.

## 1. Introduction
The python file is used to select RMY based on daily statistics distribution. It includes several functions:
1) Read data from climate database (Example of climate data is also provided in the folder: Example_climate.db)
2) Provide CDF functions, daily statistics calculation and selection
3) Daily smoothly and save to target save path.

## 2. Usage
Usage: python your_script.py <your climate database . db> <cityname> <start_year> <end_year> <RMY_save_path.xlsx>
For instance:
python RMY_selection_daily_statistics.py Example_climate.db Beijing 2001 2020 RMY_beijing_2001_2020.xlsx

## 3. Requirements
The packages required are included in the requirements.txt file.
