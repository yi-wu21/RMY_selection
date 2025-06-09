# RMY_selection
A new representative meteorological year (RMY) selection method based on daily statistics distribution.

## 1. Introduction
The python file is used to select RMY based on daily statistics distribution. It includes several functions:
1) Read data from climate database (Example of climate data is also provided in the folder: Example_climate.db)
2) Provide CDF functions, daily statistics calculation and selection
3) Daily smoothly and save to target save path.

## 2. Usage
Usage: python your_script.py <your climate database . db> <cityname / citylist.csv> <start_year> <end_year> <RMY_save_path.xlsx>
For instance:
1. python RMY_selection_daily_statistics.py Example_climate.db Beijing 2001 2020 RMY_2001_2020_Beijing.xlsx
2. python RMY_selection_daily_statistics.py Example_climate.db citylist.csv 2001 2020 RMY_.xlsx

When using citylist.csv, it will read the first column of your input csv file as the list of cities.
And automatically output the RMY into a file named "RMY_save_path_{cityname}.xlsx". The city list should be included in your Climate db file.

## 3. Requirements
The packages required are included in the requirements.txt file.
