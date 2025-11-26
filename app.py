# Streamlit dashboard for covid-19

'''
  Main question:
  When, how, and why did COVID-19 become endemic?

  |A disease outbreak is endemic when it is 
  |consistently present but limited to a particular region.
  
  Stakeholders:
  - Classmates
  - Data visualization professor
  - Gen Z audience

  Core Graphs:
  - Global new cases rate (time series)
  - Countries with the highest new cases rate
  - Countries showing negative new cases rate (declining)

  - Section 1: The Outbreak and the Crisis Phase
    How fast was COVID spreading per country (and which countries were hit hardest)?
    How fast did authorities react (delay until first case)?
    Which was the “darkest day” (peak new cases)?
    When would humanity disappear if no one was vaccinated? (simulation)

    Global new cases time series
    Country-level spread speed (small multiples or ranked lines)
    Reaction delay choropleth
    Worst-day per country bar chart / heatmap

  - Section 2: The Transition to Endemicity 
    Which countries still show high circulation (highest new cases per rate)?
    Which countries report negative new case rates (declines)?
    What signals the shift from pandemic → endemic?

    Map or chart of highest new cases rate
    Chart of countries with negative week-over-week rates
    Trend of declining case fatality ratio (CFR)
    Post-2022 stabilization time series
'''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng as rng

# Load the data
# Daily frequency reporting of new COVID-19 cases and deaths by date reported to WHO until August 2023 
DAILY_DATA = pd.read_csv('WHO-COVID-19-global-daily-data.csv')

st.set_page_config(layout="wide")
df_cases_deaths = pd.DataFrame(DAILY_DATA[['Date_reported', 'New_cases', 'New_deaths']])


with st.container(width='stretch'):
  st.write("This is inside the container")
  DAILY_DATA

