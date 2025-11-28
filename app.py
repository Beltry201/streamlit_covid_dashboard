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
import altair as alt
import pydeck as pdk


PRIMARY_BLUE = "#1d4ed8"
SECONDARY_BLUE = "#2563eb"
TERTIARY_BLUE = "#3b82f6"
LIGHT_BLUE = "#bfdbfe"
BLUE_SCALE = [
  PRIMARY_BLUE,
  SECONDARY_BLUE,
  TERTIARY_BLUE,
  "#60a5fa",
  "#93c5fd",
  "#bfdbfe",
]
EXCLUDED_COUNTRIES = {"China"}


def _load_daily() -> pd.DataFrame:
  df = pd.read_csv('WHO-COVID-19-global-daily-data.csv')
  df['Date_reported'] = pd.to_datetime(df['Date_reported'])
  df['New_cases'] = pd.to_numeric(df['New_cases'], errors='coerce').fillna(0)
  df['New_deaths'] = pd.to_numeric(df['New_deaths'], errors='coerce').fillna(0)
  return df


def _load_table() -> pd.DataFrame:
  df = pd.read_csv('WHO-COVID-19-global-table-data.csv')
  df = df.rename(columns={'Name': 'Country'})
  return df


DAILY_DATA = _load_daily()
TABLE_DATA = _load_table()

if EXCLUDED_COUNTRIES:
  DAILY_DATA = DAILY_DATA[~DAILY_DATA['Country'].isin(EXCLUDED_COUNTRIES)].copy()
  TABLE_DATA = TABLE_DATA[~TABLE_DATA['Country'].isin(EXCLUDED_COUNTRIES)].copy()

st.set_page_config(layout="wide")

DECLINE_END_DATE = pd.Timestamp('2024-12-31')
SECTION_TITLES = (
  "The Outbreak & Crisis Phase",
  "From Peak to Plateau",
)
PROJECTION_DAYS = 20
MIN_PROJECTION_GROWTH = 0.5
MAX_PROJECTION_GROWTH = 3
DECLINE_ANNOTATION = (
  "After the global peak on January 30, 2022, cases dropped by over 80% in just four months.\n"
  "Without intervention, that curve would have continued rising exponentially."
)


@st.cache_data
def get_global_timeseries(df: pd.DataFrame) -> pd.DataFrame:
  ts = df.groupby('Date_reported', as_index=False)[['New_cases', 'New_deaths']].sum()
  ts['rolling_avg_cases'] = ts['New_cases'].rolling(7).mean()
  ts['cumulative_cases'] = ts['New_cases'].cumsum()
  ts['cumulative_deaths'] = ts['New_deaths'].cumsum()
  growth = ts['New_cases'].pct_change().replace([np.inf, -np.inf], np.nan)
  ts['daily_growth_pct'] = growth.fillna(0) * 100
  return ts


@st.cache_data
def get_country_peaks(df: pd.DataFrame) -> pd.DataFrame:
  idx = df.groupby('Country')['New_cases'].idxmax()
  peaks = df.loc[idx, ['Country', 'Date_reported', 'New_cases', 'WHO_region']].reset_index(drop=True)
  return peaks.sort_values('New_cases', ascending=False)


@st.cache_data
def get_weekly_spread(df: pd.DataFrame) -> pd.DataFrame:
  weekly = df.copy()
  weekly['week'] = weekly['Date_reported'].dt.to_period('W').apply(lambda r: r.start_time)
  return weekly.groupby(['Country', 'week'], as_index=False)['New_cases'].sum()


@st.cache_data
def build_cases_geojson(metadata: pd.DataFrame) -> dict:
  import json
  from urllib.request import urlopen

  url = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json'
  with urlopen(url) as response:
    geojson = json.load(response)

  data = metadata.copy()
  data['total_cases'] = pd.to_numeric(
    data['Cases - cumulative total'], errors='coerce'
  ).fillna(0)
  data['cases_per_100k'] = pd.to_numeric(
    data['Cases - cumulative total per 100000 population'], errors='coerce'
  ).fillna(0)
  stats = data[['Country', 'total_cases', 'cases_per_100k']].set_index('Country').to_dict('index')

  max_rate = max((info['cases_per_100k'] for info in stats.values()), default=0)
  max_log_rate = np.log1p(max_rate) if max_rate > 0 else 0
  base_rgba = np.array([220, 233, 255, 60])
  peak_rgba = np.array([15, 76, 146, 220])

  for feature in geojson['features']:
    name = feature['properties'].get('name')
    info = stats.get(name)
    if not info or info['total_cases'] <= 0:
      feature['properties']['cases_total'] = None
      feature['properties']['cases_per_100k'] = None
      feature['properties']['fill_color'] = [220, 220, 220, 60]
      continue

    rate = info['cases_per_100k']
    if rate <= 0 or max_log_rate == 0:
      intensity = 0
    else:
      intensity = np.log1p(rate) / max_log_rate
    color = (base_rgba + (peak_rgba - base_rgba) * intensity).astype(int).tolist()

    feature['properties']['cases_total'] = int(info['total_cases'])
    feature['properties']['cases_per_100k'] = float(rate)
    feature['properties']['fill_color'] = color

  return geojson


def build_no_action_projection(
  start_cases: float,
  start_date: pd.Timestamp,
  daily_growth_pct: float,
  days: int = PROJECTION_DAYS,
) -> pd.DataFrame:
  if pd.isna(start_cases) or start_cases <= 0:
    return pd.DataFrame(columns=['Date_reported', 'Projected_cases'])

  growth_pct = np.clip(daily_growth_pct, MIN_PROJECTION_GROWTH, MAX_PROJECTION_GROWTH)
  growth = growth_pct / 100
  horizons = np.arange(days)
  projected_cases = start_cases * np.power(1 + growth, horizons)
  dates = pd.date_range(start=start_date, periods=days, freq='D')
  return pd.DataFrame({'Date_reported': dates, 'Projected_cases': projected_cases})


@st.cache_data
def get_post2022_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
  post = ts[ts['Date_reported'] >= pd.Timestamp('2022-01-01')].copy()
  post['rolling30_cases'] = post['New_cases'].rolling(30).mean()
  post['rolling30_deaths'] = post['New_deaths'].rolling(30).mean()
  post['cfr_pct'] = np.where(
    post['cumulative_cases'] > 0,
    (post['cumulative_deaths'] / post['cumulative_cases']) * 100,
    np.nan,
  )
  return post


@st.cache_data
def get_country_weekly_rates(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
  weekly = df.copy()
  weekly['week'] = weekly['Date_reported'].dt.to_period('W').apply(lambda r: r.start_time)
  weekly_agg = (
    weekly.groupby(['Country', 'Country_code', 'WHO_region', 'week'], as_index=False)['New_cases']
    .sum()
  )

  meta = metadata.copy()
  meta['Cases - cumulative total'] = pd.to_numeric(
    meta['Cases - cumulative total'], errors='coerce'
  )
  meta['Cases - cumulative total per 100000 population'] = pd.to_numeric(
    meta['Cases - cumulative total per 100000 population'], errors='coerce'
  )
  denominator = meta['Cases - cumulative total per 100000 population'].replace({0: np.nan})
  meta['population_est'] = (meta['Cases - cumulative total'] / denominator) * 100000

  merged = weekly_agg.merge(
    meta[['Country', 'WHO Region', 'population_est']], on='Country', how='left'
  )
  merged['WHO_region'] = merged['WHO_region'].fillna(merged['WHO Region'])
  merged = merged.drop(columns=['WHO Region'])
  merged['cases_per_100k'] = np.where(
    merged['population_est'] > 0,
    (merged['New_cases'] / merged['population_est']) * 100000,
    np.nan,
  )
  return merged


@st.cache_data
def get_negative_trends(
  weekly_rates: pd.DataFrame, window_weeks: int = 3
) -> pd.DataFrame:
  summaries = []
  for country, group in weekly_rates.groupby('Country'):
    ordered = group.sort_values('week')
    tail = ordered.tail(window_weeks)
    if tail.empty:
      continue
    values = tail['cases_per_100k'].to_numpy()
    decline = len(values) == window_weeks and np.all(np.diff(values) < 0)
    summaries.append(
      {
        'Country': country,
        'WHO_region': tail['WHO_region'].iloc[-1],
        'current_rate': tail['cases_per_100k'].iloc[-1],
        'window_start': tail['week'].iloc[0],
        'window_end': tail['week'].iloc[-1],
        'slope': values[-1] - values[0],
        'is_declining': bool(decline),
      }
    )
  return pd.DataFrame(summaries)


@st.cache_data
def get_stabilization_metrics(weekly_rates: pd.DataFrame) -> pd.DataFrame:
  post = weekly_rates[weekly_rates['week'] >= pd.Timestamp('2022-01-03')].copy()
  stats = (
    post.groupby('Country')
    .agg(
      WHO_region=('WHO_region', 'last'),
      mean_weekly=('New_cases', 'mean'),
      std_weekly=('New_cases', 'std'),
      latest_week=('week', 'max'),
    )
    .reset_index()
  )
  stats['variance_index'] = stats['std_weekly'] / stats['mean_weekly'].replace(0, np.nan)
  stats['variance_index'] = stats['variance_index'].fillna(np.inf)
  return stats


global_ts = get_global_timeseries(DAILY_DATA)
country_peaks = get_country_peaks(DAILY_DATA)
weekly_spread = get_weekly_spread(DAILY_DATA)
post2022_ts = get_post2022_timeseries(global_ts)
country_weekly_rates = get_country_weekly_rates(DAILY_DATA, TABLE_DATA)
cases_geojson = build_cases_geojson(TABLE_DATA)

darkest_day = global_ts.loc[global_ts['New_cases'].idxmax()]
darkest_growth_pct = darkest_day['daily_growth_pct']
total_countries = DAILY_DATA['Country'].nunique()
latest_totals = global_ts.iloc[-1]
latest_growth = latest_totals['daily_growth_pct']
recent_growth_avg = global_ts['daily_growth_pct'].tail(7).mean()
peak_growth_row = global_ts.loc[global_ts['daily_growth_pct'].idxmax()]
no_action_projection = build_no_action_projection(
  start_cases=darkest_day['New_cases'],
  start_date=darkest_day['Date_reported'],
  daily_growth_pct=darkest_growth_pct,
  days=PROJECTION_DAYS,
)
annotation_date = min(
  global_ts['Date_reported'].max(),
  darkest_day['Date_reported'] + pd.Timedelta(days=120),
)
annotation_level = float(darkest_day['New_cases']) * 0.55

decline_start = darkest_day['Date_reported']
decline_end = min(DECLINE_END_DATE, global_ts['Date_reported'].max())
decline_window_label = f"{decline_start:%b %d, %Y} → {decline_end:%b %d, %Y}"

post_peak_ts = post2022_ts[
  (post2022_ts['Date_reported'] >= decline_start) &
  (post2022_ts['Date_reported'] <= decline_end)
].copy()

section2_weekly_rates = country_weekly_rates[
  (country_weekly_rates['week'] >= decline_start) &
  (country_weekly_rates['week'] <= decline_end)
].copy()

negative_trends = get_negative_trends(section2_weekly_rates)
stabilization_metrics = get_stabilization_metrics(section2_weekly_rates)

section2_latest_week = section2_weekly_rates['week'].max()
region_options = sorted(section2_weekly_rates['WHO_region'].dropna().unique().tolist())
if not region_options:
  region_options = []

if not post_peak_ts.empty:
  post_peak_latest = post_peak_ts.iloc[-1]
  rolling_threshold = post_peak_ts['rolling30_cases'].quantile(0.75)
  recent_spikes = post_peak_ts[
    post_peak_ts['rolling30_cases'] >= rolling_threshold
  ]
  last_spike_date = (
    recent_spikes['Date_reported'].max()
    if not recent_spikes.empty
    else post_peak_latest['Date_reported']
  )
  days_since_spike = (post_peak_latest['Date_reported'] - last_spike_date).days
  current_cfr = post_peak_latest['cfr_pct']
else:
  days_since_spike = None
  current_cfr = None


def render_section_one():
  st.title(SECTION_TITLES[0])

  metric_cols = st.columns(3)
  metric_cols[0].metric(
    "Total reported cases",
    f"{int(latest_totals['cumulative_cases']):,}",
    help="Global cumulative cases through the end of the dataset.",
  )
  metric_cols[1].metric(
    "Darkest day · new cases",
    f"{int(darkest_day['New_cases']):,}",
    f"{darkest_day['Date_reported'].date()}",
    help="Date with the highest global daily cases.",
  )
  metric_cols[2].metric(
    "Darkest-day growth",
    f"{darkest_growth_pct:.1f}%",
    f"{darkest_day['Date_reported'].date():%b %d}",
    help="Percent change in new cases vs the day before the worst-case spike.",
  )

  st.divider()

  tab_timeline, tab_countries = st.tabs(["Crisis Timeline", "Country Comparisons"])

  with tab_timeline:
    st.subheader("Covid-19 Global new cases")
    base = alt.Chart(global_ts).properties(height=350)
    layers = []
    if not no_action_projection.empty:
      projection_chart = (
        alt.Chart(no_action_projection.rename(columns={'Projected_cases': 'New_cases'}))
        .mark_line(color="#ef4444", strokeDash=[3, 3], strokeWidth=2)
        .encode(
          x='Date_reported:T',
          y=alt.Y('New_cases:Q', title='New cases'),
          tooltip=[
            alt.Tooltip('Date_reported:T', title='Date'),
            alt.Tooltip('New_cases:Q', title='Projected cases', format=','),
          ],
        )
      )
      layers.append(projection_chart)

    line_cases = base.mark_line(color=PRIMARY_BLUE, strokeWidth=2).encode(
      x='Date_reported:T',
      y=alt.Y('New_cases:Q', title='New cases'),
      tooltip=[
        alt.Tooltip('Date_reported:T', title='Date'),
        alt.Tooltip('New_cases:Q', title='New cases', format=','),
        alt.Tooltip('rolling_avg_cases:Q', title='7d avg', format=',.0f'),
      ],
    )
    rolling_line = base.mark_line(color=SECONDARY_BLUE, strokeDash=[4, 4]).encode(
      x='Date_reported:T',
      y='rolling_avg_cases:Q',
    )
    layers.extend([line_cases, rolling_line])

    annotation_chart = (
      alt.Chart(
        pd.DataFrame(
          {
            'Date_reported': [annotation_date],
            'New_cases': [annotation_level],
            'label': [DECLINE_ANNOTATION],
          }
        )
      )
      .mark_text(
        align='left',
        color=PRIMARY_BLUE,
        fontSize=14,
        fontWeight='bold',
        lineBreak='\n',
        dx=10,
        dy=-10,
      )
      .encode(
        x='Date_reported:T',
        y='New_cases:Q',
        text='label',
      )
    )
    layers.append(annotation_chart)

    st.altair_chart(alt.layer(*layers), use_container_width=True)

  with tab_countries:
    st.subheader("Covid-19 Cases per country")
    col_a, col_b = st.columns((2, 1))

    top_country_names = country_peaks.head(10)['Country'].tolist()
    default_selected = top_country_names[:4]
    selected_countries = col_a.multiselect(
      "Compare countries",
      top_country_names,
      default=default_selected,
      help="Derived from countries with the highest recorded single-day spikes.",
    )

    if selected_countries:
      filtered = weekly_spread[weekly_spread['Country'].isin(selected_countries)]
      
      metadata_totals = TABLE_DATA.copy()
      metadata_totals['total_cases'] = pd.to_numeric(
        metadata_totals['Cases - cumulative total'], errors='coerce'
      ).fillna(0)
      country_totals = (
        metadata_totals[metadata_totals['Country'].isin(selected_countries)]
        [['Country', 'total_cases']]
        .sort_values('total_cases', ascending=False)
      )
      country_order = country_totals['Country'].tolist()
      
      spread_chart = alt.Chart(filtered).mark_line().encode(
        x='week:T',
        y=alt.Y('New_cases:Q', title='Weekly new cases'),
        color=alt.Color(
          'Country:N',
          scale=alt.Scale(domain=country_order, range=BLUE_SCALE),
          sort=country_order,
        ),
        order=alt.Order('Country:N', sort='descending'),
        tooltip=['Country:N', 'week:T', alt.Tooltip('New_cases:Q', format=',')],
      ).properties(height=300)
      
      col_a.altair_chart(spread_chart, use_container_width=True)
    else:
      col_a.info("Select at least one country to display weekly trends.")

    cases_layer = pdk.Layer(
      "GeoJsonLayer",
      data=cases_geojson,
      pickable=True,
      stroked=True,
      get_line_color=[255, 255, 255],
      get_fill_color='properties.fill_color',
      auto_highlight=True,
    )
    deck = pdk.Deck(
      layers=[cases_layer],
      initial_view_state=pdk.ViewState(latitude=10, longitude=0, zoom=0.8),
      map_style=None,
      tooltip={
        "html": "<b>{name}</b><br/>Cases: {cases_total}<br/>Rate: {cases_per_100k} per 100k",
        "style": {"color": "white"},
      },
    )
    col_b.pydeck_chart(deck)

    st.subheader("Darkest day per country")
    peaks_chart = alt.Chart(country_peaks.head(20)).mark_bar(color=PRIMARY_BLUE).encode(
      x=alt.X('New_cases:Q', title='Peak new cases'),
      y=alt.Y('Country:N', sort='-x'),
      tooltip=[
        'Country:N',
        alt.Tooltip('New_cases:Q', format=',', title='Peak cases'),
        alt.Tooltip('Date_reported:T', title='Date'),
      ],
    ).properties(height=500)
    st.altair_chart(peaks_chart, use_container_width=True)


def render_section_two():
  st.title(SECTION_TITLES[1])
  st.caption(
    "Following the descent from the darkest day through late 2024 to spot lingering hotspots, sustained declines, and early signals of endemic stability."
  )

  st.subheader(f"Come-down after the peak ({decline_window_label})")
  if post_peak_ts.empty:
    st.info("Data for the decline window is unavailable in the source file.")
  else:
    st.markdown(
      "The darkest day (vertical line) marked the inflection point—by tracking 30-day averages after that moment we can see how quickly the curve eased."
    )
    decline_area = (
      alt.Chart(post_peak_ts)
      .mark_area(color=LIGHT_BLUE)
      .encode(
        x='Date_reported:T',
        y=alt.Y('rolling30_cases:Q', title='30-day average new cases'),
      )
    )
    peak_marker = (
      alt.Chart(pd.DataFrame({'Date_reported': [decline_start]}))
      .mark_rule(color=PRIMARY_BLUE, strokeDash=[6, 4])
      .encode(x='Date_reported:T')
    )
    jan2023_marker = (
      alt.Chart(pd.DataFrame({'Date_reported': [pd.Timestamp('2023-01-01')]}))
      .mark_rule(color="#ef4444", strokeDash=[3, 3])
      .encode(x='Date_reported:T')
    )
    st.altair_chart(decline_area + peak_marker + jan2023_marker, use_container_width=True)
    st.caption(
      "Narrative focus: document the slope of the decline through 2024 to contextualize the subsequent tabs."
    )

  st.divider()

  st.subheader("Signals of stabilization")
  metrics_col, chart_col = st.columns((1, 2))
  metrics_col.metric(
    "Days since last major spike",
    f"{days_since_spike} days" if days_since_spike is not None else "n/a",
    help="Days since global rolling 30-day cases were within the top quartile of post-2022 levels.",
  )
  metrics_col.metric(
    "Current global CFR",
    f"{current_cfr:.2f}%" if current_cfr is not None else "n/a",
    help="Cumulative deaths / cumulative cases, post-2022 view.",
  )

  if post_peak_ts.empty:
    chart_col.info("Decline-window slice unavailable for CFR comparison.")
  else:
    cfr_cases_chart = (
      alt.layer(
        alt.Chart(post_peak_ts)
        .mark_line(color=SECONDARY_BLUE)
        .encode(
          x='Date_reported:T',
          y=alt.Y('rolling30_cases:Q', title='30d avg new cases'),
          tooltip=[
            alt.Tooltip('Date_reported:T', title='Date'),
            alt.Tooltip('rolling30_cases:Q', title='30d avg cases', format=',.0f'),
          ],
        ),
        alt.Chart(post_peak_ts)
        .mark_line(color=PRIMARY_BLUE)
        .encode(
          x='Date_reported:T',
          y=alt.Y('cfr_pct:Q', title='CFR (%)', axis=alt.Axis(grid=False)),
          tooltip=[
            alt.Tooltip('Date_reported:T', title='Date'),
            alt.Tooltip('cfr_pct:Q', title='CFR (%)', format='.2f'),
          ],
        ),
      )
      .resolve_scale(y='independent')
      .properties(height=320, title="Post-2022 global cases vs CFR")
    )
    chart_col.altair_chart(cfr_cases_chart, use_container_width=True)

  selected_regions_stable = st.multiselect(
    "Surface low-variance countries",
    options=region_options,
    default=region_options,
    key="stable_regions",
  )
  stable_table = stabilization_metrics.copy()
  if selected_regions_stable:
    stable_table = stable_table[stable_table['WHO_region'].isin(selected_regions_stable)]
  stable_table = stable_table.replace([np.inf, -np.inf], np.nan)
  stable_table = stable_table.dropna(subset=['variance_index'])
  stable_show = stable_table.nsmallest(10, 'variance_index')

  st.markdown("**Most stable weekly patterns since 2022** (lower variance = flatter curve)")
  st.dataframe(
    stable_show[['Country', 'WHO_region', 'variance_index', 'mean_weekly']],
    use_container_width=True,
  )


st.sidebar.title("Covid-19 Pandemic")
section_choice = st.sidebar.radio("Story sections", SECTION_TITLES, index=0)
if section_choice == SECTION_TITLES[0]:
  render_section_one()
else:
  render_section_two()

