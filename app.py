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

st.set_page_config(layout="wide")


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
def get_reaction_delay(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
  first_cases = (
    df[df['New_cases'] > 0]
    .groupby('Country', as_index=False)['Date_reported']
    .min()
    .rename(columns={'Date_reported': 'first_case'})
  )
  baseline = df['Date_reported'].min()
  first_cases['delay_days'] = (first_cases['first_case'] - baseline).dt.days
  merged = first_cases.merge(metadata[['Country', 'WHO Region']], on='Country', how='left')
  return merged


@st.cache_data
def build_delay_geojson(reaction_df: pd.DataFrame) -> dict:
  import json
  from urllib.request import urlopen

  url = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json'
  with urlopen(url) as response:
    geojson = json.load(response)

  delay_map = reaction_df.set_index('Country')['delay_days'].to_dict()

  for feature in geojson['features']:
    name = feature['properties'].get('name')
    delay = delay_map.get(name)
    if delay is None or np.isnan(delay):
      feature['properties']['delay_days'] = None
      feature['properties']['color'] = [200, 200, 200, 120]
      continue

    clipped = max(0, min(120, int(delay)))
    feature['properties']['delay_days'] = int(delay)
    feature['properties']['color'] = [255 - clipped, 100 + clipped // 2, 90, 180]

  return geojson


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
reaction_delay = get_reaction_delay(DAILY_DATA, TABLE_DATA)
post2022_ts = get_post2022_timeseries(global_ts)
country_weekly_rates = get_country_weekly_rates(DAILY_DATA, TABLE_DATA)
negative_trends = get_negative_trends(country_weekly_rates)
stabilization_metrics = get_stabilization_metrics(country_weekly_rates)

darkest_day = global_ts.loc[global_ts['New_cases'].idxmax()]
darkest_growth_pct = darkest_day['daily_growth_pct']
total_countries = DAILY_DATA['Country'].nunique()
latest_totals = global_ts.iloc[-1]
latest_growth = latest_totals['daily_growth_pct']
recent_growth_avg = global_ts['daily_growth_pct'].tail(7).mean()
peak_growth_row = global_ts.loc[global_ts['daily_growth_pct'].idxmax()]

latest_week = country_weekly_rates['week'].max()
region_options = sorted(country_weekly_rates['WHO_region'].dropna().unique().tolist())

if not post2022_ts.empty:
  post2022_latest = post2022_ts.iloc[-1]
  rolling_threshold = post2022_ts['rolling30_cases'].quantile(0.75)
  recent_spikes = post2022_ts[
    post2022_ts['rolling30_cases'] >= rolling_threshold
  ]
  last_spike_date = (
    recent_spikes['Date_reported'].max()
    if not recent_spikes.empty
    else post2022_latest['Date_reported']
  )
  days_since_spike = (post2022_latest['Date_reported'] - last_spike_date).days
  current_cfr = post2022_latest['cfr_pct']
else:
  days_since_spike = None
  current_cfr = None

st.title("Section 1 · The Outbreak & Crisis Phase")
st.caption("Tracking how fast COVID-19 spread, how authorities reacted, and when the crisis peaked.")

with st.container():
  st.subheader("Guiding Questions")
  st.markdown(
    "- How fast was COVID spreading per country and which countries were hit the hardest?\n"
    "- How fast did authorities react (delay until the first reported case)?\n"
    "- Which day recorded the worst global caseload?\n"
    "- What would the curve look like without vaccinations?"
  )

st.divider()

metric_cols = st.columns(4)
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
  "Countries reporting cases",
  f"{total_countries}",
  help="Number of unique territories with at least one reported infection.",
)
metric_cols[3].metric(
  "Darkest-day growth",
  f"{darkest_growth_pct:.1f}%",
  f"{darkest_day['Date_reported'].date():%b %d}",
  help="Percent change in new cases vs the day before the worst-case spike.",
)

st.divider()

tab_timeline, tab_countries = st.tabs(["Crisis Timeline", "Country Comparisons"])

with tab_timeline:
  st.subheader("Global new cases")
  base = alt.Chart(global_ts).properties(height=350)
  line_cases = base.mark_line(color="#e63946", strokeWidth=2).encode(
    x='Date_reported:T',
    y=alt.Y('New_cases:Q', title='New cases'),
    tooltip=[
      alt.Tooltip('Date_reported:T', title='Date'),
      alt.Tooltip('New_cases:Q', title='New cases', format=','),
      alt.Tooltip('rolling_avg_cases:Q', title='7d avg', format=',.0f'),
    ],
  )
  rolling_line = base.mark_line(color="#1d3557", strokeDash=[4, 4]).encode(
    x='Date_reported:T',
    y='rolling_avg_cases:Q',
  )
  st.altair_chart(line_cases + rolling_line, use_container_width=True)

  st.subheader("No-vaccine spread simulation")
  col_left, col_right = st.columns([1, 2])
  with col_left:
    initial_cases = col_left.number_input("Initial cases", value=1000, min_value=1)
    daily_growth = col_left.slider("Daily growth (%)", min_value=-5.0, max_value=15.0, value=4.0, step=0.5)
    horizon = col_left.slider("Days simulated", min_value=30, max_value=365, value=120, step=10)

  days = np.arange(horizon)
  trajectory = initial_cases * np.power(1 + daily_growth / 100, days)
  sim_df = pd.DataFrame({'Day': days, 'Projected_cases': trajectory})
  sim_chart = alt.Chart(sim_df).mark_line(color="#06d6a0").encode(
    x='Day:Q',
    y=alt.Y('Projected_cases:Q', title='Projected cases'),
    tooltip=['Day:Q', alt.Tooltip('Projected_cases:Q', format=',')],
  )
  col_right.altair_chart(sim_chart, use_container_width=True)

with tab_countries:
  st.subheader("Country-level pressure")
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
    spread_chart = alt.Chart(filtered).mark_line().encode(
      x='week:T',
      y=alt.Y('New_cases:Q', title='Weekly new cases'),
      color='Country:N',
      tooltip=['Country:N', 'week:T', alt.Tooltip('New_cases:Q', format=',')],
    ).properties(height=300)
    col_a.altair_chart(spread_chart, use_container_width=True)
  else:
    col_a.info("Select at least one country to display weekly trends.")

  geojson = build_delay_geojson(reaction_delay)
  delay_layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson,
    pickable=True,
    stroked=True,
    get_line_color=[255, 255, 255],
    get_fill_color='properties.color',
    auto_highlight=True,
  )
  deck = pdk.Deck(
    layers=[delay_layer],
    initial_view_state=pdk.ViewState(latitude=10, longitude=0, zoom=0.8),
    map_style=None,
    tooltip={"html": "<b>{name}</b><br/>Delay: {delay_days} days", "style": {"color": "white"}},
  )
  col_b.pydeck_chart(deck)

  st.subheader("Darkest day per country")
  peaks_chart = alt.Chart(country_peaks.head(20)).mark_bar(color="#ffba08").encode(
    x=alt.X('New_cases:Q', title='Peak new cases'),
    y=alt.Y('Country:N', sort='-x'),
    tooltip=[
      'Country:N',
      alt.Tooltip('New_cases:Q', format=',', title='Peak cases'),
      alt.Tooltip('Date_reported:T', title='Date'),
    ],
  ).properties(height=500)
  st.altair_chart(peaks_chart, use_container_width=True)

st.divider()

st.title("Section 2 · Transition to Endemicity")
st.caption("Assessing where COVID-19 still circulates, who is pulling ahead with declines, and how the global system stabilizes.")

with st.container():
  st.subheader("Guiding Questions")
  st.markdown(
    "- Which countries still show high circulation (highest new cases per rate)?\n"
    "- Which countries report negative new case rates (declines)?\n"
    "- What signals the shift from pandemic → endemic (CFR, stabilization)?"
  )

st.divider()

tab_high, tab_decline, tab_stable = st.tabs(
  ["High circulation", "Declining signals", "Stabilization"]
)

with tab_high:
  st.subheader("Where circulation remains high")
  selected_regions_high = st.multiselect(
    "Focus regions",
    options=region_options,
    default=region_options,
    help="Filter the ranking to WHO regions of interest.",
  )
  if pd.isna(latest_week):
    st.info("Weekly aggregates unavailable for high-circulation ranking.")
  else:
    latest_label = latest_week.strftime("%b %d, %Y")
    latest_rates = country_weekly_rates[country_weekly_rates['week'] == latest_week]
    if selected_regions_high:
      latest_rates = latest_rates[latest_rates['WHO_region'].isin(selected_regions_high)]

    col_metrics, col_chart = st.columns((1, 2))
    top_rates = latest_rates.dropna(subset=['cases_per_100k']).nlargest(10, 'cases_per_100k')

    if not top_rates.empty:
      avg_top5 = top_rates.head(5)['cases_per_100k'].mean()
      col_metrics.metric(
        "Avg top-5 rate (/100k)",
        f"{avg_top5:.1f}",
        help=f"Average weekly incidence among the five hottest spots ({latest_label}).",
      )
      col_metrics.metric(
        "Regions represented",
        f"{top_rates['WHO_region'].nunique()}",
        help="Distinct WHO regions in the top-10 ranking.",
      )

      high_chart = (
        alt.Chart(top_rates)
        .mark_bar(color="#e36414")
        .encode(
          x=alt.X('cases_per_100k:Q', title='Weekly new cases per 100k'),
          y=alt.Y('Country:N', sort='-x'),
          color='WHO_region:N',
          tooltip=[
            'Country:N',
            alt.Tooltip('cases_per_100k:Q', title='Rate (/100k)', format='.1f'),
            alt.Tooltip('New_cases:Q', title='Weekly cases', format=','),
            'WHO_region:N',
          ],
        )
        .properties(height=350, title=f"Top circulation · week of {latest_label}")
      )
      col_chart.altair_chart(high_chart, use_container_width=True)
    else:
      col_chart.info("No weekly data available for the selected region(s).")

with tab_decline:
  st.subheader("Sustained declines in new cases")
  selected_regions_decline = st.multiselect(
    "Highlight declines in regions",
    options=region_options,
    default=region_options,
    key="decline_regions",
  )
  declining = negative_trends[
    (negative_trends['is_declining']) &
    (
      negative_trends['WHO_region'].isin(selected_regions_decline)
      if selected_regions_decline else True
    )
  ]
  col_heat, col_lines = st.columns((2, 1))

  if declining.empty:
    col_heat.info("No countries meet the consecutive-week decline rule for the selected regions.")
  elif pd.isna(latest_week):
    col_heat.info("Weekly aggregates unavailable for decline tracking.")
  else:
    heatmap_countries = declining.sort_values('slope').head(8)['Country'].tolist()
    heatmap_start = latest_week - pd.Timedelta(weeks=6)
    heatmap_df = country_weekly_rates[
      (country_weekly_rates['Country'].isin(heatmap_countries)) &
      (country_weekly_rates['week'] >= heatmap_start)
    ]
    heatmap = (
      alt.Chart(heatmap_df)
      .mark_rect()
      .encode(
        x=alt.X('week:T', title='Week'),
        y=alt.Y('Country:N', sort=heatmap_countries),
        color=alt.Color('cases_per_100k:Q', title='Rate (/100k)', scale=alt.Scale(scheme='blues')),
        tooltip=[
          'Country:N',
          alt.Tooltip('week:T', title='Week of'),
          alt.Tooltip('cases_per_100k:Q', title='Rate (/100k)', format='.1f'),
        ],
      )
      .properties(height=320, title="Weekly rates among declining countries")
    )
    col_heat.altair_chart(heatmap, use_container_width=True)

    line_countries = declining.sort_values('slope').head(3)['Country'].tolist()
    line_df = country_weekly_rates[
      (country_weekly_rates['Country'].isin(line_countries)) &
      (country_weekly_rates['week'] >= heatmap_start)
    ]
    lines = (
      alt.Chart(line_df)
      .mark_line(point=True)
      .encode(
        x='week:T',
        y=alt.Y('cases_per_100k:Q', title='Rate (/100k)'),
        color='Country:N',
        tooltip=['Country:N', 'week:T', alt.Tooltip('cases_per_100k:Q', format='.1f')],
      )
      .properties(height=320, title="Steepest 3 declines")
    )
    col_lines.altair_chart(lines, use_container_width=True)

with tab_stable:
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

  if post2022_ts.empty:
    chart_col.info("Post-2022 slice unavailable for CFR comparison.")
  else:
    cfr_cases_chart = (
      alt.layer(
        alt.Chart(post2022_ts)
        .mark_line(color="#457b9d")
        .encode(
          x='Date_reported:T',
          y=alt.Y('rolling30_cases:Q', title='30d avg new cases'),
          tooltip=[
            alt.Tooltip('Date_reported:T', title='Date'),
            alt.Tooltip('rolling30_cases:Q', title='30d avg cases', format=',.0f'),
          ],
        ),
        alt.Chart(post2022_ts)
        .mark_line(color="#f07c15")
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

