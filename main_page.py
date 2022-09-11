import streamlit as st
import pandas as pd
from analysis import PentaconMeasurementData, Measurement


@st.cache
def create_pentacon_measurement_data():
    pmd = PentaconMeasurementData()
    pmd.load_from_loggers(Measurement.TEMPERATURE)
    first_date = pmd.measurements_Tdb.index[0].date()
    last_date = pmd.measurements_Tdb.index[-1].date()
    return pmd, first_date, last_date


# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PENTACON JUL-AUG '22",
    layout='wide',
    page_icon='./page_icon/logo-zwart.png'
)
# ------------------------------------------------------------------------------

pmd, first_date, last_date = create_pentacon_measurement_data()

st.title('Temperatuurmetingen juli-augustus 2022 bij Pentacon')
st.header('hal 10, loggers onder lichtstraat')

col1, col2, col3 = st.columns([2, 3, 1])

date = col1.date_input(
    label='Selecteer een datum',
    value=first_date,
    min_value=first_date,
    max_value=last_date
)

df = pmd.get_measurements_by_date(Measurement.TEMPERATURE, date)

time = col1.select_slider(
    label='Kies een tijdstip',
    options=df.index.time,
    value=df.index[0].time()
)

time_stamp = pd.Timestamp.combine(date, time)

# draw 3D bar chart at selected time_index
bar_chart = pmd.get_3D_bar_chart(time_stamp)
col2.pyplot(bar_chart.figure, clear_figure=True)

# draw table at selected time_index
df, datetime = pmd.get_table(Measurement.TEMPERATURE, time_stamp)
col1.write(datetime)
col1.dataframe(df.style.highlight_max(axis=None).format(precision=1))
col1.markdown("rechts = kant pulsie | links = kant extractie")
