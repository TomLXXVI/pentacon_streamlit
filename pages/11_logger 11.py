import streamlit as st
from analysis import PentaconMeasurementData, Logger, Measurement


@st.cache
def create_pentacon_measurement_data():
    pmd = PentaconMeasurementData()
    pmd.load_from_loggers(Measurement.TEMPERATURE)
    return pmd


# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PENTACON JUL-AUG '22",
    layout='wide',
    page_icon='./page_icon/logo-zwart.png'
)
# ------------------------------------------------------------------------------

pmd = create_pentacon_measurement_data()

st.title('Temperatuurmetingen juli-augustus 2022 bij Pentacon')
st.header("Tijdsgrafiek logger 11")
st.subheader("Hal 9, ruimtetemperatuur")
st.plotly_chart(pmd.get_line_chart(Logger.HALL_9, Measurement.TEMPERATURE), use_container_width=True)
