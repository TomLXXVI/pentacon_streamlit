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
st.header("Tijdsgrafiek logger 5")
st.subheader("Hal 10, onder lichtstraat, midden/midden")
st.markdown("rechts = kant pulsie | links = kant extractie")
st.plotly_chart(pmd.get_line_chart(Logger.MIDDLE_MIDDLE, Measurement.TEMPERATURE), use_container_width=True)
