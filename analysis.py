from typing import Optional, Tuple
from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass
from datetime import datetime, date
import pandas as pd
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Logger(IntEnum):  # relates logger location to logger ID
    TOP_RIGHT = 1
    MIDDLE_RIGHT = 2
    BOTTOM_RIGHT = 3
    TOP_MIDDLE = 4
    MIDDLE_MIDDLE = 5
    BOTTOM_MIDDLE = 6
    TOP_LEFT = 7
    MIDDLE_LEFT = 8
    BOTTOM_LEFT = 9
    HALL_9 = 11
    HALL_8 = 10
    HALL_7 = 12


class Measurement(IntEnum):
    TEMPERATURE = 1  # column index in csv logger file
    RH = 2


class MeasurementSeries:
    """Wrapper class around a Pandas `Series` object."""

    def __init__(self, pandas_series: pd.Series):
        self._pandas_series = pandas_series

    def __getitem__(self, logger: Logger) -> float:
        """Returns the measurement of the specified logger (specified by
         a member of `IntEnum`-class `Logger`)"""
        return self._pandas_series[logger.name]

    def __str__(self) -> str:
        """Returns the string representation of the `MeasurementSeries`-object."""
        return str(self._pandas_series)

    def get_datetime(self) -> np.datetime64:
        """Returns the date and time of the measurement as Numpy `datetime64`-
        object."""
        # noinspection PyUnresolvedReferences
        return self._pandas_series.name.to_datetime64()


@dataclass
class Params3DBarChart:
    x_arr: np.ndarray
    y_arr: np.ndarray
    z_arr: np.ndarray
    z_ref_arr: np.ndarray
    bar_width: float = 1.0
    bar_depth: float = 1.0


class PentaconMeasurementData:

    def __init__(self):
        # path to folder with logger files
        self.logger_folder = Path("./data")
        # list of logger files
        self.logger_files = [self.logger_folder / f"logger_{i}.csv" for i in range(1, 13)]

        self.measurements_Tdb: Optional[pd.DataFrame] = None
        self.measurements_RH: Optional[pd.DataFrame] = None

    def load_from_logger(self, logger: Logger, measurement: Measurement) -> pd.DataFrame:
        """Reads asked logger measurement data from file and returns it in a Pandas `DataFrame` object.
        The data frame has two columns: 'Date' (date and time of each single measurement) and the column
        with the asked measurement data: 'T[°C]' if parameter `measurement` is `Measurement.TEMPERATURE`
        or 'RH[rH%]' if parameter `measurement` is `Measurement.RH`."""
        df = pd.read_csv(
            filepath_or_buffer=self.logger_files[logger - 1],  # logger list index = logger ID - 1
            sep=';',
            usecols=[1, 2, 3],
            decimal=',',
            parse_dates=[0],
            infer_datetime_format=True,
            dayfirst=True
        )
        return df.iloc[:, [0, measurement]]

    def load_from_loggers(self, which: Measurement) -> pd.DataFrame:
        """Groups the asked measurement data (specified by `which`) of all loggers specified in the `IntEnum`-class
        `Logger` in a single Pandas `DataFrame` object. The index of this data frame is a `DateTimeIndex` (the index
        values are Pandas `TimeStamp`-objects). The columns are labeled with the name of the loggers taken from the
        `IntEnum`-class `Logger` ('TOP_LEFT', 'MIDDLE_RIGHT', etc.). The data frame is assigned to object attribute
        `self.measurements_Tdb` if `which` is `Measurement.TEMPERATURE` or assigned to object attribute
        `self.measurements_RH` if `which` is `Measurement.RH`."""
        d = {}
        for i, logger in enumerate(Logger):
            if i == 0:
                df = self.load_from_logger(logger, which)
                d['date_time'] = df.iloc[:, 0]
                d[logger.name] = df.iloc[:, 1]
            else:
                d[logger.name] = self.load_from_logger(logger, which).iloc[:, 1]
        if which == Measurement.TEMPERATURE:
            self.measurements_Tdb = pd.DataFrame(d)
            self.measurements_Tdb.set_index('date_time', inplace=True)
            return self.measurements_Tdb
        if which == Measurement.RH:
            self.measurements_RH = pd.DataFrame(d)
            self.measurements_RH.set_index('date_time', inplace=True)
            return self.measurements_RH

    def get_measurement_series(self, which: Measurement, time_stamp: datetime) -> MeasurementSeries:
        """Get the asked measurement data at the given `time_stamp`. The measurement data is returned in a
        `MeasurementSeries`-object."""
        if which == Measurement.TEMPERATURE:
            if self.measurements_Tdb is None:
                self.load_from_loggers(Measurement.TEMPERATURE)
            return MeasurementSeries(self.measurements_Tdb.loc[time_stamp, :])
        if which == Measurement.RH:
            if self.measurements_RH is None:
                self.load_from_loggers(Measurement.RH)
            return MeasurementSeries(self.measurements_RH.loc[time_stamp, :])

    @staticmethod
    def _get_params_3D_bar_chart(series: MeasurementSeries) -> Params3DBarChart:
        """Determines the parameters that are needed to draw a 3D bar chart with
        the measurement data in the given `MeasurementSeries`-object `series`.
        This is a protected method, used internally by public method
        `get_3D_bar_chart(...)`."""
        _x_arr = np.arange(3)
        _y_arr = np.arange(3)
        _xx_arr, _yy_arr = np.meshgrid(_x_arr, _y_arr)
        x_arr, y_arr = _xx_arr.ravel(), _yy_arr.ravel()
        z_arr = np.array([
            series[Logger.BOTTOM_LEFT],     # position (0, 0)
            series[Logger.BOTTOM_MIDDLE],   # position (1, 0)
            series[Logger.BOTTOM_RIGHT],    # position (2, 0)
            series[Logger.MIDDLE_LEFT],     # position (0, 1)
            series[Logger.MIDDLE_MIDDLE],   # position (1, 1)
            series[Logger.MIDDLE_RIGHT],    # position (2, 1)
            series[Logger.TOP_LEFT],        # position (0, 2)
            series[Logger.TOP_MIDDLE],      # position (1, 2)
            series[Logger.TOP_RIGHT]        # position (2, 2)
        ])
        z_ref_arr = np.zeros_like(z_arr)
        bar_width = 0.2
        bar_depth = 0.2
        return Params3DBarChart(x_arr, y_arr, z_arr, z_ref_arr, bar_width, bar_depth)

    def get_3D_bar_chart(self, time_stamp: datetime, **kwargs) -> 'BarChart3D':
        """Taken `time_stamp`, get a Matplotlib `BarChart3D`-object of the
        measurements taken by all the loggers specified in class `Logger`.
        Valid `**kwargs`-arguments` are `fig_size` and `z_label`. `fig_size` must
        be assigned a tuple of ints, being the height and the width of the figure.
        `z_label` must be assigned a string, being the label for the Z-axis
        (i.e. the measurement value axis). To access the figure-object of the
        BarChart3D`-object, use the `figure` attribute of this object.
        """
        series = self.get_measurement_series(Measurement.TEMPERATURE, time_stamp)
        bar_chart_params = self._get_params_3D_bar_chart(series)
        fig_size = kwargs.get('fig_size', (6, 6))
        z_label = kwargs.get('z_label', 'temperatuur, °C')
        bar_chart = BarChart3D(bar_chart_params, fig_size=fig_size, z_label=z_label)
        return bar_chart

    def get_line_chart(self, logger: Logger, measurement: Measurement, **kwargs):
        """Get a Plotly line chart figure showing the asked measurements of `logger`"""
        df = self.load_from_logger(logger, measurement)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.iloc[:, 0],  # date-time axis
                y=df.iloc[:, 1],  # measurement value axis
                name=logger.name,
                line={'color': kwargs.get('line_color', 'firebrick')}
            )
        )
        fig.update_layout(
            xaxis_title='tijdsas',
            yaxis_title=kwargs.get('y_label', 'temperatuur, °C'),
            height=600
        )
        return fig

    def get_table(self, which: Measurement, time_stamp: datetime) -> Tuple[pd.DataFrame, str]:
        """Returns a table (Pandas DataFrame-object) of the measurements at given `time_stamp`
        of all the loggers defined in class `Logger`. The position of a measurement in the table
        corresponds with the spatial position of its logger. Also returns the date-time as string
        that corresponds with the given time index."""
        series = self.get_measurement_series(which, time_stamp)
        index = ['boven', 'midden', 'onder']
        columns = ['links', 'midden', 'rechts']
        data = np.array([
            [series[Logger.TOP_LEFT], series[Logger.TOP_MIDDLE], series[Logger.TOP_RIGHT]],
            [series[Logger.MIDDLE_LEFT], series[Logger.MIDDLE_MIDDLE], series[Logger.MIDDLE_RIGHT]],
            [series[Logger.BOTTOM_LEFT], series[Logger.BOTTOM_MIDDLE], series[Logger.BOTTOM_RIGHT]]
        ])
        df = pd.DataFrame(data=data, index=index, columns=columns)
        date_time = str(series.get_datetime())
        return df, date_time

    def get_measurements_by_date(self, which: Measurement, date: date) -> pd.DataFrame:
        """Returns a Pandas DataFrame-object of the asked measurement data (specified by `which`)
        on the given date `date` of all the loggers defined in class `Logger`."""
        if which == Measurement.TEMPERATURE:
            if self.measurements_Tdb is None:
                self.load_from_loggers(Measurement.TEMPERATURE)
            # in dataframe self.measurements_Tdb find all measurements made on given date
            # noinspection PyUnresolvedReferences
            is_date = self.measurements_Tdb.index.date == date
            df = self.measurements_Tdb[is_date]
            return df
        if which == Measurement.RH:
            if self.measurements_RH is None:
                self.load_from_loggers(Measurement.RH)
            # in dataframe self.measurements_RH find all measurements made on given date
            # noinspection PyUnresolvedReferences
            is_date = self.measurements_RH.index.date == date
            df = self.measurements_RH[is_date]
            return df


class BarChart3D:

    def __init__(self, params: Params3DBarChart, **kwargs):
        self.params = params
        self.fig_size = kwargs.get('fig_size')
        self.z_label = kwargs.get('z_label')
        self.figure = self._create_chart()

    def _create_bar_colors(self):
        color_map = cm.get_cmap('coolwarm')
        norm = Normalize(
            vmin=min(self.params.z_arr),
            vmax=max(self.params.z_arr)
        )
        colors = color_map(norm(self.params.z_arr))
        return colors

    def _create_chart(self):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(projection='3d')
        colors = self._create_bar_colors()
        ax.bar3d(
            self.params.x_arr,
            self.params.y_arr,
            self.params.z_ref_arr,
            self.params.bar_width,
            self.params.bar_depth,
            self.params.z_arr,
            color=colors
        )
        ax.set_xlabel('horizontale richting')
        ax.set_ylabel('verticale richting')
        ax.set_zlabel(self.z_label)
        x_ticks = [0, 1, 2]
        y_ticks = [0, 1, 2]
        z_ticks = np.arange(0, 50, 5)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)
        return fig

    @staticmethod
    def show():
        plt.show()
