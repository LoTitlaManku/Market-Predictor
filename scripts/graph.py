
from __future__ import annotations

# Standard library imports
from datetime import timedelta
from typing import TYPE_CHECKING

# External library imports
import finplot as fplt
import numpy as np
import pandas as pd
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

# Custom imports
from scripts.data_management import load_data

# Type hinting imports
if TYPE_CHECKING:
    from scripts.default_gui import MainWindow

############################################################################

# Class for graph object
class StockGraph:
    def __init__(self, parent: MainWindow):
        # Define dictionaries for graph colours, and initialise other variables
        self.parent = parent
        self.line_colours = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#17b4cf"]
        self.candle_colours = [
            {"bull": "#00ff00", "bear": "#ff0000"},
            {"bull": "#ffff00", "bear": "#9467bd"},
            {"bull": "#3486eb", "bear": "#2b2b2b"},
            {"bull": "#f76fe5", "bear": "#f5b318"},
            {"bull": "#7df6ff", "bear": "#7800a3"}
        ]
        self.loaded = {}
        self.selected_type = "Line"
        self.resolution = "1d"
        self.saved_view = None

        # Create the graph and edit its visuals
        self.ax = fplt.create_plot(title="Stocks", init_zoom_periods=200)
        self.ax.showGrid(x=True, alpha=0.2)
        self.ax.showAxis('left'); self.ax.hideAxis('right')

    # Helper function to recreate a new graph
    def rebuild_self(self) -> None:
        # Delete current axis object
        self.ax.vb.win.setParent(None)
        del self.ax
        QApplication.processEvents()

        # Create a new graph widget
        self.ax = fplt.create_plot(title="Stocks", init_zoom_periods=200)
        self.ax.showGrid(x=True, alpha=0.2); self.ax.showAxis('left'); self.ax.hideAxis('right')

        # Iterate through every loaded stock, and re-add them to the graph
        for ticker, info in self.loaded.items():
            df = info.get(f"{self.resolution}_df", None)
            if df is None:
                df = load_data(ticker, self.resolution)
                self.loaded[ticker][f"{self.resolution}df"] = df
            colour_index = info['colour_index']
            line_colour, candle_colour = self.line_colours[colour_index], self.candle_colours[colour_index]

            # Create the graphs
            line_plot = fplt.plot(df["Close"], ax=self.ax, color=line_colour, width=2, legend=None)
            candle_items = fplt.candlestick_ochl(df[["Open", "Close", "High", "Low"]], ax=self.ax, candle_width=0.6)
            candle_items.colors.update({
                'bull_body': candle_colour['bull'],
                'bull_shadow': candle_colour['bull'],
                'bear_body': candle_colour['bear'],
                'bear_shadow': candle_colour['bear']
            })

            # Update the dictionary with new plot
            info['Candle'] = candle_items if isinstance(candle_items, list) else [candle_items]
            info['Line'] = line_plot if isinstance(line_plot, list) else [line_plot]

            # Hide not-selected
            if self.selected_type == "Line":
                for candle in info['Candle']: candle.hide()
            else: line_plot.hide()

        self.update_keys_html()
        fplt.refresh()

    # Helper function to switch between candlestick and line graph types
    def switch_graph_type(self) -> None:
        # Save current viewing range to maintain same position
        try:
            vr = self.ax.vb.viewRange()
            self.saved_view = ([float(vr[0][0]), float(vr[0][1])], [float(vr[1][0]), float(vr[1][1])])
        except: self.saved_view = None

        self.selected_type = self.parent.type_dropdown.currentText()

        # Hide all items not part of selected graph type
        for info in self.loaded.values():
            for point in info.get("Line", []):
                try:
                    if self.selected_type == "Line": point.show()
                    else: point.hide()
                except: pass

            for candle in info.get("Candle", []):
                try:
                    if self.selected_type == "Candle": candle.show()
                    else: candle.hide()
                except: pass

        fplt.refresh()
        self.update_keys_html()

        # Try to restore the previous view
        if not self.saved_view: return

        xr, yr = self.saved_view
        try: self.ax.vb.setRange(xRange=xr, yRange=yr, padding=0)
        except: pass

    # Helper function to switch time-period for plotted data
    def switch_graph_resolution(self, res: str) -> None:
        # Get current resolution
        self.resolution = res
        self.parent.res_dropdown.blockSignals(True)
        # Update to new resolution
        index = self.parent.res_dropdown.findText(res)
        self.parent.res_dropdown.setCurrentIndex(index)
        # Rebuild graph on new resolution
        self.parent.res_dropdown.blockSignals(False)
        self.parent.rebuild_graph()

    # Helper function to add a stock to the graph
    def add_ticker(self, ticker: str, replace: bool = False) -> str:
        if ticker in self.loaded: return f"{ticker} already added"

        # Check if new ticker takes priority on graph
        if len(self.loaded) >= 5:
            if replace: self.remove_ticker(self.loaded.popitem()[0])
            else: return "Cannot load more than 3 tickers"

        # Ensure cache is up-to-date for that ticker
        self.parent.updater.prioritize(ticker)
        QApplication.processEvents()

        # Load the data and check that it's valid
        data = load_data(ticker, self.resolution)
        if data is None or data.empty: return "No data or invalid ticker"

        # Get the colour the stock should be drawn as
        used_colours = {info["colour_index"] for info in self.loaded.values()}
        colour_index = next((i for i in range(len(self.line_colours)) if i not in used_colours), 0)
        line_colour, candle_color = self.line_colours[colour_index], self.candle_colours[colour_index]

        # Create both the candle and line versions of the graphs
        line_plot = fplt.plot(data["Close"], ax=self.ax, color=line_colour, width=2, legend=None)
        candle_items = fplt.candlestick_ochl(data[["Open", "Close", "High", "Low"]], ax=self.ax, candle_width=0.6)

        # Set colour of the candlesticks
        fplt.candle_bear_color, fplt.candle_bull_color = None, None
        candle_items.colors.update({
            'bull_body': candle_color["bull"],
            'bull_shadow': candle_color["bull"],
            'bear_body': candle_color["bear"],
            'bear_shadow': candle_color["bear"]
        })

        # Update the dictionary of loaded tickers
        self.loaded[ticker] = {
            f"{self.resolution}df": data,
            "Line": line_plot if isinstance(line_plot, list) else [line_plot],
            "Candle": candle_items if isinstance(candle_items, list) else [candle_items],
            "colour_index": colour_index
        }
        self.parent.ticker_list_widget.addItem(ticker)

        # Hide which type is not selected
        if self.selected_type == "Line":
            for candle in self.loaded[ticker]['Candle']: candle.hide()
        else: line_plot.hide()

        # Update key and restore view
        self.update_keys_html()
        if len(self.loaded) == 1:
            self.ax.vb.setRange(xRange=[0, 100000], yRange=[0,max(data["Close"])], padding=0)
        else: self.ax.vb.autoRange()

        return "success"

    # Helper function to plot result of predictor
    def add_future(self, ticker: str, interval: str, forecast_results: dict) -> None:
        # Set graph settings to match prediction settings
        if self.resolution != interval: self.switch_graph_resolution(interval)
        else: self.parent.rebuild_graph()
        if ticker not in self.loaded.keys(): self.add_ticker(ticker, replace=True)

        # Get data and information
        real_data = self.loaded[ticker][f"{self.resolution}df"]
        last_trade_date = real_data.index[-1]
        period = interval[1]
        delta_type = "hours" if period == "h" else "days"
        current_price = float(real_data["Close"].iloc[-1])

        # Setup data to show "future" by 30 days
        future_dates = pd.date_range(start=last_trade_date + timedelta(**{delta_type: 1}), periods=30, freq=period)
        full_data = pd.concat([real_data, pd.DataFrame(np.nan, index=future_dates, columns=real_data.columns)])
        forecast_dates = [last_trade_date] + [forecast_results[d]['target_date'] for d in [1,5,21]]

        # Turn prediction dots into smooth line
        def create_path(key):
            forecast_prices = [current_price] + [forecast_results[d][key] for d in [1,5,21]]
            path_series = pd.Series(forecast_prices, index=forecast_dates)
            return path_series.reindex(full_data.index).interpolate(method="linear").dropna()

        tline_mid, tline_up, tline_lo = create_path('price'), create_path('up'), create_path('lo')

        # Helper to shade areas between lines
        def paint_uncertain_zone(start_date, end_date, colour):
            s_up = tline_up.loc[start_date:end_date]
            if len(s_up) > 1:
                upper_anchor = fplt.plot(s_up, width=0)
                lower_anchor = fplt.plot(tline_lo.loc[start_date:end_date], width=0)
                fill_colour = QColor(colour)
                fill_colour.setAlphaF(0.2)
                fplt.fill_between(upper_anchor, lower_anchor, color=fill_colour)

        # Paint each horizon region for uncertainty
        paint_uncertain_zone(last_trade_date, forecast_results[1]['target_date'], '#00ff88')
        paint_uncertain_zone(forecast_results[1]['target_date'], forecast_results[5]['target_date'], '#0099ff')
        paint_uncertain_zone(forecast_results[5]['target_date'], forecast_results[21]['target_date'], '#ffcc00')

        # Plot outlines and actual prediction
        fplt.plot(tline_up, ax=self.ax, color='#bbbbbb', width=0.5)
        fplt.plot(tline_lo, ax=self.ax, color='#bbbbbb', width=0.5)
        fplt.plot(tline_mid, ax=self.ax, color='#000000', style='--', width=2)

    # Helper function to remove a stock from graph
    def remove_ticker(self, ticker: str) -> str:
        # Validation on ticker
        if not ticker: return "No ticker selected"
        if ticker not in self.loaded: return "Ticker not loaded"

        # Delete the entry and rebuild the graph without it
        del self.loaded[ticker]
        self.parent.rebuild_graph()

        list_index = self.parent.ticker_list_widget.findText(ticker)
        self.parent.ticker_list_widget.removeItem(list_index)
        return "Success"

    # Helper function to update colour keys for graph
    def update_keys_html(self) -> None:
        if not self.loaded: self.parent.keys_label.setText(""); return

        # Iterate through every loaded stock and add its colour code to a html format string
        parts = []
        for ticker, info in self.loaded.items():
            if self.selected_type == "Line":
                # If line type, single colour key
                line_colour = self.line_colours[info["colour_index"]]
                parts.append(f"<span style='display:inline-block; padding:2px 6px; background:{line_colour};"
                             f"color:#fff; border-radius:3px; margin-right:6px;'>{ticker}</span>")
            elif self.selected_type == "Candle":
                # If candle type, double colour key for close gain and close loss
                candle_colour = self.candle_colours[info.get("colour_index")]
                parts.append(f"""
                <span style="display:inline-block; padding:2px 6px; color:#000000;
                             border-radius:3px; margin-right:6px;">{ticker}</span>
                <span style="display:inline-block; padding:2px 6px; background:{candle_colour['bull']}; 
                             color:{candle_colour['bull']}; border-radius:3px; margin-right:6px;">---</span>
                <span style="display:inline-block; padding:2px 6px; background:{candle_colour['bear']}; 
                             color:{candle_colour['bear']}; border-radius:3px; margin-right:6px;">---</span>
                """)

        # Set the label to the html style
        keys_html = '<div style="text-align: right;">' + " ".join(parts) + "</div>"
        self.parent.keys_label.setText(keys_html)

