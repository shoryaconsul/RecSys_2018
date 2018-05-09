#%% HW5 Q3: Bokeh visualization

import pandas as pd
import numpy as np
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import CustomJS, ColumnDataSource, HoverTool, BoxZoomTool, Jitter
from bokeh.plotting import figure, output_notebook, show, output_file

dataU = pd.read_csv("CF_User_rprec.csv")
dataI = pd.read_csv("CF_Item_rprec.csv")

data = pd.DataFrame([],columns=['pid','Rprec_U','Rprec_I'])
data['pid'] = dataU['pl_ind']
data['Rprec_U'] = dataU['Rprec']
data['Rprec_I'] = dataI['Rprec']


#%%
output_file("sim_scatter.html")
source = ColumnDataSource(data) # Conversion for ease of using Bokeh features

hover1 = HoverTool(tooltips=[
    ("pid", "@pid"),
])

p1 = figure(plot_width=600, plot_height=500,tools=[hover1,'wheel_zoom','zoom_in','zoom_out','box_zoom'],title="Comparison of R-precision")
p1.scatter('Rprec_U','Rprec_I',marker="asterisk", color="red",source=source)
p1.xaxis.axis_label = "Rprec_U"
p1.yaxis.axis_label = "Rprec_I"
tab1 = Panel(child=p1, title="Scatter")

p2 = figure(plot_width=1000, plot_height=500,tools=[hover1,'wheel_zoom','zoom_in','zoom_out','box_zoom'],title="R-precision for User-based")
p2.vbar(x = data['pid'],width=None, top = data['Rprec_U'],line_color='green')
p2.xaxis.axis_label = "PID"
p2.yaxis.axis_label = "Rprec_U"
tab2 = Panel(child=p2, title="User")

p3 = figure(plot_width=1000, plot_height=500,tools=[hover1,'wheel_zoom','zoom_in','zoom_out','box_zoom'],title="R-precision for Item-based")
p3.vbar(x = data['pid'],width=None, top = data['Rprec_I'],line_color='blue')
p3.xaxis.axis_label = "PID"
p3.yaxis.axis_label = "Rprec_I"
tab3 = Panel(child=p3, title="Item")


tabs = Tabs(tabs=[ tab1, tab2, tab3 ])
#tabs = Tabs(tabs=[tab1])

show(tabs)

