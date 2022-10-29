from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from csv import writer
import random
import os

# 5 constant objects

# All the PATHs below should be given relative to the location of this script.
# Required csv-file format: column # 0 - time column where time in datetime format is located,
#                           column # 1 - integer/float numbers allowed,
#                           column # 2 - integer/float numbers allowed, ...
DATABASES = (
    'data_dummy/dummy_csv.csv',  # PATH to online updated DB ('Количество корма на кормовом столе (онлайн)')
    'data_dummy/dummy_csv_2.csv',  # PATH to DB 'Кормовое поведение в разные дни'
    'data_dummy/dummy_csv_3.csv'  # PATH to DB 'Время подачи корма, остатки по дням в разные дни'
)

# column names that should be used as index column (column where date-time objects are) for the corresponding DB above
INDEX_COLUMN_NAMES = (
    'time',
    'time',
    'time'
)

TIME_LABEL_VS_TIME_DELTA_ITEMS = {
    '15 мин': timedelta(minutes=15),
    '30 мин': timedelta(minutes=30),
    '1 ч': timedelta(minutes=60),
    '2 ч': timedelta(minutes=2 * 60),
    '6 ч': timedelta(minutes=6 * 60),
    '12 ч': timedelta(minutes=12 * 60),
    '24 ч': timedelta(minutes=24 * 60)
}

FOOD_TABLE_ITEMS = ('  №1 + №2  ', '  №1  ', '  №2  ')

VIRTUAL_LINE_ITEMS = ('  да  ', '  нет  ')


app = Dash(__name__)

app.layout = html.Div([
    # html.Div('One-sided slider:'),
    # dcc.Slider(
    #     id='time_slider',
    #     min=0,
    #     max=len(df.index) - 1,
    #     step=None,
    #     value=len(df.index) - 1,
    #     marks={ind: val.strftime('%d/%m/%Y %H:%M') for ind, val in enumerate(df.index)},
    # ),

    # html.Div('Two-sided slider:'),
    # dcc.RangeSlider(
    #     id='time_two-sided_slider',
    #     min=0,
    #     max=len(df.index)-1,
    #     step=None,
    #     value=[0, len(df.index)-1],
    #     marks={ind: val.strftime('%d/%m/%Y %H:%M') for ind, val in enumerate(df.index)},
    # ),

    html.Div([
        html.Div(html.H2('Количество корма на кормовом столе (онлайн)'),
                 style={'text-align': 'center'}),

        html.Div([

            html.Div([
                html.Div('Последние:'),

                dcc.Dropdown(
                    list(TIME_LABEL_VS_TIME_DELTA_ITEMS.keys()),
                    list(TIME_LABEL_VS_TIME_DELTA_ITEMS.keys())[0],
                    clearable=False,
                    style={'width': '65%'},
                    id='id_window_size'),
            ],
                style={'width': '30%', 'display': 'inline-block', 'margin': 'auto', 'vertical-align': 'top'}
            ),

            html.Div([
                html.Div('Кормовой стол:'),

                dcc.RadioItems(
                    FOOD_TABLE_ITEMS,
                    FOOD_TABLE_ITEMS[0],
                    inline=True,
                    id='id_radio_table'),
            ],
                style={'width': '40%', 'display': 'inline-block', 'margin': 'auto', 'vertical-align': 'top'}
            ),

            html.Div([
                html.Div('Учёт виртуальной линии:'),

                dcc.RadioItems(
                    VIRTUAL_LINE_ITEMS,
                    VIRTUAL_LINE_ITEMS[1],
                    inline=True,
                    id='id_radio_virtual_line'),
            ],
                style={'width': '30%', 'display': 'inline-block', 'margin': 'auto', 'vertical-align': 'top'}
            ),
        ],
            style={'width': '60%', 'margin': 'auto'}
        ),


        html.Div([
            dcc.Interval(
                interval=5 * 1000,  # in milliseconds
                n_intervals=0,
                id='id_interval'),

            dcc.Graph(
                id='id_food_vs_time_graph_live',
                style={'height': '45vh'}
            ),
        ]),
    ],
        style={'width': '80%', 'margin': 'auto'}
    ),

    html.Div([
        html.Div([
            html.H2('Кормовое поведение в разные дни (непрерывный промежуток):'),

            html.Div('Начало/конец промежутка:'),

            dcc.DatePickerRange(
                min_date_allowed=date(2022, 1, 1),  # date is in format 'year,month,day'
                max_date_allowed=datetime.now().date(),
                initial_visible_month=date(2022, 1, 1),
                start_date=date(2022, 1, 3),
                end_date=date(2022, 1, 7),  # datetime.now().date()
                display_format='DD.MM.YYYY',
                id='id_date_picker_range'
            ),

            dcc.Graph(
                id='id_food_vs_time_graph_over_certain_period',
                style={'height': '45vh'}
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([
            html.H2('Кормовое поведение в отдельные дни:'),

            html.Div('День (добавить/удалить):'),

            dcc.DatePickerSingle(
                min_date_allowed=date(2022, 1, 1),  # date is in format 'year,month,day'
                max_date_allowed=datetime.now().date(),
                initial_visible_month=date(2022, 1, 1),
                date=date(2022, 1, 1),
                display_format='DD.MM.YYYY',
                id='id_date_picker_single'
            ),

            dcc.Graph(
                id='id_food_vs_time_graph_individual_days',
                style={'height': '45vh'}
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}
        ),
    ]),

    html.Div([
        html.Div([
            html.H2('Время подачи корма в разные дни:'),

            dcc.Graph(
                id='id_time_of_food_delivery_vs_day_graph',
                style={'height': '45vh'}
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([
            html.H2('Остатки по дням после каждой кормёжки:'),

            dcc.Graph(
                id='id_food_left_vs_day_graph',
                style={'height': '45vh'}
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}
        ),
    ]),
])


# updates graph 'Количество корма на кормовом столе (онлайн)' if any of the 3 input variables changes
@app.callback(
    Output('id_food_vs_time_graph_live', 'figure'),
    Input('id_interval', 'n_intervals'),
    Input('id_window_size', 'value'),
    Input('id_radio_table', 'value')
)
def update_metrics(_, window_size, table_number):

    if window_size is None:
        return

    df = pd.read_csv(DATABASES[0], index_col=INDEX_COLUMN_NAMES[0], parse_dates=True)
    filtered_df = df[df.index[-1] - TIME_LABEL_VS_TIME_DELTA_ITEMS[window_size]:]

    y = None
    if table_number == FOOD_TABLE_ITEMS[1]:
        y = filtered_df.columns[0]
    elif table_number == FOOD_TABLE_ITEMS[2]:
        y = filtered_df.columns[1]
    else:
        filtered_df[filtered_df.columns[0] + filtered_df.columns[1]] = filtered_df[filtered_df.columns[0]] + \
                                                                       filtered_df[filtered_df.columns[1]]
        y = filtered_df.columns[2]

    fig = px.line(filtered_df, x=filtered_df.index, y=y)
    fig.update_layout(
        transition_duration=500,  # smooth update of graph
        xaxis_title='Время',
        yaxis_title='Кол-во корма'
    )
    return fig


# updates graph 'Кормовое поведение в разные дни (непрерывный промежуток)' if range of dates changes
@app.callback(
    Output('id_food_vs_time_graph_over_certain_period', 'figure'),
    Input('id_date_picker_range', 'start_date'),
    Input('id_date_picker_range', 'end_date')
)
def update_metrics(start_date, end_date):

    if start_date is None or end_date is None:
        return

    df = pd.read_csv(DATABASES[1], index_col=INDEX_COLUMN_NAMES[1], parse_dates=True)

    # adding curves for chosen dates
    fig = make_subplots(rows=1, cols=1)
    cur_date = date.fromisoformat(start_date)
    for i in range((date.fromisoformat(end_date) - cur_date).days):
        if cur_date > df.index[-1]:  # current date is greater than the date of the last row of the DB
            break
        filtered_df = df.loc[cur_date.strftime('%m/%d/%Y')]
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index.hour,
                y=filtered_df[filtered_df.columns[0]],
                name=cur_date.strftime('%d.%m.%Y'),
                line=dict(dash='dash' if i % 3 == 2 else 'dot' if i % 3 == 1 else 'solid')
            ),
            row=1, col=1
        )
        cur_date += timedelta(days=1)

    fig.update_layout(
        transition_duration=500,  # smooth update of graph
        xaxis_title='Время',
        yaxis_title='Кол-во корма'
    )
    return fig


# variable needed to keep track of dates have been added(/removed) to(/from) the below graph
dates = []


# updates graph 'Кормовое поведение в отдельные дни' if new date is added/deleted
@app.callback(
    Output('id_food_vs_time_graph_individual_days', 'figure'),
    Input('id_date_picker_single', 'date')
)
def update_metrics(input_date):

    if input_date is None:
        return

    df = pd.read_csv(DATABASES[1], index_col=INDEX_COLUMN_NAMES[1], parse_dates=True)
    fig = make_subplots(rows=1, cols=1)
    cur_date = date.fromisoformat(input_date)

    if df.index[0] <= cur_date <= df.index[-1]:
        if cur_date not in dates:
            dates.append(cur_date)
        else:
            dates.remove(cur_date)

    # adding curves for chosen dates
    for i, i_date in enumerate(dates):
        filtered_df = df.loc[i_date.strftime('%m/%d/%Y')]
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index.hour,
                y=filtered_df[filtered_df.columns[0]],
                name=i_date.strftime('%d.%m.%Y'),
                line=dict(dash='dash' if i % 3 == 2 else 'dot' if i % 3 == 1 else 'solid')
            ),
            row=1, col=1
        )

    fig.update_layout(
        transition_duration=500,  # smooth update of graph
        xaxis_title='Время',
        yaxis_title='Кол-во корма'
    )
    return fig


# updates graph 'Время подачи корма в разные дни' if range of dates changes
@app.callback(
    Output('id_time_of_food_delivery_vs_day_graph', 'figure'),
    Input('id_date_picker_range', 'start_date'),
    Input('id_date_picker_range', 'end_date')
)
def update_metrics(start_date, end_date):

    if start_date is None or end_date is None:
        return

    df = pd.read_csv(DATABASES[2], index_col=INDEX_COLUMN_NAMES[2], parse_dates=True)
    filtered_df = df.loc[date.fromisoformat(start_date).strftime('%m/%d/%Y'):
                         date.fromisoformat(end_date).strftime('%m/%d/%Y')]
    filtered_df['feed_time'] = filtered_df.index.time
    filtered_df['feed_time_in_minutes'] = filtered_df['feed_time'].apply(lambda x: x.hour + x.minute * 60)

    fig = go.Figure([go.Bar(x=filtered_df.index.date,
                            y=filtered_df['feed_time_in_minutes'],
                            name='Время раздачи корма')])

    # change tick labels from minutes to full time (hours:minutes:seconds)
    fig.update_layout(yaxis=dict(tickmode='array',
                                 tickvals=filtered_df['feed_time_in_minutes'],
                                 ticktext=filtered_df['feed_time']))

    # adding three lines to the graph: min, mean, max
    for i, param in enumerate((('min', 'Самое раннее'), ('mean', 'Среднее'), ('max', 'Самое позднее'))):
        # getattr('instance of class', 'class method') - get access to the corresponding method ('min', 'mean', 'max')
        fig.add_trace(go.Scatter(
            x=[filtered_df.index[0].date(),
               filtered_df.index[-1].date()],
            y=[getattr(filtered_df['feed_time_in_minutes'], param[0])(),
               getattr(filtered_df['feed_time_in_minutes'], param[0])()],
            mode='lines',
            name=param[1],
            line=dict(color='Green' if i % 2 else 'Red',
                      width=2,
                      dash='dash' if i % 2 else 'solid')
        ))

    fig.update_layout(
        transition_duration=500,  # smooth update of graph
        xaxis_title='День',
        yaxis_title='Время подачи корма'
    )
    return fig


# updates graph 'Остатки по дням после каждой кормёжки' if range of dates changes
@app.callback(
    Output('id_food_left_vs_day_graph', 'figure'),
    Input('id_date_picker_range', 'start_date'),
    Input('id_date_picker_range', 'end_date')
)
def update_metrics(start_date, end_date):

    if start_date is None or end_date is None:
        return

    df = pd.read_csv(DATABASES[2], index_col=INDEX_COLUMN_NAMES[2], parse_dates=True)
    filtered_df = df.loc[date.fromisoformat(start_date).strftime('%m/%d/%Y'):
                         date.fromisoformat(end_date).strftime('%m/%d/%Y')]

    fig = go.Figure([go.Bar(x=filtered_df.index.date,
                            y=filtered_df['some_number_1'],
                            name='Остатки корма')])

    # adding three lines to the graph: min, mean, max
    for i, param in enumerate((('min', 'Минимальные'), ('mean', 'Средние'), ('max', 'Максимальные'))):
        # getattr('instance of class', 'class method') - get access to the corresponding method ('min', 'mean', 'max')
        fig.add_trace(go.Scatter(
            x=[filtered_df.index[0].date(),
               filtered_df.index[-1].date()],
            y=[getattr(filtered_df[filtered_df.columns[0]], param[0])(),
               getattr(filtered_df[filtered_df.columns[0]], param[0])()],
            mode='lines',
            name=param[1],
            line=dict(color='Green' if i % 2 else 'Red',
                      width=2,
                      dash='dash' if i % 2 else 'solid')
        ))

    fig.update_layout(
        transition_duration=500,
        xaxis_title='День',
        yaxis_title='% оставшегося корма'
    )  # smooth update of graph
    return fig


if __name__ == '__main__':
    # app.run_server(debug=True, dev_tools_props_check=False)
    app.run_server(debug=True)

