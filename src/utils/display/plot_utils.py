from plotly import graph_objects as go
import plotly.colors as pc
import numpy as np
from .common import  mean_std_1darray
import re
import plotly.io as pio
import re
from plotly import graph_objects as go
from plotly.subplots import make_subplots
# Define a custom template with your desired font settings
custom_template = pio.templates["plotly"]
custom_template.layout.font.family = "Times New Roman, serif"  # Change to Times New Roman
custom_template.layout.width = 500  # Set the default width (in pixels)
custom_template.layout.height = 375  # Set the default height (in pixels)
custom_template.layout.margin = {"t":40}

# Set the custom template as the default template
pio.templates.default = custom_template

# Set the default configuration globally
pio.renderers.default = 'notebook'
pio.renderers.default_config = {'responsive': False}

dark_colors = pc.qualitative.Dark2+pc.qualitative.Plotly

print(len(dark_colors))
dark_colors = iter(dark_colors)
name_register_for_colors={}


def error_line_arrays(x, y, std, name, error_bar_opacity=0.5, use_legend_group=False):
    if name in name_register_for_colors:
        color=name_register_for_colors[name]
    else:
        color=next(dark_colors)
        name_register_for_colors[name]=color
    if not isinstance(x,np.ndarray):
        x=np.array(x)
    if not isinstance(y,np.ndarray):
        y=np.array(y)
    if not isinstance(std,np.ndarray):
        std=np.array(std)
        
    error_band_x = np.concatenate([x, x[::-1]])
    error_band_y = np.concatenate([y - std, (y + std)[::-1]])
    
    return go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        line=dict(color=color),
        name=name,
        showlegend=True,
        legendgroup=name.split(" ")[0] if use_legend_group else None
    ), go.Scatter(
        x=error_band_x,
        y=error_band_y,
        fill='toself',
        fillcolor=color,
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        legendgroup=name.split(" ")[0] if use_legend_group else None,
        opacity=error_bar_opacity,
        hoverinfo="skip",
        name=name
    )


def plot_ece_ngrid(ece_df, n_level_df, title, skip_colors=0, use_legend_group=False):
    global dark_colors
    figures=[]
    name_register_for_colors.clear()
    dark_colors = iter(pc.qualitative.Dark2+pc.qualitative.Plotly)
    for i in range(skip_colors):
        next(dark_colors)
    for alg in ece_df.columns.get_level_values(0).unique():
        alg_name,y_mean,y_std=mean_std_1darray(ece_df,alg)
        alg_name,x_mean,x_std=mean_std_1darray(n_level_df,alg)
        
        figures.extend(error_line_arrays(x_mean, y_mean, y_std, alg_name,
                       error_bar_opacity=0.2, use_legend_group=use_legend_group))
        
    # Set the size of the fig
    fig = go.Figure(data=figures)
    
    fig.update_layout(
        xaxis_title="Size of Predictor Range",
        yaxis_title="Multicalibration Error",
        title=title
    )
    return fig


def plot_noise(mean, std):
    colors=pc.qualitative.Plotly
    # Parse the data into a more structured format
    methods_data = {}
    methods_std = {}  # New dictionary to track standard deviations

    # Regular expression to extract method and noise value
    pattern = r'(.*?) - .*noise=(\d+(.\d+)?)'

    # Process mean values
    for key, value in mean.items():
        match = re.match(pattern, key)
        if match:
            method = match.group(1)
            noise = float(match.group(2))

            if method not in methods_data:
                methods_data[method] = {}
                methods_std[method] = {}  # Initialize std dictionary too

            methods_data[method][noise] = value

            # Get corresponding std value if available
            if key in std:
                methods_std[method][noise] = std[key]
            else:
                methods_std[method][noise] = 0  # Default to 0 if not available

    # Create the plot
    fig = go.Figure()

    # Define colors for different methods

    # Add traces for each method
    for i, (method, noise_values) in enumerate(methods_data.items()):
        # Sort by noise level
        noise_levels = sorted(noise_values.keys())
        error_values = [noise_values[noise] for noise in noise_levels]
        std_values = [methods_std[method].get(
            noise, 0) for noise in noise_levels]

        # Get color (cycling through the list if needed)
        color = colors[i % len(colors)]

        # Create error band coordinates
        error_band_x = np.concatenate([noise_levels, noise_levels[::-1]])
        error_band_y = np.concatenate([
            [ev - sv for ev, sv in zip(error_values, std_values)],
            [ev + sv for ev, sv in zip(error_values[::-1], std_values[::-1])]
        ])

        # Add main line
        fig.add_trace(go.Scatter(
            x=noise_levels,
            y=error_values,
            mode='lines+markers',
            name=method,
        ))

        # Add error band
        fig.add_trace(go.Scatter(
            x=error_band_x,
            y=error_band_y,
            fill='toself',
            fillcolor=color,
            mode='none',
            showlegend=False,
            opacity=0.3,
            hoverinfo="skip",
            marker=dict(size=0),
        ))

    # Add grid lines
    # log scale
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray', type='log')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray', type='log')

    # Optional: You can use logarithmic scale if the difference is large
    # fig.update_yaxes(type='log')

    # Show the figure
    return fig
def plot_random_single(ece):
    ece_new=ece.rename(columns={"Baseline":"Baseline - noise=0","Ours":"Ours - noise=0"})
    
def plot_random(ece):
    ece_new=ece.rename(columns={"Baseline":"Baseline - noise=0","Ours":"Ours - noise=0"})
    all_figs=[]
    if ece_new.index.nlevels==1:
        df=ece_new.copy()
        mean=df.mean(0)
        std=df.std(0)
        mean.index = mean.index.droplevel(1)
        std.index = std.index.droplevel(1)
        fig=plot_noise(mean,std)
        all_figs.append(fig)
    else:
        for i in ece_new.index.get_level_values(0).unique():
            df=ece_new.loc[i]
            mean=df.mean(0)
            std=df.std(0)
            mean.index = mean.index.droplevel(1)
            std.index = std.index.droplevel(1)
            fig=plot_noise(mean,std)
            all_figs.append(fig)
    if len(all_figs)==1:
        row=1;col=1
    else:
        row=2;col=3
    fig = make_subplots(rows=row, cols=row, subplot_titles=["# Level sets: "+str(i) for i in ece_new.index.get_level_values(0).unique()],y_title="MC Error",x_title="Noise Level")
    colors=pc.qualitative.Plotly
    # Loop through your figures and add them to the subplots
    for i, subplot_fig in enumerate(all_figs):
        # Calculate the row and column
        row = i // 3 + 1  # Integer division to get row (1 or 2)
        col = i % 3 + 1   # Modulo to get column (1, 2, or 3)

        # Add each trace from the figure to the subplot
        
        for j,trace in enumerate(subplot_fig.data):
            trace.marker.color=colors[j//2]
            if i!=0:
                trace.showlegend=False
            fig.add_trace(trace, row=row, col=col)

    # Update the layout
    fig.update_layout(
        height=500,
        width=800,
    )
    fig.update_xaxes(type='log')    
    fig.update_yaxes(type='log')

    # Show the figure
    fig.show()