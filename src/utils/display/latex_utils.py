import pandas as pd
import numpy as np
from functools import partial
import re

from .common import average_folds
def pm(mean,std):
    text_df= mean.astype(str)+" ± "+std.astype(str)
    return text_df.applymap(lambda x: x.replace("nan ± nan",""))
    

def style_highlight(df_with_folds, transpose=True, overall_best_format="font-weight:bold", method_best_format="font-style:italic", multiplier=1000):
    df_with_folds=df_with_folds.copy()
    df_with_folds.columns.names=["Method","$m$ at Cal"]
    ece_df_mean, ece_df_std = average_folds(df_with_folds*multiplier)
    # ece_df_mean.index=pd.MultiIndex.from_tuples(product(["# Test Grids"],ece_df_mean.index))
    # ece_df_std.index=pd.MultiIndex.from_tuples(product(["# Test Grids"],ece_df_std.index))
    ece_df_mean.index.name="$m$ at Test"
    ece_df_std.index.name="$m$ at Test"
    if transpose:
        ece_df_mean=ece_df_mean.T
        ece_df_std=ece_df_std.T

    data=pm(ece_df_mean,ece_df_std)
    
    # def highlight_min(mean, std, color='yellow'):
    # Function to highlight min values in yellow
    axis=0 if transpose else 1
    def highlight_min(data,axis, attr):
        mean_values = data.applymap(lambda x: float(x.split(' ± ')[0]) if x!="" else np.inf)
        min_mean=mean_values.min(axis=axis).to_numpy()
        min_mean=min_mean[None,:] if axis==0 else min_mean[:,None]
        is_min = mean_values.to_numpy() == min_mean
        return pd.DataFrame(np.where(is_min, attr, ''), index=data.index, columns=data.columns)

    def highlight_min_level0(data,axis,  attr):
        mean_values = data.applymap(lambda x: float(x.split(' ± ')[0]) if x!="" else np.inf)
        
        level_0_direction=data.columns if axis==1 else data.index
        # Check the number of sub-columns for each level-0 column
        level0_counts = level_0_direction.get_level_values(0).value_counts()
        
        # Create a mask for highlighting
        is_min = pd.DataFrame(False, index=data.index, columns=data.columns)
        
        for level0 in level0_counts.index:
            if level0_counts[level0] > 1:
                if axis==1:
                    sub_columns = mean_values.loc[:, level0]
                    min_mask = sub_columns == sub_columns.min(axis=1).to_numpy()[:, None]
                    is_min.loc[:, level0] = min_mask.to_numpy()
                else:
                    sub_columns = mean_values.loc[level0]
                    min_mask = sub_columns == sub_columns.min(axis=0).to_numpy()[None, :]
                    is_min.loc[level0] = min_mask.to_numpy()
        
        return pd.DataFrame(np.where(is_min, attr, ''), index=data.index, columns=data.columns)


    # Function to format mean ± std strings to 2 decimal places
    def format_mean_std_strings(data):
        def format_string(s, std_scheme="keep"):
            if s == "":
                return s
            mean, std = s.split(' ± ')

            mean = f"{float(mean):.2f}"
            std = f"{float(std):.2f}"
            if std_scheme == "keep":
                return f"{mean} ± {std}"
            elif std_scheme == "remove":
                assert std == "0.00"
                return mean

        # Create a copy to avoid modifying the original data
        result = data.copy()

        # Create a mask for rows whose level 0 index doesn't contain "calibrate"
        mask = ~result.index.get_level_values(0).astype(
            str).str.contains('calibrate', case=False)

        # Apply formatting only to those rows
        result.loc[mask] = result.loc[mask].applymap(
            format_string, std_scheme="keep")
        result.loc[~mask] = result.loc[~mask].applymap(
            format_string, std_scheme="remove")

        return result

    # Combine the highlighted and formatted data
    styled_df = data.style.apply(partial(highlight_min,axis=axis,attr=overall_best_format), axis=None)\
                                .apply(partial(highlight_min_level0,axis=axis,attr=method_best_format), axis=None)
    styled_df.data=format_mean_std_strings(data)
    # center align the text
    styled_df.set_properties(**{'text-align': 'center'})
    return styled_df


def to_latex(style, task_name):
    latex=style.to_latex(convert_css=True, hrules=True,
        clines="all;data",
        multirow_align="b",
        caption=f"Multicalibration error of rounded predictor for {task_name}. Results have been multiplied by 1000.",
        label=f"tab:{task_name.lower().replace(' ','_')}",
        column_format="rc" + "c"*6,
        position_float="centering",
        )
    def replace_patterns(input_string):
        # Regular expression to match the pattern 'number ± number'
        mean_std_pattern = r'(\d+\.\d+)\s±\s(\d+\.\d+)'
        
        # Replacement function for mean ± std
        def mean_std_replacement(match):
            mean = match.group(1)
            std = match.group(2)
            return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}} {mean} \\\\ \\small{{±{std}}} \\end{{tabular}}"
        
        # Regular expression to match the pattern 'xxxx Baseline'
        baseline_pattern = r'(\w+)\sBaseline'
        
        # Replacement function for xxxx Baseline
        def baseline_replacement(match):
            text = match.group(1)
            return f"\\begin{{tabular}}[c]{{@{{}}r@{{}}}} {text}\\\\ Baseline \\end{{tabular}}"
        
        # Replace all occurrences in the input string
        result = re.sub(mean_std_pattern, mean_std_replacement, input_string)
        result = re.sub(baseline_pattern, baseline_replacement, result)
        header_pattern=" & $m$ at Test & 10 & 20 & 30 & 50 & 75 & 100 \\\\\nMethod & $m$ at Cal &  &  &  &  &  &  "
        result = re.sub(re.escape(header_pattern),
            r"\\multirow{2}{*}{Method} & \\multirow{2}{*}{$m$ at Cal} & \\multicolumn{6}{c}{$m$ at Test} \\\\ \\cline{3-8}                &                             & 10  & 20  & 30  & 50 & 75 & 100",
                            result
        )
        return result
    return replace_patterns(latex)
