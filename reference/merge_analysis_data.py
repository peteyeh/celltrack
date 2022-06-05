import os
import pandas as pd
import re

data_dirs = [
    "saved_output/analysis_data/plate1",
    "saved_output/analysis_data/plate2",
    "saved_output/analysis_data/plate3",
    "saved_output/analysis_data/plate4",
    "saved_output/analysis_data/plate5",
    "saved_output/analysis_data/plate6",
    "saved_output/analysis_data/plate7",
    "saved_output/analysis_data/plate8",
    "saved_output/analysis_data/plate9",
    "saved_output/analysis_data/plate10",
    "saved_output/analysis_data/plate11",
    "saved_output/analysis_data/plate12"
]

def parse_stack_name(name):
    r = re.search("(1?[0-9])([A-H])(1?[0-9])", name)
    return r.group(1), r.group(2), r.group(3)

merged_df = None
for data_dir in data_dirs:
    for image_stack in os.listdir(data_dir):
        # hacky check to ignore Mac temp files
        if image_stack[0] == ".":
            continue
        df = pd.read_csv(os.path.join(data_dir, image_stack, "analysis.csv"))
        plate, row, col = parse_stack_name(image_stack)
        df['plate'] = plate
        df['row'] = row
        df['col'] = col
        df = df.reset_index().rename(columns={"index": "frame"}).set_index(["plate", "row", "col", "frame"])
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.append(df)

merged_df.sort_index().to_csv("saved_output/merged_data.csv")
