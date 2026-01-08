import glob
import json
import os
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm

DATA_DIR="../data"
OUTPUT_DIR="../experiments/figures"

linestyles = ["-", "--", "dashdot"]
marker_list = ["o", "s", "^", "v"]

font_size = 9
label_size = 9
inches_per_pt = 1 / 72.27
columnwidth = 240.94499 * inches_per_pt
textwidth = 505.89 * inches_per_pt

golden_ratio = (5**0.5 - 1) / 2
default_width = columnwidth
full_width = textwidth
default_height = default_width * golden_ratio
full_height = textwidth * golden_ratio
figsize = (default_width, default_height)
print(f"default figsize: {figsize}")


plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['figure.titlesize'] = font_size
plt.rcParams['figure.labelsize'] = font_size
plt.rcParams["figure.dpi"] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['markers.fillstyle'] = 'none'
plt.rcParams['axes.autolimit_mode'] = 'data'


linestyles = ["-", "--", "dashdot"]
marker_list = ["o", "s", "^", "v"]
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

def set_xlim_pad(ax, lo, hi):
    pad_frac = 0.05
    ax.set_xmargin(0)
    rng = hi - lo
    pad = rng * pad_frac
    ax.set_xlim(lo - pad, hi + pad)

def set_ylim_pad(ax, lo, hi):
    pad_frac = 0.05
    rng = hi - lo
    pad = rng * pad_frac
    ax.set_ylim(lo - pad, hi + pad)

def set_xlim_pad_auto(ax, xs):
    pad_frac = 0.05
    lo = min(xs)
    hi = max(xs)
    rng = hi - lo
    pad = rng * pad_frac
    ax.set_xlim(lo - pad, hi + pad)
 
def set_ylim_pad_auto(ax, ys):
    pad_frac = 0.05
    lo = 0
    hi = max(ys)
    rng = hi - lo
    pad = rng * pad_frac
    ax.set_ylim(lo - pad, hi + pad)

def read_trace(path):
    traces = pd.read_csv(os.path.join(DATA_DIR, path))
    # Convert all numeric columns to int
    numeric_cols = traces.select_dtypes(include=['number']).columns
    traces[numeric_cols] = traces[numeric_cols].astype('Int64')
    return traces
