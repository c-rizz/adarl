#!/usr/bin/env python3 
import h5py
import argparse
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import readline # enables better input() features (arrow keys, history)
# import seaborn as sns
import os
from typing import TypeVar
import shutil
import math 

_K = TypeVar("_K")
_V = TypeVar("_V")


def recdict_access(rdict : dict[_K,_V], keylist : list[_K]) -> dict[_K,_V]:
    if len(keylist)==0:
        return rdict
    return recdict_access(rdict[keylist[0]], keylist[1:])

# def multiplot(n_cols_rows, plotnames, datas : dict, filename : dict, labels : dict, titles : dict):

plot_count = 0
def plot(data, filename, labels = None, title : str = "HDF5Plot"):
    print(f"plotting data with shape {data.shape}")

    global plot_count
    plot_count += 1
    ax : matplotlib.axes.Axes
    fig, ax = plt.subplots(num=title+str(plot_count))
    ax.grid(True, linestyle=":")
    ax.set_title(title)
    if len(data.shape)==1:
        data = np.expand_dims(data,1)
    series_num = data.shape[1]
    if labels is None:
        labels = [f"{i}" for i in range(series_num)]
        if len(labels)==1:
            labels = labels[0]
    lines = ax.plot(data, label=labels)
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    legend.set_draggable(True)

    map_legend_to_ax = {}  # Will map legend lines to original lines.
    for legend_line, ax_line in zip(legend.get_lines(), lines):
        legend_line.set_picker(5)  # Enable picking on the legend line. (radius at 5pt)
        map_legend_to_ax[legend_line] = ax_line
    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legend_line = event.artist
        if legend_line not in map_legend_to_ax:
            return
        ax_line = map_legend_to_ax[legend_line]
        visible = not ax_line.get_visible()
        ax_line.set_visible(visible)
        # Change the alpha on the line in the legend, so we can see what lines
        # have been toggled.
        legend_line.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()
    fig.canvas.mpl_connect('pick_event', on_pick)
    # matplotlib.use('TkAgg')
    fig.tight_layout()
    fig.show()

def cmd_cd(file, current_path, *args, **kwargs):
    """ Move into a the dataset structure as if it was a folder structure. \
        E.g. 'cd data' moves into the 'data' dict and 'cd ..' moves back \
        up the hierarchy."""
    k = recdict_access(f, current_path).keys()
    if len(cmd) == 1:
        current_path = []                
    elif cmd[1] == "..":
        current_path = current_path[:-1]
    else:
        if cmd[1] in k:
            new_path = current_path + [cmd[1]]
            if isinstance(recdict_access(f, current_path), dict):
                current_path = new_path
            else:
                print(f"{cmd[1]} is not dict-like")
        else:
            print(f"{cmd[1]} not found")
    return current_path, True

def cmd_ls(file, current_path, *args, **kwargs):
    ks = recdict_access(f, current_path).keys()
    max_k_len = max([len(k) for k in ks]) 
    ks = [(str(k)+" ").rjust(max_k_len) for k in ks]
    elements_per_row = int(shutil.get_terminal_size().columns/max_k_len)
    print('\n'.join([''.join(ks[p:p+elements_per_row]) for p in range(0,math.ceil(len(ks)/elements_per_row), elements_per_row)]))
    return current_path, True

def cmd_quit(file, current_path, *args, **kwargs):
    return current_path, False


def cmd_plot(file, current_path, *args, **kwargs):
    if len(args) < 1:
        print(f"Argument missing for plot.")
    print(f"cmd_plot({args})")
    available_fields = recdict_access(f, current_path).keys()
    field : str = ""
    if cmd[1] in available_fields:
        field = cmd[1]
    else:
        matches = []
        for af in recdict_access(f, current_path).keys():
            if  af.startswith(cmd[1]):
                matches.append(af)
        if len(matches)==1:
            field = matches[0]
        else:
            print(f"Possible fields = "+(",".join(matches)))
    data = np.array(recdict_access(f, current_path+[field]))
    if len(data.shape) == 1:
        data = np.expand_dims(data,1)
    col_num = data.shape[1]
    columns = None
    if len(args)==2:
        columns = []
        groups = args[1].split(",") # e.g. "1:4,7:9,11,12" gets split in ["1:4","7:9","11","12"]
        for g in groups:
            if ":" in g:
                e = g.split(":")
                if len(e)>3:
                    raise RuntimeError(f"Invalid slice '{g}'")
                if len(e)==2:
                    e.append("")
                if e[0] == "": e[0] = 0
                if e[1] == "": e[1] = col_num
                if e[2] == "": e[2] = 1
                e = [int(es) for es in e]
                columns += list(range(col_num))[e[0]:e[1]:e[2]]
            else:
                columns.append(int(g))
    if columns is not None:
        data = data[:,columns]
    maybe_labels_name = field+"_labels"
    if maybe_labels_name in recdict_access(f, current_path).keys():
        labels = np.array(recdict_access(f, current_path+[maybe_labels_name]))[0]
        labels = [a.tobytes().decode("utf-8").strip() for a in list(labels)]
        print(f"Found {len(labels)} labels {labels}")
        if columns is not None:
            labels = [labels[i] for i in columns]
        else:
            columns = list(range(col_num))
        n = "\n"
        print(f"using labels {n.join([f'{i} : {l}' for i,l in zip(columns,labels)])}")
    else:
        labels = columns
    plot(data, labels=labels, filename = "./plot.pdf", title = os.path.basename(kwargs["filename"])+"/"+"/".join(current_path))
    return current_path, True

from collections import defaultdict
def cmd_help(file, current_path, *args, **kwargs):
    """ This help command. """
    cmds = kwargs["cmds"]
    cmds_by_func = defaultdict(list)
    for key, value in sorted(cmds.items()):
        cmds_by_func[value].append(key)
    print(f"Available commands:")
    n = "\n"
    for func,cmd_names in cmds_by_func.items():
        doc = func.__doc__
        if doc is None:
            doc = "No documentation."
        doc = doc.replace(n,' ')
        doc = ' '.join([k for k in doc.split(" ") if k])
        print(f" - {', '.join(cmd_names)} :\n"
              f"    {doc}")
    return current_path, True

if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--file", default = None, type=str, help="File to open")
        ap.add_argument("file", nargs='?', default = None, type=str, help="File to open")

        ap.set_defaults(feature=True)
        args = vars(ap.parse_args())

        fname = args["file"]
        if fname is None:
            print(f"Not input file provided.")
            input("Press ENTER to exit.")
            exit(0)
        current_path = []
        running = True
        cmds = {"cd" : cmd_cd,
                "ls" : cmd_ls,
                "quit" : cmd_quit,
                "exit" : cmd_quit,
                "q" : cmd_quit,
                "plot" : cmd_plot,
                "p" : cmd_plot,
                "help" : cmd_help}
        with h5py.File(fname, "r") as f:
            print(f"Opened file {fname}")
            print(f"Content:")
            print(list(recdict_access(f, current_path).keys()))
            cmd_help(f,current_path,cmds = cmds)
            while running:
                cmd = input("/"+"/".join(current_path)+"> ")
                cmd = " ".join(cmd.split()) # remove repeated spaces
                cmd = cmd.split(" ")
                if len(cmd) == 0:
                    continue

                cmd_name = cmd[0]
                cmd_args = cmd[1:]
                cmd_func = cmds.get(cmd[0],None)
                if cmd_func != None:
                    kwargs = {}
                    kwargs["cmds"] = cmds
                    kwargs["filename"] = fname
                    try:
                        current_path, running = cmd_func(f,current_path, *cmd_args, **kwargs)
                    except Exception as e:
                        print(f"Command failed with exception {e.__class__.__name__}: {e}")
                else:
                    print(f"Command {cmd[0]} not found.")
    except Exception as e:
        print(f"Failed with exception: {e}")
        input("Press ENTER to close")