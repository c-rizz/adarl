#!/usr/bin/env python3 
import h5py
import argparse
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import readline # enables better input() features (arrow keys, history)
# import seaborn as sns
import os

def recdict_access(rdict, keylist):
    if len(keylist)==0:
        return rdict
    return recdict_access(rdict[keylist[0]], keylist[1:])

def plot(data, filename, gui = True, labels = None, title : str = "HDF5Plot"):
    print(f"plotting data with shape {data.shape}")


    ax : matplotlib.axes.Axes
    fig, ax = plt.subplots()
    ax.grid(True, linestyle=":")
    ax.set_title(title)
    if len(data.shape)==1:
        data = np.expand_dims(data,1)
    series_num = data.shape[1]
    print(f"got labels {labels}")
    if labels is None:
        labels = [f"{i}" for i in range(series_num)]
        if len(labels)==1:
            labels = labels[0]
    print(f"using labels {labels}")
    ax.plot(data, label=labels)
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if gui:
        # matplotlib.use('TkAgg')
        fig.tight_layout()
        fig.show()
    else:
        fig.savefig(filename)

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
    k = recdict_access(f, current_path).keys()
    print(list(k))
    return current_path, True

def cmd_quit(file, current_path, *args, **kwargs):
    return current_path, False


def cmd_plot(file, current_path, *args, **kwargs):
    if len(args) < 1:
        print(f"Argument missing for plot.")
    print(f"cmd_plot({args})")
    data = np.array(recdict_access(f, current_path+[cmd[1]]))
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
    maybe_labels_name = cmd[1]+"_labels"
    if maybe_labels_name in recdict_access(f, current_path).keys():
        labels = np.array(recdict_access(f, current_path+[maybe_labels_name]))[0]
        labels = [a.tobytes().decode("utf-8").strip() for a in list(labels)]
        print(f"Found {len(labels)} labels {labels}")
        if columns is not None:
            labels = [labels[i] for i in columns]
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