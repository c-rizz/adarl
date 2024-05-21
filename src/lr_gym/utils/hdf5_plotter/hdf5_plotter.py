#!/usr/bin/env python3 
import h5py
import argparse
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

def recdict_access(rdict, keylist):
    if len(keylist)==0:
        return rdict
    return recdict_access(rdict[keylist[0]], keylist[1:])

def plot(data, filename, gui = True):
    print(f"plotting data with shape {data.shape}")
    ax : matplotlib.axes.Axes
    fig, ax = plt.subplots()
    if len(data.shape)==1:
        data = np.expand_dims(data,1)
    series_num = data.shape[1]
    labels = [f"{i}" for i in range(series_num)]
    if len(labels)==1:
        labels = labels[0]
    ax.plot(data, label=labels)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if gui:
        # matplotlib.use('TkAgg')
        fig.show()
    else:
        fig.savefig(filename)

def cmd_cd(file, current_path, *args, **kwargs):
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
    if len(args) != 1:
        print(f"Argument missing for plot.")
    print(f"Plotting...")
    data = recdict_access(f, current_path+[cmd[1]])
    plot(np.array(data), filename = "./plot.pdf")
    return current_path, True


def cmd_help(file, current_path, *args, **kwargs):
    cmds = kwargs["cmds"]
    print(f"Available commands:")
    for c in cmds.keys():
        print(f" - {c}")
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
                "plot" : cmd_plot,
                "help" : cmd_help}
        with h5py.File(fname, "r") as f:
            print(f"Opened file {fname}")
            print(f"Content:")
            print(list(recdict_access(f, current_path).keys()))
            cmd_help(f,current_path,cmds = cmds)
            while running:
                cmd = input("/"+"/".join(current_path)+"> ")
                cmd = cmd.split(" ")
                if len(cmd) == 0:
                    continue

                cmd_name = cmd[0]
                cmd_args = cmd[1:]
                cmd_func = cmds.get(cmd[0],None)
                if cmd_func != None:
                    kwargs = {}
                    kwargs["cmds"] = cmds
                    current_path, running = cmd_func(f,current_path, *args, **kwargs)
                else:
                    print(f"Command {cmd[0]} not found.")
    except Exception as e:
        print(f"Failed with exception: {e}")
        input("Press ENTER to close")