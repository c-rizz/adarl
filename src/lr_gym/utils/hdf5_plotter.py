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
        fig.show()
    else:
        fig.savefig(filename)


if __name__ == "__main__":


    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required = True, type=str, help="File to open")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    fname = args["file"]
    current_path = []
    running = True
    with h5py.File(fname, "r") as f:
        while running:
            cmd = input("/"+"/".join(current_path)+"> ")
            cmd = cmd.split(" ")
            if len(cmd) == 0:
                continue
            elif cmd[0] == "cd":
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
            elif cmd[0] == "ls":
                k = recdict_access(f, current_path).keys()
                print(list(k))
            elif cmd[0] in ["quit", "exit"]:
                running = False
            elif cmd[0] == "plot":
                if len(cmd) != 2:
                    print(f"Argument missing for plot.")
                data = recdict_access(f, current_path+[cmd[1]])
                plot(np.array(data), filename = "./plot.pdf")
            else:
                print(f"unknown command {cmd[0]}")