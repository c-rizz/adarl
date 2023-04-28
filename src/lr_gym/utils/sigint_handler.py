import signal
import traceback
import lr_gym.utils.dbg.ggLog as ggLog
from lr_gym.utils.utils import shutdown, is_shutting_down
import os
import time
from multiprocessing import shared_memory
import atexit
import threading

main_pid = None
did_initialize_sigint_handling = False
sigint_received = False
sigint_counter = 0
sigint_max = 10
original_sigint_handler = None


shared_memory_name = None
shared_memory_list = None


def destroy_shm():
    ggLog.info(f"destroying shm")
    shared_memory_list.shm.close()
    shared_memory_list.shm.unlink()

def close_shm():
    ggLog.info(f"closing shm")
    shared_memory_list.shm.close()

def sigint_handler(signal_num, stackframe):
    global sigint_received
    global sigint_counter
    global sigint_max
    global original_sigint_handler
    sigint_received = True
    sigint_counter += 1
    print(f"\n"+
            f"-----------------------------------------------------------------------------------------------------\n"+
            f"-----------------------------------------------------------------------------------------------------\n"+
            f"Received sigint, will halt at first opportunity. ({sigint_max-sigint_counter} presses to hard SIGINT, pid = {os.getpid()})\n"+
            f"-----------------------------------------------------------------------------------------------------\n"+
            f"-----------------------------------------------------------------------------------------------------\n\n")
    # print(f"current handler = {signal.getsignal(signal.SIGINT)}")
    # print(f"stackframe = {stackframe}")
    traceback.print_stack()
    if sigint_counter>sigint_max:
        shutdown()
        shared_memory_list[0] = "shutdown"
        try:
            original_sigint_handler(signal_num,stackframe)
        except KeyboardInterrupt:
            pass #If it was the original one, doesn't do anything, if it was something else it got executed
        raise KeyboardInterrupt
    if os.getpid() == main_pid:
        shared_memory_list[0] = "wait"
        
def setupSigintHandler():
    global original_sigint_handler
    global did_initialize_sigint_handling
    global main_pid
    global shared_memory_name
    global shared_memory_list
    if main_pid is None:
        main_pid = os.getpid()
        shared_memory_name = f"lr_gym_sigint_handler_{main_pid}_{time.time()}"
        shared_memory_list = shared_memory.ShareableList(sequence = ["run"], name=shared_memory_name)
        atexit.register(destroy_shm)
    else:
        shared_memory_list = shared_memory.ShareableList(name=shared_memory_name)
        atexit.register(close_shm)
    currenthandler = signal.getsignal(signal.SIGINT)
    if original_sigint_handler is None:
        original_sigint_handler = currenthandler

    if original_sigint_handler == sigint_handler:
        ggLog.warn(f"Sigint handler already set. Not setting again")
        return
    else:
        ggLog.info(f"Setting signal handler ")
    signal.signal(signal.SIGINT, sigint_handler)
    # ggLog.info(f"Sigint handler was {currenthandler}, set to {signal.getsignal(signal.SIGINT)}")
    did_initialize_sigint_handling = True

def haltOnSigintReceived():
    if not did_initialize_sigint_handling:
        return
    global sigint_received
    global sigint_counter
    if os.getpid() == main_pid:
        if sigint_received:
            answer = input(f"SIGINT received. Press Enter to resume or type 'exit' to terminate:\n> ")
            if answer == "exit":
                shutdown()
                shared_memory_list[0] = "shutdown"
                original_sigint_handler(signal.SIGINT, None)
                raise KeyboardInterrupt
            print("Resuming...")
            sigint_received = False
            sigint_counter = 0
            shared_memory_list[0] = "resume"
    else:
        status = shared_memory_list[0]
        if status == "wait":
            ggLog.info(f"SIGINT received, halting and waiting for main process...")
            while status == "wait":
                time.sleep(1)
                status = shared_memory_list[0]
            if status == "resume":
                sigint_counter = 0
        if status == "shutdown":
            shutdown()
            original_sigint_handler(signal.SIGINT, None)
            raise KeyboardInterrupt


def wait_halt_loop():
    while not is_shutting_down():
        # print(f"wait_halt_loop")
        haltOnSigintReceived()
        time.sleep(1)
    # print(f"ending wait_halt_loop {is_shutting_down()}")

def launch_halt_waiter():
    t = threading.Thread(target=wait_halt_loop)
    t.start()
    # print(f"starting wait_halt_loop")