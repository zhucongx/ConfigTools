from cfg.config import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # config = read_poscar("POSCAR")
    # write_poscar(config, "T")
    config = read_config("T")
    # write_config(config, "T2")
    write_poscar(config, "T2")
