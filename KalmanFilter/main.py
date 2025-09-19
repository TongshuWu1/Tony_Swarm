# main.py

from simulation import Simulator
import matplotlib
matplotlib.use("TkAgg")


def main():
    sim = Simulator()
    sim.run()

if __name__ == "__main__":
    main()
