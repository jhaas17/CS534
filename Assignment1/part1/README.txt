The program can be run in the terminal as such:

python queens.py [file] [alg] [heu]

Where [file] is replaced with the path to the input file, the file
should be a csv file and its name cannot contain spaces.
[alg] is to replaced with 1 if you would like to run the A* search
[alg] is to be replaced with 2 if you would like to run the
Greedy search
An example of the command with values filled in:

python queens.py heavyqueensboard.csv 1 H2

The following libraries will need to be installed:
argparse
queue
numpy
random
time
csv
