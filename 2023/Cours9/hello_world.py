#!/usr/bin/env python3

from mpi4py import MPI
import sys

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

print(
    "Je suis le processus %d parmi %d sur %s.\n"
    % (rank, size, name))
