#!/bin/bash

qsub -l nodes=30:ppn=2 -N hpc0855844-test test.pbs
