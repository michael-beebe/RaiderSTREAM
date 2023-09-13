#!/bin/bash

# Set C compiler
CC=gcc

$CC arraygen.c -o arraygen.exe
./arraygen.exe > IDX1.txt
./arraygen.exe > IDX2.txt
./arraygen.exe > IDX3.txt
rm arraygen.exe