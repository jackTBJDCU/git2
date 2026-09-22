# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:34:24 2026

@author: boris
"""
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

def dx(v):
    return v

def dv(x):
    return -x
def x(t):
    return sin(t)


x0 = 0
v0 = 1
