# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:54:23 2026

@author: boris
"""
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def f(t,I ,V,R,L):
    didt = (V - (R * I)) / L
    return didt
  
L_f = [50.0,100.0,1000.0]
R_f = [5.0,50.0,100.0]
V_f = [1.0,10.0,100.0]
I0 = np.array([0.0])
t0 = 0.0
tf = 50.0
n_f = [100,500,1000]
for L in L_f:
    fig, (ax1,ax2) = plt.subplots(2, 1,)
    fig.set_size_inches(10, 11) 
    V = 10.0
    R = 50.0
    for n in n_f:
        diff = []
        t_e = np.linspace(t0, tf, n)
        I_exact = (V / R) * (1 - np.exp(-(R * t_e / L)))
        
        ax1.plot(t_e,I_exact,"--g",label = f"exact I at n = {n}" ) 
        results = solve_ivp(fun=f,t_span=(t0,tf),y0=I0,t_eval=t_e,method="RK45",args=(V,R,L))
        I = results.y[0]
        t = results.t
        ax1.plot(t, I,alpha=0.7, label = f"Aproximation of I at n = {n}")
        for i in range(len(I)):
            diff.append(I_exact[i] - I[i])
        ax2.plot(t_e,diff,label=f"diffrence at n = {n}",alpha = 0.5)
    ax1.set_xlabel("t")
    ax1.set_ylabel("I")
    ax2.set_xlabel("t")
    ax2.set_ylabel(f" {u"\u0394"} I")
    ax1.set_title(f"Runge-Kutta RK45 solution of I when the circuit is closd and its diffrence at L = {L}")
    ax1.legend()
    ax2.legend()
    plt.show()
         
for V in V_f:
    fig, (ax1,ax2) = plt.subplots(2, 1,)
    fig.set_size_inches(10, 11) 
    L = 100.0
    R = 50.0
    for n in n_f:
        diff = []
        t_e = np.linspace(t0, tf, n)
        I_exact = (V / R) * (1 - np.exp(-(R * t_e / L)))
        
        ax1.plot(t_e,I_exact,"--g",label = f"exact I at n = {n}" ) 
        results = solve_ivp(fun=f,t_span=(t0,tf),y0=I0,t_eval=t_e,method="RK45",args=(V,R,L))
        I = results.y[0]
        t = results.t
        ax1.plot(t, I,alpha=0.7, label = f"Aproximation of I at n = {n}")
        for i in range(len(I)):
            diff.append(I_exact[i] - I[i])
        ax2.plot(t_e,diff,label=f"diffrence at n = {n}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("I")
    ax2.set_xlabel("t")
    ax2.set_ylabel(f" {u"\u0394"} I")
    ax1.set_title(f"Runge-Kutta RK45 solution of I when the circuit is closd and its diffrence at V = {V}")
    ax1.legend()
    ax2.legend()
    plt.show()
         
for R in R_f:
    fig, (ax1,ax2) = plt.subplots(2, 1,)
    fig.set_size_inches(10, 11) 
    L = 100.0
    V = 10.0
    for n in n_f:
        diff = []
        t_e = np.linspace(t0, tf, n)
        I_exact = (V / R) * (1 - np.exp(-(R * t_e / L)))
        
        ax1.plot(t_e,I_exact,"--g",label = f"exact I at n = {n}" ) 
        results = solve_ivp(fun=f,t_span=(t0,tf),y0=I0,t_eval=t_e,method="RK45",args=(V,R,L))
        I = results.y[0]
        t = results.t
        ax1.plot(t, I,alpha=0.7, label = f"Aproximation of I at n = {n}")
        for i in range(len(I)):
            diff.append(I_exact[i] - I[i])
        ax2.plot(t_e,diff,label=f"diffrence at n = {n}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("I")
    ax2.set_xlabel("t")
    ax2.set_ylabel(f" {u"\u0394"} I")
    ax1.set_title(f"Runge-Kutta RK45 solution of I when the circuit is closd and its diffrence at R = {R}")
    ax1.legend()
    ax2.legend()
    plt.show()
         
