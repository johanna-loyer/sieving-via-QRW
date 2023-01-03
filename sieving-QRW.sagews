# -*- coding: utf-8 -*-

#=========================================================================

# Authors : André Chailloux & Johanna Loyer
# Programming language : Sage

# This file implements the numerical optimisation of the paramaters in the paper 'Lattice sieving via Quantum random walks' ( https://eprint.iacr.org/2021/570.pdf )
# It also gives the trade-off curves and the synthesis table.

# To choose the wanted part of the code, modify the lines indicated by #<--- (Number of the code part)
#     (1) Numerical optimisation, prints the main results , (1bis) More data printed
#     (2) Computes points of the trade-off curve for fixed amonts of quantum memory and QRAQM
#     (3) Computes points of the trade-off curve for fixed amonts of QRACM (GAMMA)
#     (4) Display of the wanted trade-off curve + saves in PDF (requires to first run 2 and 3)
#     (5) Display of the two theorems statements on trade-offs (requires to first run 2 and 3)
#     (6) Display the synthesis table (requires to first run 2 and 3)
# By default, this file executes (1).
#=========================================================================


from math import sqrt, cos, sin, asin, acos, log, log2, log10, pi
import matplotlib.pyplot as plt
import numpy as np



val = log2(sqrt(4/3))
# N = sqrt(4/3)^d, with d the dimension of the lattice.
# N vectors are required to ensure N reducing pairs.
# All the following values, except angles, are in notation logN as in the paper.

# Auxiliary functions
def V(theta):
    return sin(theta)

def W(alpha,theta):
    return sqrt(1 - 2* cos(alpha)**2/ (1+cos(theta)))

def Exp(X):
    return log(X)/log(sqrt(4/3))

def Pr(alpha,theta):
    """ """
    return W(alpha,theta)/V(alpha)

def Alpha(c):
    return asin( sqrt(3/4)**c )

def Delta(c):
    a = Alpha(c)
    return 2 * asin( 1/(2*sin(a)) )

def Rho(c, delta):
    return Exp( Pr(Alpha(c), delta) )



#====================================#
#       Numerical optimisation       #
#====================================#

# Parameters of the algorithm :
#    N^c vectors per alpha-filter
#    N^c1 vectors per vertex in the quantum walk
#    N^c2 vectors per beta-filter, u:= c1-c2
#    N^rho : number of beta-filters each vector is inserted in

# Derivated values :
#    alpha : angle of the LSF alpha-filters
#    beta : angle of the LSF beta-filters
#    delta : maximal angle of the pairs of vectors we are searching during the quantum walk. (Corresponds to the notation theta^* in the paper)
#    N^rho0 : number of beta-filters each vector is inserted in if we want to not miss any collision through beta-filters.
#    N^NbSols : number of reducing pairs in one alpha-filter (Corresponds to the notation Zeta in the paper)
#    N^NbRepetitions : number of repetitions of the algorithm in order to reach N reduced vectors
#    N^SETUP, N^UPDATE, N^CHECK, N^EPSILON, N^DELTA : time complexity of the quantum walk steps
#    N^PREPROC : time complexity to insert vectors in their alpha-filter (includes the repetitions of the algorithm)
#    N^QUERIES : time complexity of the search of reducing pairs through quantum walk (includes the repetitions of the algorithm)


def time(c, c1, u): # time to find the solutions in one alpha-filter
    delta = Delta(1-c)
    rho0 = -Rho(u,delta)
    rho = max(rho0-2*c1 + Exp(V(delta)), 0) # Optimised choice for rho
    NbSols = 2*c + Exp(V(delta))
    NbRepetitions = max(0,NbSols)
    SETUP =  rho + c1
    UPDATE = max(rho, (rho+c1-u)/2, 0)
    CHECK = 0
    EPSILON = min(2*c1 + Exp(V(delta)) -rho0+rho, 0)
    DELTA = -c1

    QTIME = max(SETUP,-EPSILON/2 + max(UPDATE -DELTA/2, CHECK) ) + NbRepetitions
    return QTIME - min(NbSols/2,0)

def overall_time(c, c1, u): # time to find the solutions
    delta = Delta(1-c)
    NbSols = 2*c + Exp(V(delta)) #zeta
    QWalk = time(c, c1, u)
    PREPROC = 1 + c-NbSols
    QUERIES = QWalk + 1 -max(NbSols, 0)
    return max(PREPROC, QUERIES)

def optu(c,c1):
    def partial(u):
        return overall_time(c,c1,u)
    (t,u) = find_local_minimum(partial,0,c1)
    return t, u

def optc1(c):
    def partial(c1):
        return optu(c,c1)[0]
    (t,c1) = find_local_minimum(partial,0,c)
    return t, c1

def optc():
    def partial(c):
        #return overall_time(c,0,0) #c1=u=0 fixés
        return optc1(c)[0]
    (t, c) = find_local_minimum(partial,0,0.5,tol=0.001)
    return t, c


if 1: # <--- (1) Optimal time
    c = optc()[1]
    c1= optc1(c)[1]
    u = optu(c, c1)[1]
    t = overall_time(c, c1, u)

    c2 = c1-u
    delta = Delta(1-c)
    rho0 = -Rho(u,delta)
    rho = max(rho0-2*c1 + Exp(V(delta)), 0)

    print("TIME = N**(", t, ")") # Time complexity
    print("     = 2**(", t*val, "*d)")

    print("QRACM = 2**(", c*val, "*d)") # QRACM (Corresponds to the notation N^gamma in the paper)
    print("QRAQM = 2**(", (c1+rho)*val, "*d)") # qmem & QRAQM , (Corresponds to the notation N^mu in the paper)

    if 0: #<--- (1bis) Prints more data. Refers up to the meaning of these values.
        print("\nalpha  =", Alpha(1-c))
        print("theta^*=", delta)
        print("rho    =", rho)
        print("beta   =", Alpha(1-u))
        print("zeta   =", 2*copt + Exp(V(delta))) # = NbSols

        # Quantum walk steps
        S = c1 + rho
        U = rho
        C = 0
        e = min(2*c1 + Exp(V(delta)) -rho0+rho, 0)
        d = -c1
        print("\nS =", S)
        print("U =", U)
        print("C =", C)
        print("e =", e, "    -sqrt(e) =", -e/2)
        print("d =", d, "      -sqrt(d) =", -d/2)


copt = 0.3696256174206613
c1opt= 0.238418120630441
uopt = 0.23841811040362165



#====================================#
#          Trade-off curves          #
#====================================#

# It computes the points used to draw the trade-off curves.
# Scripts (2) and (3) are long to execute, but are required before to run the scripts (4), (5) and (6). 

if 0: #<--- (2) Constrainst over quantum memory and QRAQM (MU)
    time0 = 0.2653
    p = 1300  # precision = number of computed points
    MU = [] # qmem & QRAQM
    GAMMA = [] # QRACM
    T = [] # time
    RHO = []
    for ic in range(int(p/2)):
        c = ic/p
        delta = Delta(1-c)
        tmini = time0
        for ic1 in range(ic):
            c1 = ic1/p
            for iu in range(ic1):
                u = iu/p
                t = overall_time(c,c1,u)*val
                if t < tmini:
                    rho0 = -Rho(u,delta)
                    rho = max(rho0-2*c1 + Exp(V(delta)), 0)
                    if c1+rho <= c1opt+rhoopt:
                        mu = (c1+rho)*val
                        gamma = c*val
                        tmini = t
                        rho_ = rho
        if tmini < time0:
            MU.append(mu)
            GAMMA.append(gamma)
            T.append(tmini)
            RHO.append(rho_)

    # Deleting the redundant points to smooth the curve
    MUp, GAMMAp, Tp = [], [], []
    for i in range(len(T)):
        if T[i]<time0 :
            if not(MU[i] == MU[-1] and T[i] > min(T) ):
                  MUp.append(MU[i])
                  GAMMAp.append(GAMMA[i])
                  Tp.append(T[i])
    #print("MUp =", MUp) ; print("GAMMAp =", GAMMAp) ; print("Tp =", Tp)


if 0: #<--- (3) Constrainst over QRACM (GAMMA)
    time0 = 0.2926
    p = 1300  # precision  = number of computed points
    MU = [] # qmem & QRAQM
    GAMMA = [] # QRACM
    T = [] # time
    RHO = []
    for ic in range(int(copt*p)+1):
        c = ic/p
        delta = Delta(1-c)
        tmini = time0
        for ic1 in range(ic):
            c1 = ic1/p
            for iu in range(ic1):
                u = iu/p
                t = overall_time(c,c1,u)*val
                if t < tmini:
                    rho0 = -Rho(u,delta)
                    rho = max(rho0-2*c1 + Exp(V(delta)), 0)
                    mu = (c1+rho)*val
                    gamma = c*val
                    tmini = t
                    rho_ = rho
        if tmini < time0:
            MU.append(mu)
            GAMMA.append(gamma)
            T.append(tmini)
            RHO.append(rho_)

    # Deleting noise for MU ~= 0 (caused by Sage approximation errors)
    GAMMAs = [GAMMA[i] for i in range(len(GAMMA))]
    MUs = [MU[i] for i in range(len(MU))]
    Ts = [T[i] for i in range(len(T))]
    ib = 0
    while MUs[ib] < 0.0006:
        MUs[ib] = 0
        ib += 1
    gamma_break = GAMMAs[ib] #valeur de gamma au moment de la "cassure" de la courbe
    #print("MUs =", MUs) ; print("GAMMAs =", GAMMAs) ; print("Ts =", Ts)


# Display of the curve (+ saving in PDF)

# Replace 'CURVE = 0' by "CURVE = i" to display and save in PDF the curve i=1,2,3 or 4
# Curve 1 : time in function of quantum memory and QRAQM
# Curve 2 : QRACM in fonction of quantum memory and QRAQM
# Curve 3 : quantum memory and QRAQM in function of QRACM
# Curve 4 : time in function of QRACM


# /!\ Warning /!\ This code requires to first run the scripts (2) for curves 1 and 2, and (3) for curves 3 and 4.

import matplotlib.pyplot as plt
COURBE = 0 #<--- (4)

if COURBE != 0:
    fig = plt.figure()
    fig.subplots_adjust(top=2)
    ax = fig.add_subplot(211)

if COURBE == 1: # time for fixed qmem & QRAQM
    ax.set_xlabel("Quantum memory M")
    ax.set_ylabel("Time τ")
    ax.set_ylim([0.2565, 0.266])
    plt.plot(MUp,Tp, color="black")
    #plt.savefig("1time_qspace.pdf")
elif COURBE == 2: # QRACM for fixed qmem & QRAQM
    ax.set_xlabel("Quantum memory M")
    ax.set_ylabel("QRAM γ")      #ax.set_ylabel("QRACM γ")
    plt.plot(MUp, GAMMAp, color="black")
    #plt.savefig("2QRACM_qspace.pdf")
elif COURBE == 3: # time for fixed QRACM
    ax.set_xlabel("QRAM M'") #ax.set_xlabel("QRACM M'")
    ax.set_ylabel("Time τ")
    ax.set_ylim([0.255, 0.295])
    plt.plot(GAMMAs, Ts, color="black")
    #plt.savefig("3time_QRACM.pdf")
elif COURBE == 4: # qmem & QRAQM for fixed QRACM
    ax.set_xlabel("QRAM M'") #ax.set_xlabel("QRACM M'")
    ax.set_ylabel("Quantum memory µ")
    plt.plot(GAMMAs, MUs, color="black")
    #plt.savefig("4qspace_QRACM.pdf")





#====================================#
#  Affine approx. & max. deviations  #
#====================================#


def formule(X,Y, nomX="X", nomY="Y", pr=False):
    """Y = a*X + b"""
    x1, x2 = X[0], X[-1]
    y1, y2 = Y[0], Y[-1]
    a = (y1-y2)/(x1-x2)
    b = y1-a*x1
    return a,b

def ecart_max(Yexp, Ytheo):
    e1max = 0 # deviation pour Yexp > Ytheo
    e2max = 0 # deviation pour Yexp < Ytheo
    for i in range(len(Yexp)):
        e1 = Yexp[i]-Ytheo[i]
        e2 = Ytheo[i]-Yexp[i]
        if e1 > e1max:
            e1max = e1
        elif e2 > e2max:
            e2max = e2
    return e1max, e2max

def ap(x):
    """approximation of with 5 digits"""
    return int(x*10**5)/10**5


# /!\ Warning /!\ This code requires to first run the scripts (2) and (3).

if 0: #<--- (5) Display of the 2 theorems statements on trade-offs
    n = len(Tp)
    AFF_COURBE = True    # Superposition of curves + affine approximations

    #======= Fixed qmem & QRAQM (MU) =======#
    MUf = [0, (c1opt+rhoopt)*val]
    GAMMAf = [GAMMAp[0], copt*val] 
    Tmuf = [log2(sqrt(13/9)), topt] # time at 0 QRAQM: quantum [Laa16]

    # Time (T) in function of qmem & QRAQM (MU)
    if AFF_COURBE:
        plt.plot(MUp, Tp, color="black")  # time in funciton of qmem & QRAQM
        plt.plot(MUf, Tmuf)
        plt.show()
    a,b = formule(MUf, Tmuf)
    Ttheo = [a*MUp[i]+b for i in range(n)]
    e1, e2 = ecart_max(Tp, Ttheo)
    print("T =", ap(b), "+", ap(a), "* MU + [", -ap(e2), ",", ap(e1) , "]\n")

    # QRACM (GAMMA) in function of qmem & QRAQM (MU)
    if AFF_COURBE:
        plt.plot(MUp, GAMMAp, color="black") # QRACM in function of qmem & QRAQM 
        plt.plot(MUf, GAMMAf)
        plt.show()
    a,b = formule(MUf, GAMMAf)
    GAMMAtheo = [a*MUp[i]+b for i in range(n)]
    e1, e2 = ecart_max(GAMMAp, GAMMAtheo)
    print("GAMMA =", ap(b), "+", ap(a), "* MU + [", -ap(e2), ",", ap(e1) , "]\n")


    #======= Fixed QRACM (GAMMA) =======#


    # Time (T) in function of QRACM (GAMMA)
    GAMMAf = [GAMMAs[0], copt*val]
    Tgammaf = [log2(sqrt(3/2)), topt] # time at 0: classical [BDGL]

    if AFF_COURBE:
        plt.plot(GAMMAs, Ts, color="black") # time in function of QRACM
        plt.plot(GAMMAf, Tgammaf)
        plt.show()
    a,b = formule(GAMMAf, Tgammaf)
    Ttheo = [a*GAMMAp[i]+b for i in range(n)]
    e1, e2 = ecart_max(Tp, Ttheo)
    print("T =", ap(b), "+", ap(a), "* GAMMA + [", -ap(e2), ",", ap(e1) , "]\n")


    # qmem & QRAQM (MU) in function of QRACM (GAMMA)
    GAMMAf = [gamma_break, copt*val]
    MUf = [0, (c1opt+rhoopt)*val] 

    if AFF_COURBE:
        plt.plot(GAMMAs, MUs, color="black") # qmem & QRAQM in function of QRACM
        plt.plot(GAMMAf, MUf)
        plt.show()
    a,b = formule(GAMMAf, MUf)
    MUtheo = [a*GAMMAp[i]+b for i in range(n)]
    e1, e2 = ecart_max(MUp, MUtheo)
    #print("MU =", a, "* GAMMA +", b, "+ {", -e2, ",", e1 , "}\n") 
    print("MU =", ap(a), "* (GAMMA +", ap(b/a), ") + [", -ap(e2), ",", ap(e1) , "]\n")



#===================================#
#        Synthesis table            #
#===================================#

def qracm(c,c1,u): # gamma
    return c*val

def qraqm(c,c1,u): # mu
    delta = Delta(1-c)
    rho0 = -Rho(u,delta)
    rho = max(rho0-2*c1 + Exp(V(delta)), 0)
    return (c1+rho)*val

def iproche(L, v):
    """returns i st L[i] is the nearest value to v"""
    imini = 0
    diffmini = abs(L[0]-v)
    for i in range(1,len(L)):
        diff = abs(L[i]-v)
        if diff < diffmini:
            diffmini = diff
            imini = i
    return imini


# /!\ Warning /!\ This code requires to first run the scripts (2) and (3).

if 0: #<--- (6) Display the synthesis table
    BDGL = overall_time(0,0,0)*val # time complexity from article [BDGL] ( https://epubs.siam.org/doi/pdf/10.1137/1.9781611974331.ch2 )

    cLaa = 0.27830482810362156
    Laa  = overall_time(cLaa,0,0)*val # quantum time complexity from T. Laarhoven PhD thesis ( https://thijs.com/docs/phd-final.pdf )

    optparam = overall_time(copt,c1opt,uopt)*val # our time complexity for optimal parameters

    I = [iproche(Ts, BDGL), iproche(GAMMAs, 0.02), iproche(GAMMAs, 0.04), iproche(Ts, Laa)]
    I+= [iproche(GAMMAs, 0.065), iproche(GAMMAs, 0.07), iproche(Ts, overall_time(copt,c1opt,uopt)*val)]

    LT, LGAMMA, LMU = [], [], []
    for i in I:
        LT.append(ap(Ts[i]))
        LGAMMA.append( ap(GAMMAs[i]) )
        LMU.append( ap(MUs[i]) )

    print("Time :", LT)
    print("GAMMA:", LGAMMA)
    print("MU   :", LMU)
    
    
    




