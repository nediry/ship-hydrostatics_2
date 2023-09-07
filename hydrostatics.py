"""
Created on Wed Jan 26 18:04:16 2022
@author: nedir ymamov
"""

# import using all packages in project
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


def draw_wl_sl(offset, length, breadth, draft, pn, wl):
    # drawing water lines
    plt.figure(figsize = (10, 3.5))
    pn_new = np.linspace(0, length, 51)
    offset_new = np.zeros((51, offset.shape[1]))
    for i in range(offset.shape[1]):
        f = interp1d(pn, offset[:, i], kind = "cubic")
        offset_new[:, i] = np.round(f(pn_new), 3)

    for i in range(offset_new.shape[1]):
        plt.plot(pn_new, offset_new[:, i])
        plt.plot(pn_new, -offset_new[:, i])
    
    plt.legend(["wl0", "wl0.3", "wl1", "wl2", "wl3", "wl4", "wl5", "wl6"], loc = "lower center")
    plt.title("Waterlines Curves")
    plt.xlabel("Length of ship [m]")
    plt.ylabel("Breadth of ship [m]")
    plt.savefig("waterline_curves.png")
    plt.show()

    # drawing section lines
    wl_new = np.linspace(0, 1.5 * draft, 21)
    offset_new = np.zeros((13, 21))
    for i in range(13):
        f = interp1d(wl, offset[i, :], kind = "cubic")
        offset_new[i, :] = np.round(f(wl_new), 3)

    for i in range(13):
        if i <= 6:
            plt.plot(-offset_new[i], wl_new)
        else:
            plt.plot(offset_new[i], wl_new)
    
    plt.legend(["pn0", "pn0.5", "pn1", "pn2", "pn3", "pn4", "pn5", "pn6", "pn7", "pn8", "pn9", "pn9.5", "pn10"], loc = "upper right")
    
    plt.plot([-breadth / 2 - 0.1, breadth / 2 + 3.5], [draft, draft], [0, 0], [0, 1.5 * draft])
    plt.title("Section Lines Curves")
    plt.xlabel("Breadth of ship [m]")
    plt.ylabel("Depth of ship [m]")
    plt.savefig("sectionlines_curves.png")
    plt.show()


def offset_expand(offset, length, breadth, draft, pn, wl):
    row, col = 50, 15  # new dimensions (rows, columns)
    
    wl_new = np.linspace(0, 1.5 * draft, col + 1) # new array of water lines (planes)
    pn_new = np.linspace(0, length, row + 1) # new array of section lines (planes)
    offset_pre = np.zeros((13, col + 1))
    for i in range(13):
        f = interp1d(wl, offset[i, :], kind = "cubic")
        offset_pre[i, :] = np.round(f(wl_new), 3)
    
    offset_new = np.zeros((row + 1, col + 1)) # new array of expanded offset tables
    for i in range(col + 1):
        f1 = interp1d(pn, offset_pre[:, i], kind = "cubic")
        offset_new[:, i] = np.round(f1(pn_new), 3)
    return pn_new, wl_new, offset_new


def area_moment(offset, wl):
    row, col = offset.shape # number of section and water lines
    area = np.zeros((row, col))   # new array of bon-jean areas
    for i in range(row):
        area[i, 1:] = 2 * cumtrapz(offset[i, :], wl)
    area = np.round(area, 3)
    
    moment = np.zeros((row, col))  # new array of bon-jean moments
    for i in range(col):
        moment[:, i] = offset[:, i] * wl[i]
    for i in range(row):
        moment[i, 1:] = 2 * cumtrapz(moment[i, :], wl)
    return area, moment


def vol_dep(area, pn):
    # calculation volume and volume displacement of ship
    
    col = offset.shape[1] # number of water lines (planes)
    volume = np.zeros(col) # new array of volumes
    for i in range(1, col):
        volume[i] = np.trapz(area[:, i], pn) # volume calculation
    volume = np.round(volume, 3)

    rho = 1.025 # density of sea water
    displacement = rho * volume # volume displacement calculation
    displacement = np.round(displacement, 3)
    return volume, displacement


def area_wp(offset, pn):
    col = offset.shape[1] # number of water lines
    Awp = np.zeros(col)  # new array of waterline (waterplane) areas
    for i in range(col):
        Awp[i] = np.trapz(offset[:, i], pn)
    Awp = np.round(Awp, 3)
    return Awp


def location(offset, pn, Awp, area, volume):
    col = offset.shape[1] # number of water lines
    
    # longitudinal center of flotation location (starts from the stern (aft) of the ship)
    MxAwp = np.zeros(col)
    for i in range(col):
        MxAwp[i] = 2 * np.trapz(offset[:, i] * pn, pn)
    LCF = MxAwp / Awp - length / 2

    # longitudinal center of buoyancy location (starts from the stern (aft) of the ship)
    LCB = np.zeros(col) # new array of LCB
    for i in range(1, col):
        LCB[i] = np.trapz(area[:, i] * pn, pn) / volume[i]

    # vertical location of center of volume, KB = GM + KG - BM
    KB = np.zeros(col)
    for i in range(1, col):
        KB[i] = np.trapz(moment[:, i], pn) / volume[i]
    return LCF, LCB, KB


def coefficients(length, offset, area, volume, wl):
    # block coefficients
    CB = np.array([0, *volume[1:] / (length * 2 * offset[6, 1:] * wl[1:])])

    # midship section area coefficients
    # Cm = Immersed area of midship section / (B * T)
    # B is the overall breadth or beam. T is the full load draft
    CM = np.array([0, *area[6, 1:] / (2 * offset[6, 1:] * wl[1:])])

    # prismatic coefficients
    # Cp = Immersed volume V / (L * immersed midship area)
    CP = np.array([0, *CB[1:] / CM[1:]])

    # waterplane coefficients
    # Cwp = Awp / (L * B)
    CWP = Awp / (length * 2 * offset[6, :])
    return CB, CM, CP, CWP


def metacentr(offset, length, Awp, LCF, volume):
    col = offset.shape[1] # number of waterlines (waterplanes)
    # ICL is the second moment of the half waterplane about the centreline
    ICL = np.zeros(col)
    for i in range(col):
        ICL[i] = (2 / 3) * np.trapz(offset[:, i]**3, pn)
    # transverse BM is the height of the transverse metacentre above the centre of bouyancy
    BM = np.array([0, *ICL[1:] / volume[1:]])

    # IM is the moment of inertia about amidships
    IM = np.zeros(col)
    for i in range(col):
        IM[i] = np.trapz(offset[:, i] * pn**2, pn)
    
    # IF is the moment of inertia about the LCF
    IF = IM - Awp * (length / 2 - LCF)**2
    
    # BML is the height of the longitudinal metacentre above the centre of bouyancy
    BML = np.array([0, *IF[1:] / volume[1:]])
    return BM, BML


def wet_area(offset, pn, wl):
    row, col = offset.shape # number of section and water lines (planes)
    
    def arc_length(x, y):
        # calculating the length of the curves
        npts = len(x)
        arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
        for k in range(1, npts):
            arc = arc + np.sqrt((x[k] - x[k - 1])**2 + (y[k] - y[k - 1])**2)
        return arc / 2
    
    l = np.zeros((row, col))
    for i in range(row):
        for j in range(1, col):
            l[i, j] = round(arc_length(offset[i, j - 1 : j + 1], wl[j - 1 : j + 1]), 3)
            
    S = np.zeros(col) # new array of wet surface area curves
    for i in range(1, col):
        S[i] = S[i - 1] + 2 * np.trapz(l[:, i], pn)
    return S


def sink_one_cm(Awp, displacement, bml, length):
    rho = 1.025 # density of sea water
    # Tonnes Per Centimeter, loaded or discharged to change the ship's mean draft by one centimetre
    TPC = Awp * rho / 100  # one centimeter sinking tonnage
    
    # Moment to Change Trim one centimeter, for metric units of metric tonsmeters/cm
    MCT = np.array([0, *displacement[1:] * bml[1:] / (100 * length)]) # one centimeter sinking moment
    return TPC, MCT


def draw_curves(length, breadth, draft, area, moment, hacim, displacement, Awp, lcf, lcb, kb, cb, cm, cp, cwp, TPC, MCT, S):
    # drawing all hydrostatics (bon-jean) curves
    row, col = area.shape # number of section and water lines
    
    plt.figure(figsize = (10, 5))
    # drawing networks
    for i in range(row):
        plt.plot([pn[i], pn[i]], [0, 1.5 * draft], "k")
    for i in range(col):
        plt.plot([0, length], [wl[i], wl[i]], "k")

    # curves of bon-jean areas
    s = length / 10 # distance between frames of the ship
    
    # Why is the ratio necessary? All curves must somehow fit into the same
    # drawing with the scaled ratio in order for it to be scaled later on.
    ratio = np.max(area) / s
    for i in range(row - 1):
        plt.plot(area[i] / ratio + pn[i], wl, "g")

    # curves of bon-jean moments
    for i in range(row - 1):
        plt.plot(area[i] / (ratio + 1.5) + pn[i], wl, "r--")

    # curves of volumes and displacements
    ratio = displacement[-1] / length
    plt.plot(displacement / ratio, wl)
    
    plt.plot(hacim / (ratio + 1.5), wl, "r--")

    # curves of waterline (water plane) areas
    Awp[2] -= 10
    Awp[4] += 6
    ratio = Awp[-1] / (length - 15)
    wl_new = np.linspace(0, 1.5 * draft, 25)
    spline = interp1d(wl, Awp, kind = "quadratic")
    plt.plot(spline(wl_new) / ratio, wl_new, "c--")

    # curves of LCF, LCB and KB
    spline = interp1d(wl, lcf, kind = "cubic")
    plt.plot(spline(wl_new), wl_new, "r")
    plt.plot(lcb[1:], wl[1:], "b")
    ratio = kb[-1] / (2 * s)
    plt.plot(kb / ratio, wl, "b--")

    # curves of coefficients
    ratio = cb[-1] / (.5 * s)
    plt.plot(cb[1:] / ratio, wl[1:], "b")
    plt.plot(cm[1:] / ratio, wl[1:], "r")
    plt.plot(cp[1:] / ratio + s, wl[1:], "b")
    plt.plot(cwp[1:] / ratio + s, wl[1:], "r")
    
    # curves of TPC and MCT
    wl_new = np.linspace(wl[1], 1.5 * draft, 50)
    spline = interp1d(wl, TPC, kind = "cubic")
    plt.plot(spline(wl_new) * 3, wl_new, "r")
    
    ratio = MCT[-1] / (length / 3)
    spline = interp1d(wl, MCT, kind = "cubic")
    plt.plot(spline(wl_new) / ratio, wl_new)
    
    # curves of wet surface areas
    ratio = S[-1] / (length / 2)
    plt.plot(length / 2 + S / ratio, wl)
    
    plt.title("Bon-Jean Curves")
    plt.xlabel("Length of ship [m]")
    plt.ylabel("Depth of ship [m]")
    plt.savefig("hydrostatics_curves.png")
    plt.show()


np.set_printoptions(precision = 3)
length = 100 # length of ship (LBP)
breadth = 15 # breadth of ship
draft = 6 # draft of ship (draft = 1.5 * depth), depth of ship
wl = np.array([0, .5, 1, 2, 3, 4, 5, 6]) * draft / 4 # array of water lines
pn = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) * length / 10 # array of section lines
offset = np.loadtxt("s60_cb70.txt", dtype=float) # table of dimensionless half-breadth offset
offset *= breadth / 2


draw_wl_sl(offset, length, breadth, draft, pn, wl)
pn_new, wl_new, offset_new = offset_expand(offset, length, breadth, draft, pn, wl)

area, moment = area_moment(offset, wl)
volume, displacement = vol_dep(area, pn)
Awp = area_wp(offset, pn)
LCF, LCB, KB = location(offset, pn, Awp, area, volume)
CB, CM, CP, CWP = coefficients(length, offset, area, volume, wl)
BM, BML = metacentr(offset, length, Awp, LCF, volume)
TCP, MCT = sink_one_cm(Awp, displacement, BML, length)
S = wet_area(offset, pn, wl)

draw_curves(length, breadth, draft, area, moment, volume, displacement, Awp, LCF, LCB, KB, CB, CM, CP, CWP, TCP, MCT, S)