#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import os.path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as mpl
mpl.rcParams.update({'font.size': 14})
import numpy as np
from scipy import signal
from scipy.io import FortranFile
from scipy.signal import hilbert, chirp, find_peaks

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--case", type=str, default=False,
    help='case [-c] [--case] file name containing single case')
parser.add_argument("-l", "--list", type=str, default=False,
    help='case [-l] [--list] file name containing list of cases')

args = parser.parse_args()

if type(args.case) is str:
    input_type = 1
    input_file = args.case
elif type(args.list) is str:
    input_type = 2
    input_file = args.list
else:
    print("need to give case [-c] or database list [-l]")

def read_xyz(dname, fname, N):
    """
    Read data file

    Args:
        :dname: directory name
        :fname: file name
        :N: number of steps
    """

    # find number of timesteps
    if N == -1:
        fid = FortranFile(dname + fname, "r")
        for n in range(10000000):
            try:
                data = fid.read_reals(float)
            except:
                break
        N = n
        fid.close()

    # initialise arrays
    time = np.zeros((N,))
    Fx = np.zeros((N,))
    Fy = np.zeros((N,))
    Fz = np.zeros((N,))
    x = np.zeros((N,))
    y = np.zeros((N,))
    z = np.zeros((N,))

    # read Fortran unformatted data
    fid = FortranFile(dname + fname, "r")
    for n in range(N):
        try:
            data = fid.read_reals(float)
        except:
            break
        time[n] = data[0]
        Fx[n] = data[1]
        Fy[n] = data[2]
        Fz[n] = data[3]
        x[n] = data[4]
        y[n] = data[5]
        z[n] = data[6]

    fid.close()

    return time, Fx, Fy, Fz, x, y, z, N

def spectrum(t_in, s_in, t_start=None):
    """
    Calculate power spectral density

    Args:
        :t_in: time
        :s_in: signal
        :t_start: start time
    """

    Nt = len(t_in)
    if not t_start==None:
        n_start = np.argmax(t_in > t_start)
    else:
        n_start = 0
    t = t_in[n_start:]
    s = s_in[n_start:]
    T = t[1] - t[0]
    n = len(t)
    N = n // 2
    f = np.fft.fftfreq(n, d=T)
    f = f[:N]
    y = np.abs(np.fft.fft(s) / n)[:N]
    return f, y

####

# import case with all details on run to be analysed
if input_type == 1:
    exec("from %s import case, save_plot, show_plot, write_data, incl_title, Ndtime" % input_file)
    datalist = [case]
elif input_type == 2:
    exec("from %s import datalist, save_plot, show_plot, write_data, incl_title, Ndtime" % input_file)

def run_datalist(nca, case):

    Npeaks_min = 1000000000
    Npeaks_max = 0
    
    ftmp = case['rootdir'] + case['cname']
    if not os.path.exists(ftmp):
        print('missing: %s' % ftmp)
        return

    if save_plot:
        # folder to store plots
        folder_plots = ftmp + '/plots'
        print("Create plot output folder : %s" % folder_plots)
        if not os.path.exists(folder_plots):
            os.makedirs(folder_plots)
    
    yDmax = np.zeros((len(case['nc'])))
    fqy = np.zeros((len(case['nc'])))
    fqCl = np.zeros((len(case['nc'])))
    fqCd = np.zeros((len(case['nc'])))
    Cxm = np.zeros((len(case['nc'])))
    Cym = np.zeros((len(case['nc'])))
    Cymax = np.zeros((len(case['nc'])))
    
    t_start = case['t_start']
    
    for nci, nc in enumerate(case['nc']):
    
        # reading the data from the file
        pfile = case['rootdir'] + case['cname'] + "/parameters.dat"
        if os.path.isfile(pfile):
            with open(pfile) as f:
                data = f.read()
            js = json.loads(data)
            diameter = js['D']
            radius = diameter * 0.5
            xyc = [js['xcyl'] * diameter, js['ycyl'] * diameter]
            try:
                Uref = js['speeds'][nc-1]
            except:
                Uref = js['speeds'][0]
            mu = js['mu']  # Pa.s
            rho = js['rho']  # kg/m3
            Lxyz = [l * diameter for l in js['Lxyz']]  # m
            Nz = js['ncouches']
            Ns = js['Ns']
            Re = rho * Uref * diameter / mu
            strip_theory = js['strip_theory']
            if strip_theory:
                L = js['L']
                smodel = js['model']
                if smodel == 0:  # MSD
                    try:
                        k = js['k'][nc-1]
                    except:
                        k = js['k'][0]
                    fc = np.sqrt(k / js['ml']) / (2.0 * np.pi)
                elif smodel >= 1:  # string
                    try:
                        H = js['H'][nc-1]
                    except:
                        H = js['H'][0]
                    fc = 0.5 / js['L'] * np.sqrt(H / js['ml'])
                
                fn = diameter / Uref * fc
                Ustar = 1.0 / fn
                mstar = js['ml'] / (js['rho'] * js['D']**2)
                zeta = js['zeta']
    
            """print('diameter = %gm, xyc = (%g,%g)m, Uref = %gm/s' % (diameter, xyc[0], xyc[1], Uref))
            print('mu = %ePa.s, rho = %gkg/m3' % (mu, rho))
            print('Lxyz = (%g,%g,%g) m' % (Lxyz[0], Lxyz[1], Lxyz[2]))
            print('Nz = %d, Ns = %d' % (Nz, Ns))
            print('strip_theory = %r, Ns = %d' % (strip_theory, Ns))"""
            
            if strip_theory:
                if smodel == 0:  # MSD
                    str_tmp = 'MSD'
                elif smodel >= 1:  # String
                    str_tmp = 'String'
                fig_title = r'%s, Re = %g, $m^* = %g$, $U^* = %g$, $\zeta = %g%s$' % \
                    (str_tmp, Re, mstar, Ustar, 100 * zeta, r'\%')
            elif not strip_theory:
                fig_title = 'CFD Uref = %gm/s, Re = %g' % (Uref, Re)

            try:
                p = case['p']
            except:
                p = 1
            if 'choose_strips' in case.keys() and 'mode' in case.keys():
                choose_strips = case['choose_strips']
                mode = case['mode']
                Ncfd = len(case['choose_strips'])
            elif strip_theory \
                    and js['model'] == 1 \
                    and 'strip_positions' in js \
                    and js['strip_positions'] in [1, 3] and p > 0:
                jump = 2**p
                start = 2**(p - 1) - 1
                choose_strips = range(start, Ns, jump)
                mode = int((Ns + 1) / 2**p)
                Ncfd = mode
                #print("Ns = %d, Ncfd = %d, mode = %d" % (Ns, Ncfd, mode))
            else:
                choose_strips = range(Ns)
                Ncfd = Ns
                mode = Ns

        else:
            print("missing parameter file!")
            print("%s" % pfile)
            exit()
    
        ### import xml
        U = np.zeros((Ns))
        rho = np.zeros((Ns))
        mu = np.zeros((Ns))
        for ns in range(Ns):
            if strip_theory:
                casename = case['rootdir'] + case['cname'] + '/RESU_COUPLING/' + 'case_%d/' % nc
                dname = casename + 'fluid%d/' % (ns+1)
            else:
                casename = case['rootdir'] + case['cname'] + '/fluid/RESU/' + 'fluid_%d/' % nc
                dname = casename
    
            # check if case is missing, normally due to job failure
            if not os.path.isdir(casename):
                print("case %d missing!" % nc)
                return

            if strip_theory:
                root = ET.parse(dname + 'fluid%d_%d.xml' % (ns+1, nc)).getroot()
            else:
                root = ET.parse(dname + 'fluid_%d.xml' % nc).getroot()
            # fluid velocity
            for type_tag in root.findall('boundary_conditions/inlet/velocity_pressure/norm'):
                U[ns] = float(type_tag.text)
            c = 0
            for type_tag in root.findall('physical_properties/fluid_properties/property/initial_value'):
                if c == 0:
                    rho[ns] = float(type_tag.text)
                elif c == 1:
                    mu[ns] = float(type_tag.text)
                c += 1
    
            #print("ns = %d, U = %g m/s, rho = %g kg/m3, mu = %e Pa.s" % (ns, U[ns], rho[ns], mu[ns]))
    
        # get simulation data
    
        fname = 'Fxyz_xyzstr%d.bin' % (ns+1)
        time, Fx1, Fy1, Fz1, x1, y1, z1, Nt = read_xyz(dname, fname, -1)
    
        Fx = np.zeros((Nt, Ncfd))
        Fy = np.zeros((Nt, Ncfd))
        x = np.zeros((Nt, Ncfd))
        y = np.zeros((Nt, Ncfd))
        Cd = np.zeros((Nt, Ncfd))
        Cl = np.zeros((Nt, Ncfd))
    
        if nci == 0:
            # variables needed to find forced cases map (Morse & Williamson, 2009)
            Nc = len(case['nc'])
            Clfc = np.zeros((Ncfd, Nc))
            phi_mean = np.zeros((Ncfd, Nc))
            phi_rms = np.zeros((Ncfd, Nc))
            t_start = np.ones((Nc)) * t_start
            if 't_end' in case:
                tmax = np.ones((Nc)) * min(case['t_end'], max(time))
            else:
                tmax = np.zeros((Nc))
    
        #print("Nt, Ncfd = %d, %d" % (Nt, Ncfd))
    
        ####
        for ncfd, ns in enumerate(choose_strips):
            if strip_theory:
                dname = casename + 'fluid%d/' % (ns + 1)
                fname = 'Fxyz_xyzstr%d.bin' % (ns + 1)
            time, Fx1, Fy1, Fz1, x1, y1, z1, tmp = read_xyz(dname, fname, Nt)
            Fx[:, ncfd] = Fx1
            Fy[:, ncfd] = Fy1
            x[:, ncfd] = x1
            y[:, ncfd] = y1
    
        # calculate drag and lift coefficients
        A = diameter * Lxyz[2]  # m2
    
        dt = time[1] - time[0]
    
        # drag and lift coefficients
        
        for ncfd, ns in enumerate(choose_strips):
            Cd[:, ncfd] = 2.0 * Fx[:, ncfd] / (rho[ncfd] * U[ncfd]**2 * A)
            Cl[:, ncfd] = 2.0 * Fy[:, ncfd] / (rho[ncfd] * U[ncfd]**2 * A)
    
        # start time for calculations
        if tmax[nci] < t_start[nci]:
            # automated
            tmax[nci] = max(time)
            if tmax[nci] < t_start[nci] * 1.2 and tmax[nci] < t_start[nci] + 1.0:
                fSt_approx = 0.2 * np.mean(U) / diameter
                # number of wavelengths
                Nw = 20.0
                t_start[nci] = max(0.0, tmax[nci] - Nw / fSt_approx)
            for nt in range(Nt):
                if time[nt] > t_start[nci]:
                    nt0 = nt
                    break
            t_start[nci] = time[nt0]
    
            if js['FtoSI']:
                # check for y oscillation convergence before t_start default
                Nprd = 20.0 # check over number of Nprd periods
                Dnt = int(Nprd / (fc * mode * dt))
                val0 = 1e6
                for nt in range(Dnt, Nt, Dnt):
                    val1 = np.max(y[nt-Dnt:nt,:])
                    if abs(val1 - val0) < val1 * 5e-3:
                        break
                    val0 = val1
                    #print("    nt = %d, val0,1 = %g, %g" % (nt, val0, val1))
                #print("nt = %d, Nt = %d, time = %gs" % (nt, Nt, time[nt]))

                # use this time only when less than original t_start
                #   and more than typical basic VS development time
                if time[nt] < t_start[nci] and time[nt] > 2.0:
                    t_start[nci] = time[nt]

        print('%d, t_start = %gs, tmax = %gs' % (nc, t_start[nci], tmax[nci]))

        def get_peaks(time, t_start, tmax, y, Npeaks_min, Npeaks_max):
            # find time range of interest for IFS statistics
            nt = np.argmax(time > t_start[nci])
            nt = max(nt, 1)
            Nt = np.argmax(time >= tmax[nci])
            ntr = [nt for nt in range(nt, Nt-1)]
            
            # find slice with maximum amplitude
            myvals = np.zeros(len(choose_strips))
            for ncfd, ns in enumerate(choose_strips):
                myvals[ncfd] = np.amax(y[:,ncfd])
            ncfd = np.argmax(myvals)
            
            # find all peaks in sample signal (work on first slice and assume all have same periods)
            peaks = []
            for n in ntr:
                if y[n, ncfd] > y[n-1, ncfd] and y[n, ncfd] > y[n+1, ncfd]:
                    peaks.append(n)
            
            if len(peaks) > 0:
                # first peak index
                npk0 = peaks[0]
                if max(peaks) > peaks[0]:
                    # final peak index
                    npk1 = max(peaks)
                else:
                    print("only one peak detected! Spectra not valid")
                    npk1 = Nt-1
                # number of peaks
                Npeaks = len(peaks)
                print("number of peaks in phi signal sample = %d" % Npeaks)
                # allow for checking # of peaks at end
                Npeaks_min = min(Npeaks_min, Npeaks)
                Npeaks_max = max(Npeaks_max, Npeaks)
                nppr = [n for n in range(npk0, npk1+1)] # peak to peak range
            else:
                print("   --> could not find any peaks!")
                nppr = [nt for nt in range(Nt)]
            
            return nppr, Npeaks_min, Npeaks_max, peaks
            
        if strip_theory:
        
            nppr, Npeaks_min, Npeaks_max, peaks = \
                get_peaks(time, t_start, tmax, y, Npeaks_min, Npeaks_max)
    
            # find correlation between y and fy as time progresses
            for ncfd in range(Ncfd):
                # find amplitude in last nE full oscillations
                maxy = np.amax(y[nppr, ncfd])
                yDmax[nci] = max(maxy / js['D'], yDmax[nci])
    
            for ncfd in range(Ncfd):
                
                # find phase difference using Hilbert transformation
                signal1 = [y[n, ncfd] for n in nppr]
                signal2 = [Fy[n, ncfd] for n in nppr]
                analytic_signal1 = hilbert(signal1)
                instantaneous_phase1 = np.unwrap(np.angle(analytic_signal1))
                analytic_signal2 = hilbert(signal2)
                instantaneous_phase2 = np.unwrap(np.angle(analytic_signal2))
                DphaseH = instantaneous_phase2-instantaneous_phase1
                phi_mean[ncfd, nci] = np.mean(DphaseH)
                phi_rms[ncfd, nci] = np.std(DphaseH)
                
                #print("ncfd = %d, phi_mean = %g" % (ncfd, phi_mean[ncfd, nci]))
                
                # plots to check phi calculation
                if False: #nci == 3 and ncfd == 0:
                    
                    t = [time[n] for n in nppr]
                    amplitude_envelope1 = np.abs(analytic_signal1)
                    amplitude_envelope2 = np.abs(analytic_signal2)
                    
                    fig, (ax0, ax1, ax2) = mpl.subplots(nrows=3)
                    
                    ax0.plot(t, signal1, label='signal1')
                    ax0.plot(t, analytic_signal1.real, ':', label='real')
                    ax0.plot(t, analytic_signal1.imag, ':', label='imag')
                    ax0.plot(t, amplitude_envelope1, label='envelope')
                    ax0.set_xlabel("time")
                    ax0.legend()
                    
                    ax1.plot(t, signal2, label='signal2')
                    ax1.plot(t, analytic_signal2.real, ':', label='real')
                    ax1.plot(t, analytic_signal2.imag, ':', label='imag')
                    ax1.plot(t, amplitude_envelope2, label='envelope')
                    ax1.set_xlabel("time")
                    ax1.legend()
                    
                    ax2.plot(t, DphaseH, label='phase1')
                    #ax2.plot(t, instantaneous_phase1, label='phase1')
                    #ax2.plot(t, instantaneous_phase2, label='phase2')
                    ax2.set_xlabel("time")
                    ax2.legend()
                    
                    fig.tight_layout()
                    mpl.show()
                    exit()
        else:
            
            nppr, Npeaks_min, Npeaks_max, peaks = \
                get_peaks(time, t_start, tmax, Cl, Npeaks_min, Npeaks_max)
        
        # calculate spectra
        f1, s1 = spectrum(time[nppr], Cl[nppr, 0])
        Nfr = len(f1)
        ss = np.zeros((Nfr, Ncfd))
        sCd = np.zeros((Nfr, Ncfd))
        sCl = np.zeros((Nfr, Ncfd))
    
        # limit the range over which the fundamental/Stroudal frequancy can be found
        fStmin = 0.164 * np.mean(U) / diameter
        for nfmin,f in enumerate(f1):
            if fStmin * 0.7 < f:
                break
        fStmax = 0.2 * np.mean(U) / diameter
        for nfmax,f in enumerate(f1):
            if fStmax * 1.4 < f:
                break
        if not js['FtoSI']: # forced vibrations
            nfmin = 1
            nfmax = len(f1 - 1)
        #print("min/max frequency = %g/%gHz" % (ftmp[nfmin], ftmp[nfmax]))

        # Cl amplitude at cable natural frequency
        for ncfd in range(Ncfd):

            # Strouhal frequency
            ftmpl, sCl1 = spectrum(time[nppr], Cl[nppr, ncfd])
            if nfmin < nfmax:
                iflm = np.argmax(sCl1[nfmin:nfmax]) + nfmin
                Slfreq = ftmpl[iflm]
            else:
                Slfreq = -1

            ftmpd, sCd1 = spectrum(time[nppr], Cd[nppr, ncfd])
            if nfmin < nfmax:
                ifdm = np.argmax(sCd1[2*nfmin:2*nfmax]) + 2*nfmin
                Sxfreq = ftmpd[ifdm]
            else:
                Sxfreq = -1
            #print("ncfd = %d, VS (Cd, Cl) frequency = (%g, %g) Hz" % (ncfd, ftmpCd[ifdm], Slfreq[ncfd]))
            #print("  Strouhal = %g (val ~ 0.224)" % (Slfreq[ncfd] * diameter / U))
            
            fqCl[nci] += Slfreq * diameter / (U[ncfd] * mode)
            fqCd[nci] += Sxfreq * diameter / (U[ncfd] * mode)

            sCd[:, ncfd] = sCd1
            sCl[:, ncfd] = sCl1
    
            if strip_theory:
                fc_md = fc * mode
                # find amplitude at cable natural frequency
                for ns in range(1, Nfr):
                    if f1[ns] > fc_md:
                        n1 = ns - 1
                        n2 = ns
                        w1 = (f1[n2] - fc_md) / (f1[n2] - f1[n1])
                        w2 = (fc_md - f1[n1]) / (f1[n2] - f1[n1])
                        # need to multiply spectrum by two to get amplitude
                        Clfc[ncfd, nci] = 2.0 * (w1 * sCl1[n1] + w2 * sCl1[n2])
                        break
                #print("  Cl amplitude = %g at fc * mode = %gHz" % (Clfc[ncfd, nci], fc_md))

        if strip_theory and len(peaks) > 1:
            # limit frequency range
            nfmin = 1
            nfmax = len(f1 - 1)
            
            non_sin_Cl = 0.0
            non_sin_y = 0.0
            for ncfd in range(Ncfd):
                ftmpy, s1 = spectrum(time[nppr], y[nppr, ncfd])
                ss[:, ncfd] = s1
                
                #nfmin = np.argmax(ftmpy>(fc * mode / 2 + fc * 0.05)) # ignore lower mode peaks
                tmp_max = np.max(s1[nfmin:nfmax])
                ifym = np.argmax(s1[nfmin:nfmax]) + nfmin
                #nfmax = 2*ifym
    
                peaks2, _ = find_peaks(s1[nfmin:nfmax], height=tmp_max*0.1)
                peaks2 += nfmin
                
                """fig, ax = mpl.subplots()
                ax.loglog(ftmpy, s1)
                ax.loglog(ftmpy[peaks], s1[peaks], 'x')
                mpl.show()"""
                
                yfreq = ftmpy[ifym]
                fqy[nci] += yfreq * diameter / U[ncfd]
                
                # first peak
                """if len(peaks) == 1:
                    fqy[nci] += ftmpy[peaks[0]] * diameter / (U * mode) # average over all cfd slices
                elif len(peaks) > 1:
                    # choose peak that's furthest from resonance
                    #print(ftmpy[ifym])
                    #print(ftmpy[peaks])
                    #print(np.absolute(ftmpy[peaks] - fc * mode))
                    choose = np.argmax(np.absolute(ftmpy[peaks] - fc * mode))
                    #print(choose)
                    fqy[nci] += ftmpy[peaks[choose]] * diameter / (U * mode) # average over all cfd slices
                    #print(ftmpy[peaks[choose]] * diameter / (U))
                else:
                    fqy[nci] += ftmpy[ifym] * diameter / (U * mode) # average over all cfd slices"""
                
                #print("ncfd = %d, y frequency = %g Hz" % (ncfd, yfreq))

                non_sin_Cl += np.std(Cl[nppr, ncfd]) * np.sqrt(2.0) / (np.amax(Cl[nppr, ncfd]) * Ncfd)
                non_sin_y += np.std(y[nppr, ncfd]) * np.sqrt(2.0) / (np.amax(y[nppr, ncfd]) * Ncfd)

            #print("non_sin_Cl = %g, non_sin_y = %g" % (non_sin_Cl, non_sin_y))
            #print(fqy[nci])
            #exit()
            
            fqy[nci] /= Ncfd

        # write measurements and spectra to file for use in database analysis
        if write_data:
            # only save lowest freqencies
            Nfs = Nfr
            for nfs in range(Nfr):
                if f1[nfs] > 10.0 * max(fc, fqCl[nci] * U / diameter) * mode:
                    Nfs = min(nfs, Nfr)
                    break

            fid = open('GetTimeEvolutionStrip_%s.dat' % input_file, 'a')
            fid.write('{')

            fid.write('"nci" : %d, "Ns" : %d, ' %
                 (nci, Ns))

            fid.write('"tmax" : %12e, "D" : %12e, "U" : %12e, "rho" : %12e, ' %
                 (tmax[nci], diameter, U, rho))

            if strip_theory:
                fid.write('"model" : %d, "ml" : %12e, "zeta" : %12e, "FtoSI" : %d, ' %
                     (smodel, js['ml'], js['zeta'], js["FtoSI"]))
                if smodel == 0:  # MSD
                    fid.write('"k" : %12e, ' %
                         (k))
                elif smodel >= 1:  # String
                    fid.write('"H" : %12e, "L" : %12e, ' %
                         (H, js['L']))
                if js['FtoSI']:
                    if type(js['ya0']) is list:
                        ya0 = js['ya0'][nci]
                    else:
                        ya0 = js['ya0']
                    fid.write('"ya0" : %12e, ' %
                         (ya0))

            fid.write('"Ustar" : %12e, "yDmax" : %12e, "fc" : %12e, "fqy" : %12e, "fqCl" : %12e, "fqCd" : %12e,' %
                 (Ustar, yDmax[nci], fc, fqy[nci], fqCl[nci], fqCd[nci]))

            fid.write('"non_sin_Cl": %12e, "non_sin_y": %12e, ' %
                 (non_sin_Cl, non_sin_y))

            Ll = ["Clfc", "phi_mean", "phi_rms", "f1", "ECl", "Ey"]
            Nn = [Ncfd, Ncfd, Ncfd, Nfs, Nfs*Ncfd, Nfs*Ncfd]
            Ww = [Clfc[:, nci], phi_mean[:, nci], phi_rms[:, nci],
                f1[:Nfs], sCl[:Nfs, :], ss[:Nfs, :]]

            for ll, nn, ww in zip(Ll, Nn, Ww):
                txt = np.array2string(ww,
                    separator=',', threshold=nn,
                    formatter={'float_kind':lambda ww: "%g" % ww}
                    ).replace("\n", "")
                fid.write('"%s" : %s, ' % (ll, txt))

            fid.write('"nc" : %d, "pfile" : "%s"}\n' %
                 (nc, pfile))
            fid.close()
    
        ###

        #### Plotting ####
        if show_plot or save_plot:
            if Ndtime:
                NdU_d = U / diameter
            else:
                NdU_d = 1.0
        
            Nfc_plot = 4.0

            lines = ('--', '-', '--', '-.', '--', '-', '--', '-.')
            Nl = len(lines)
            dash = [(None, None) for n in range(Nl)]
            dash[2] = (6, 2)
            dash[3] = (1, 5)
            dash[6] = (6, 2)
            colours = ['C0', 'C2', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
            fig1a, a1 = mpl.subplots(1)
            fig1b, a2 = mpl.subplots(1)

            if strip_theory:
                fig1c, a3 = mpl.subplots(1)

            # plot drag
            for ncfd in range(Ncfd):
                a1.plot(time * NdU_d, Cd[:, ncfd],
                    str(lines[ncfd % Nl]), color=colours[ncfd % 10],
                    label=r'$%d$' % (choose_strips[ncfd]+1), dashes=dash[ncfd % Nl])
            # plot lift
            for ncfd in range(Ncfd):
                a2.plot(time * NdU_d, Cl[:, ncfd],
                    str(lines[ncfd % Nl]), color=colours[ncfd % 10],
                    label=r'$%d$' % (choose_strips[ncfd]+1), dashes=dash[ncfd % Nl])
            if Ndtime:
                a1.set_xlabel(r"$tU / D$")
                a2.set_xlabel(r"$tU / D$")
            else:
                a1.set_xlabel("time [s]")
                a2.set_xlabel("time [s]")
            a1.set_ylabel(r"$C_d$")
            a2.set_ylabel(r"$C_l$")
            if strip_theory:
                freq = fc
            else:
                St = 0.164
                freq = St * NdU_d
            dNt = int(Nfc_plot / (freq * mode * dt))
            Clmax = np.max(abs(Cl[max(0,Nt-dNt):Nt-1,:]))
            try:
                Cdmin = np.min(abs(Cd[200:Nt-1,:]))
                Cdmax = np.max(abs(Cd[200:Nt-1,:]))
            except:
                Cdmin = 1.0
                Cdmax = 2.0
            if strip_theory and not Nfc_plot == None:
                tx1 = 0.0 #(tmax[nci] - Nfc_plot / (fc * mode)) * NdU_d
                #tx1 = t_start[nci] * NdU_d
                tx2 = tmax[nci] * NdU_d
                a1.set_xlim([tx1, tx2])
                a2.set_xlim([tx1, tx2])
            a1.set_ylim([Cdmin - 0.05, Cdmax + 0.05])
            a2.set_ylim([- (Clmax + 0.05), Clmax + 0.05])
            """a1.set_ylim([1.3, 2.9])
            a1.set_xticks([478, 480, 482, 484])
            a1.set_yticks([1.6, 2.0, 2.4, 2.8])
            a2.set_ylim([-1.03, 1.03])
            a2.set_xticks([478, 480, 482, 484])
            a2.set_yticks([-1.0, 0.0, 1.0])"""
            if incl_title:
                a1.set_title(fig_title)
                a2.set_title(fig_title)

            a1.legend(ncol=2)
            a2.legend(ncol=2)
    
            if strip_theory:
                ymax = np.max(abs(y[max(0,Nt-dNt):Nt-1,:]))
                for ncfd in range(Ncfd):
                    a3.plot(time * NdU_d, y[:, ncfd] / diameter,
                        str(lines[ncfd % Nl]), color=colours[ncfd % 10],
                        label=r'$%d$' % (choose_strips[ncfd]+1), dashes=dash[ncfd % Nl])
                if Ndtime:
                    a3.set_xlabel(r"$tU / D$")
                else:
                    a3.set_xlabel("time [s]")
                a3.set_ylabel(r"$\xi / D$")
                if strip_theory and not Nfc_plot == None:
                    a3.set_xlim([tx1, tx2])
                """a3.set_ylim([ymax * 1.1 * n / diameter for n in [-1, 1]])
                a3.set_ylim([-0.65, 0.65])
                a3.set_xticks([478, 480, 482, 484])"""
                if incl_title:
                    a3.set_title(fig_title)
                a3.legend(ncol=2)
    
            if save_plot:
                fig1a.tight_layout()
                fig1b.tight_layout()
                fig1a.savefig(folder_plots + "/time_Cd_%d.png" % nc, dpi=400)
                fig1b.savefig(folder_plots + "/time_Cl_%d.png" % nc, dpi=400)
                if strip_theory:
                    fig1c.tight_layout()
                    fig1c.savefig(folder_plots + "/time_y_%d.png" % nc, dpi=400)
    
            if (type(peaks) is np.ndarray or type(peaks) is list) and len(peaks) > 5 or \
                    type(peaks) is int and peaks > 5:
                if strip_theory:
                    fig2, axs2 = mpl.subplots(3)
                    a3 = axs2[2]
                else:
                    fig2, axs2 = mpl.subplots(2)
                a1 = axs2[0]
                a2 = axs2[1]
                for ncfd in range(Ncfd):
                    if max(sCd[:,ncfd]) > 0.0:
                        a1.loglog(f1 / NdU_d, sCd[:,ncfd],
                            label='%d' % (choose_strips[ncfd]+1),
                            color=colours[ncfd % 10])
                for ncfd in range(Ncfd):
                    if max(sCl[:,ncfd]) > 0.0:
                        a2.loglog(f1 / NdU_d, sCl[:,ncfd],
                            label='%d' % (choose_strips[ncfd]+1),
                            color=colours[ncfd % 10])
                if Ndtime:
                    a1.set_xlabel(r"$fd / U$")
                    a2.set_xlabel(r"$fd / U$")
                else:
                    a1.set_xlabel("f [Hz]")
                    a2.set_xlabel("f [Hz]")
                a1.set_ylabel(r"psd $C_d$")
                a2.set_ylabel(r"psd $C_l$")
                a1.set_title(fig_title + ", $t > %.3g$s" % t_start[nci])
                a1.legend()
                a2.legend()
        
                if strip_theory:
                    for ncfd in range(Ncfd):
                        if max(ss[:, ncfd]) > 0.0:
                            a3.loglog(f1 / NdU_d, ss[:, ncfd],
                            label='%d' % (choose_strips[ncfd]+1),
                            color=colours[ncfd % 10])
                    if Ndtime:
                        a3.set_xlabel(r"$fd / U$")
                    else:
                        a3.set_xlabel("f [Hz]")
                    a3.set_ylabel(r"psd $y$")
                    a3.legend()
                    a1.set_xlabel("")
        
                if save_plot:
                    fig2.savefig(folder_plots + "/Cl_y_spec_%d.png" % nc, dpi=400)
    
            if show_plot:
                mpl.show()
            else:
                mpl.close('all')
    
    if strip_theory:
        if smodel == 0:
            print("MSD, m* = %g, Ncfd = %d, zeta = %g" % (mstar, Ncfd, zeta))
        elif smodel >= 1:
            print("String, m* = %g, Ncfd = %d, zeta = %g" % (mstar, Ncfd, zeta))
    
        print("\n%12s, %12s, %12s, %12s" %
              ("U*", "ymax/D", "fy", "mean Cx,"))
    
        for nci, nc in enumerate(case['nc']):
    
    
            if smodel == 0:
                if type(js['k']) is list:
                    fc = np.sqrt(js['k'][nc-1] / js['ml']) / (2.0 * np.pi)
                else:
                    fc = np.sqrt(js['k'] / js['ml']) / (2.0 * np.pi)
            elif smodel >= 1:
                if type(js['H']) is list:
                    fc = 0.5 / js['L'] * np.sqrt(js['H'][nc-1] / js['ml'])
                else:
                    fc = 0.5 / js['L'] * np.sqrt(js['H'] / js['ml'])

            if type(js['speeds']) is list:
                U = js['speeds'][nc-1]
            else:
                U = js['speeds']
    
            ncfd = 0
    
            if type(js['ya0']) is list:
                ya0 = js['ya0'][nc-1]
            else:
                ya0 = js['ya0']
            fn = js['D'] / U * fc
            Ustar = 1.0 / fn
            print("%12g, %12g, %12g, %12g," %
                  (Ustar, yDmax[nci], fqy[nci], Cxm[nci]))
    else:
        print("Cx mean = %g, Cy mean = %g, Cl max = %g" % (Cxm[nci], Cym[nci], Cymax[nci]))
                  
    print("min, max number of peaks in phi signal sample = %d, %d" % (Npeaks_min, Npeaks_max))

# loop over all cases
for nca, case in enumerate(datalist):
    run_datalist(nca, case)

