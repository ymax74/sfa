import healpy as hp
from math import *
import numpy as np
import pandas as pd

import gc

from astropy import units as u
from astropy.coordinates import SkyCoord

import os
import glob

def tangential_coordinates(ra,dec,RA,DEC):
    ksi = cos(dec)*sin(ra-RA)/(sin(dec)*sin(DEC)+cos(dec)*cos(DEC)*cos(ra-RA))
    eta = (sin(dec)*cos(DEC)-cos(dec)*sin(DEC)*cos(ra-RA))/(sin(dec)*sin(DEC)+cos(dec)*cos(DEC)*cos(ra-RA))
    return ksi,eta

def tangential_coordinates_np(ra,dec,RA,DEC):
    denominator = np.sin(dec)*np.sin(DEC)+np.cos(dec)*np.cos(DEC)*np.cos(ra-RA)
    ksi = np.cos(dec)*np.sin(ra-RA)/denominator
    eta = (np.sin(dec)*np.cos(DEC)-np.cos(dec)*np.sin(DEC)*np.cos(ra-RA))/denominator
    return ksi,eta

def equatorial_coordinates(ksi,eta,RA,Dec):
    x,y,z = np.dot(np.array([[-sin(RA),-cos(RA) * sin(Dec),cos(RA) * cos(Dec)],
                             [cos(RA),-sin(RA) * sin(Dec),sin(RA) * cos(Dec)],
                             [0,cos(Dec),sin(Dec)]]),
                   np.array([ksi,eta,1]))/sqrt(1+ksi*ksi+eta*eta)
    ra = atan2(y,x)
    dec = atan2(z,sqrt(x*x+y*y))
    if(ra<0):
        ra+=2*pi
    return ra,dec

def gaiadr3f(RA,DEC,FOV):
    c = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
    gaia_path = '/mnt/data_storage/GaiaDR3/cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source'
    fl = glob.glob(f'{gaia_path}/GaiaSource*')
    minhpx = np.zeros(len(fl))
    maxhpx = np.zeros(len(fl))
    for k in range(len(fl)):
        fp = fl[k].split('GaiaSource_')[1].split('.')[0].split('-')
        minhpx[k] = int(fp[0])
        maxhpx[k] = int(fp[1])
    ndot = 5
    p = np.radians(np.linspace(-FOV/2,FOV/2, num=ndot))
    flist=[]
    for i in range(ndot):
        for j in range(ndot):
            tx = p[j];ty=p[i]
            ra,dec = equatorial_coordinates(tx,ty,c.ra.radian,c.dec.radian)
            ipix = hp.ang2pix(256, degrees(ra), degrees(dec), nest=True, lonlat=True)
            for k in range(len(minhpx)):
                if (minhpx[k] <= ipix <= maxhpx[k]):
                    if(not (fl[k] in flist)):
                        flist.append(fl[k])
    outp=[]
    for fn in flist:
        df = pd.read_csv(fn,compression='gzip', comment='#')
        tx,ty = np.degrees(tangential_coordinates_np(np.radians(df['ra']),np.radians(df['dec']),c.ra.radian,c.dec.radian))
        indx = np.where((tx>-FOV/2)&(tx<FOV/2)&(ty>-FOV/2)&(ty<FOV/2))
        outp.append(df.iloc[indx])
        del df
        gc.collect()
    return pd.concat(outp, ignore_index=True)
#     result_df.to_csv('gaiadr3.csv', index=False)
