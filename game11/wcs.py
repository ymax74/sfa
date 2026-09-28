#!/Users/ymax/.local/pipx/venvs/jupyterlab/bin/python3
# ###!/home/max/.local/share/pipx/venvs/astropy/bin/python3
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

import time

import os
import subprocess

import requests

import glob

import argparse

import re

bot_token = "7922143151:AAGaul6uQFxtaN9TYGeB_rDYrywoaaAKhsY"

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str)
parser.add_argument("--o", type=str)
parser.add_argument("--fnt", type=str)
parser.add_argument("--ch", type=str)
args = parser.parse_args()

# filelist = glob.glob(f'{args.i}/{args.fnt}*[!wcs].fits')
filelist = glob.glob(f'{args.i}/{args.fnt}*[!wcs].fit')
filelist.sort()

print(filelist)
# if int(args.ch)!=0:
#     response = requests.get(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={args.ch}&parse_mode=Markdown&text={filelist}')

for filename in filelist:
    try:
        bn, en = os.path.splitext(filename)
        fname = os.path.splitext(os.path.basename(filename))[0]
        new_file_name = f"{args.o}/{fname}.fits"
        print(filename,new_file_name)
        pipe = os.popen(f"cp {filename} {new_file_name}")
        exit_status = pipe.close()
        hdulist = fits.open(new_file_name)
    except Exception:
        continue
    # c = SkyCoord('%s %s'%(hdulist[0].header['RA'],hdulist[0].header['DEC']), unit=(u.hourangle, u.deg))
    c = SkyCoord('%s %s'%(hdulist[0].header['OBJCTRA'],hdulist[0].header['OBJCTDEC']), unit=(u.hourangle, u.deg))
    cmd = ['solve-field', '--use-source-extractor' ,\
           '--source-extractor-path',\
           'sex',\
           '--scale-units', 'arcsecperpix', \
           '--scale-low', '0.1', '--scale-high','0.5', \
           '--no-plots', '--overwrite', '--crpix-center',\
           '--ra', '%f' % (c.ra.degree), '--dec', '%f' % (c.dec.degree), \
           '--radius', '2', f"{new_file_name}"]

    code = 1
    print(' '.join(cmd))
    try:
        process = subprocess.Popen(cmd)
        code = process.wait(timeout=40)

    except subprocess.TimeoutExpired:
        print(code)
        os.popen(f"" 'rm ' + os.path.dirname(filename) + '/*.axy')
        continue

    # bn, en = os.path.splitext(filename)
    # new_file_name = os.path.splitext(os.path.basename(filename))[0]
    print(f"let find file: {args.o}/{fname}.new ")
    if os.path.exists(f"{args.o}/{fname}.new"):
        os.popen(f"cp {args.o}/{fname}.new {args.o}/{fname}_wcs.fits")
        time.sleep(0.5)
    else:
        print('WCS has not added')
    os.popen(f"rm {args.o}/*.xyls {args.o}/*.axy {args.o}/*.corr {args.o}/*.match {args.o}/*.rdls  {args.o}/*.wcs {args.o}/*.solved {args.o}/*.new")
# parser = argparse.ArgumentParser()
# parser.add_argument("--path", type=str)
# parser.add_argument("--fnt", type=str)
# parser.add_argument("--ch", type=str)
# args = parser.parse_args()
#
# filelist = glob.glob(f'{args.path}/{args.fnt}*[!wcs].fits')
# filelist.sort()
#
# print(filelist)
# if int(args.ch)!=0:
#     response = requests.get(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={args.ch}&parse_mode=Markdown&text={filelist}')
#
# for filename in filelist:
#     try:
#         print(filename)
#         hdulist = fits.open(filename)
#     except Exception:
#         continue
#     c = SkyCoord('%s %s'%(hdulist[0].header['RA'],hdulist[0].header['DEC']), unit=(u.hourangle, u.deg))
#     cmd = ['solve-field', '--use-source-extractor' ,\
#            '--source-extractor-path',\
#            'sex',\
#            '--scale-units', 'arcsecperpix', \
#            '--scale-low', '0.1', '--scale-high','0.5', \
#            '--no-plots', '--overwrite', '--crpix-center',\
#            '--ra', '%f' % (c.ra.degree), '--dec', '%f' % (c.dec.degree), \
#            '--radius', '2', filename]
#
#     code = 1
#     print(' '.join(cmd))
#     try:
#         process = subprocess.Popen(cmd)
#         code = process.wait(timeout=40)
#
#     except subprocess.TimeoutExpired:
#         print(code)
#         os.popen('rm ' + os.path.dirname(filename) + '/*.axy')
#         continue
#
#     bn, en = os.path.splitext(filename)
#     new_file_name = os.path.splitext(os.path.basename(filename))[0]
#     print('let find file: %s.new '%(bn))
#     if(os.path.exists('%s.new'%(bn))):
#         os.popen('cp %s.new %s/%s_wcs.fits' % (bn, args.path, new_file_name))
#         os.popen('rm '+os.path.dirname(filename)+'/*.xyls')
#         os.popen('rm '+os.path.dirname(filename)+'/*.axy')
#         os.popen('rm '+os.path.dirname(filename)+'/*.corr')
#         os.popen('rm '+os.path.dirname(filename)+'/*.match')
#         os.popen('rm '+os.path.dirname(filename)+'/*.rdls')
#         os.popen('rm '+os.path.dirname(filename)+'/*.wcs')
#         os.popen('rm '+os.path.dirname(filename)+'/*.solved')
#         os.popen('rm ' + os.path.dirname(filename) + '/*.new')
#         # print(int(args.ch))
#         if int(args.ch)!=0:
#             print(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={args.ch}&parse_mode=Markdown&text={re.sub(r"_", "\_", filename, flags=re.IGNORECASE)}')
#             response = requests.get(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={args.ch}&parse_mode=Markdown&text={re.sub(r"_", "\_", filename)} is solved')
#         time.sleep(0.5)
#     else:
#         print('WCS has not added')
#         if int(args.ch)!=0:
#             response = requests.get(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={args.ch}&parse_mode=Markdown&text={re.sub(r"_", "\_", filename)} is not solved')
#
#         os.popen('rm ' + os.path.dirname(filename) + '/*.xyls')
#         os.popen('rm ' + os.path.dirname(filename) + '/*.axy')
