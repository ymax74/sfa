import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import ctypes
import os

# _cpy = ctypes.CDLL('libshapelet.so')
_cpy = ctypes.CDLL(os.path.abspath('libshapelet.so'))

#1
_cpy.cshbox.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,
                        ctypes.POINTER(ctypes.c_double),ctypes.c_int, ctypes.POINTER(ctypes.c_double))

#2 void cshvariate(double xc, double yc, double flux, double bkg, double b, int nmax, double* shf, int boxsz, double* J, double* I)
_cpy.csh1point.argtypes = (ctypes.c_double,ctypes.c_int,ctypes.c_double,
                        ctypes.POINTER(ctypes.c_double),ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
#3
_cpy.cshapprox.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,
                        ctypes.POINTER(ctypes.c_double),ctypes.c_int, ctypes.POINTER(ctypes.c_double))

_cpy.cshapproxw.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,
                        ctypes.POINTER(ctypes.c_double),ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
# 4
_cpy.cshFXY.argtypes = (ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))

_cpy.csh_radius.argtypes = (ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
_cpy.csh_radius.restype = ctypes.c_double
# 5
_cpy.photocenter.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))

_cpy.findsrc.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))

_cpy.cshposfluxbkg.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
_cpy.cshposfluxbkg2.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
# 6
_cpy.csh2points.argtypes = (ctypes.c_double,ctypes.c_int,ctypes.c_double,
                        ctypes.POINTER(ctypes.c_double),ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))

_cpy.cshellipticity.argtypes = (ctypes.c_double,ctypes.c_int,ctypes.POINTER(ctypes.c_double))
_cpy.cshellipticity.restype = ctypes.c_double
# double cshellipticity(double b, int nmax, double* shf)
_cpy.csh_set_to_value.argtypes = (
                                  ctypes.c_int,#int X
                                  ctypes.c_int, #int Y
                                  ctypes.c_int, #int r
                                  ctypes.c_double, #double v
                                  ctypes.c_int, #int boxsz
                                  ctypes.POINTER(ctypes.c_double)#double* I
                                  )
# csh_set_to_value(int n, int* X, int* Y, int r, double v, int boxsz, double* I)

def shbox(x,y,b,nmax,shf,boxsz):
    J = np.zeros((boxsz,boxsz))
    global _cpy
    _cpy.cshbox(ctypes.c_double(x),ctypes.c_double(y),ctypes.c_double(b),ctypes.c_int(nmax),
                shf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_int(boxsz),
                J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return J

def sh1point(x,y,f,B,b,sl,nmax,shf,J):
    global _cpy
    boxsz = np.shape(J)[0]
    P = np.array([B,f,x,y])
    _cpy.csh1point(ctypes.c_double(b),ctypes.c_int(nmax),ctypes.c_double(sl),\
                    shf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_int(boxsz),\
                    J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                    P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return P

def sh2points(x1,y1,f1,x2,y2,f2,B,b,sl,nmax,shf,J):
    global _cpy
    boxsz = np.shape(J)[0]
    P = np.array([B,f1,x1,y1,f2,x2,y2])
    _cpy.csh2points(ctypes.c_double(b),ctypes.c_int(nmax),ctypes.c_double(sl),\
                    shf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_int(boxsz),\
                    J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                    P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return P

def shapprox(x,y,b,nmax,boxsz,J):
    global _cpy
    Q = int((nmax+1)*(nmax+2)/2+1)
    shf = np.zeros(Q)
    _cpy.cshapprox(ctypes.c_double(x),ctypes.c_double(y),ctypes.c_double(b),ctypes.c_int(nmax),
                shf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_int(boxsz),
                J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return shf

def shapproxw(x,y,b,nmax,boxsz,J,W):
    global _cpy
    Q = int((nmax+1)*(nmax+2)/2+1)
    shf = np.zeros(Q)
    _cpy.cshapproxw(ctypes.c_double(x),ctypes.c_double(y),ctypes.c_double(b),ctypes.c_int(nmax),
                shf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_int(boxsz),
                J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return shf

def shFXY(b,nmax,shf):
    global _cpy
    xyfb = np.zeros(4)
    _cpy.cshFXY(ctypes.c_double(b),ctypes.c_int(nmax),shf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),xyfb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    xyfb[3]=shf[0]
    return xyfb

def shradius(b,nmax,shf):
    global _cpy
    return _cpy.csh_radius(ctypes.c_double(b),ctypes.c_int(nmax),shf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))


def shposfluxbkg(b,nit,J):
    global _cpy
    boxsz = np.shape(J)[0]
    p = np.zeros(4)
    _cpy.cshposfluxbkg(ctypes.c_int(boxsz),ctypes.c_int(nit),ctypes.c_double(b),J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return p

def shposfluxbkg2(b,nit,nmax,J):
    global _cpy
    boxsz = np.shape(J)[0]
    p = np.zeros(4)
    _cpy.cshposfluxbkg2(ctypes.c_int(boxsz),ctypes.c_int(nit),ctypes.c_double(b),ctypes.c_int(nmax),J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return p

def shphotocenter(J):
    global _cpy
    boxsz = np.shape(J)[0]
    p = np.zeros(5)
    _cpy.photocenter(ctypes.c_int(boxsz),p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return p

def shfindsrc(J):
    global _cpy
    boxsz = np.shape(J)[0]
    p = np.zeros(2)
    _cpy.findsrc(ctypes.c_int(boxsz),p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return p

def shellipticity(b,nmax,shf):
    global _cpy
    e = _cpy.cshellipticity(ctypes.c_double(b), ctypes.c_int(nmax), shf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return e

def shsolve(nmax,J):
    global _cpy
    boxsz = np.shape(J)[0]
    B,f,xp,yp,snr = shphotocenter(J)
    bs, be = 0.1, boxsz/2
    for k in range(3):
        bset = np.linspace(bs, be, 10)
        err = np.zeros(np.size(bset))
        for l in range(np.size(bset)):
            psf = shapprox(xp, yp, bset[l], nmax, boxsz, J)
            PSF = shbox(xp, yp, bset[l], nmax, psf, boxsz)
            err[l] = np.std(J - PSF)
        if(np.argmin(err)!=0):
            bs = bset[np.argmin(err) - 1]
        else:
            bs = bset[0]
        if (np.argmin(err) != 9):
            be = bset[np.argmin(err) + 1]
        else:
            be = bset[9]
    b = (bs + be) / 2.0
    psf = shapprox(xp, yp, b, nmax, boxsz, J)
    dx, dy, f, B = shFXY(b, nmax, psf)
    xp,yp = xp+dx,yp+dy
    psf = psf / f
    psf[0]=0
    AP = int(5 * b)
    return xp,yp,f,B,AP,b,psf

def shsolvew(nmax,J,W):
    global _cpy
    boxsz = np.shape(J)[0]
    B,f,xp,yp,snr = shphotocenter(J)
    bs, be = 0.1, boxsz/2
    for k in range(3):
        bset = np.linspace(bs, be, 10)
        err = np.zeros(np.size(bset))
        for l in range(np.size(bset)):
            psf = shapproxw(xp, yp, bset[l], nmax, boxsz, J,W)
            PSF = shbox(xp, yp, bset[l], nmax, psf, boxsz)
            err[l] = np.std(J - PSF)
        if(np.argmin(err)!=0):
            bs = bset[np.argmin(err) - 1]
        else:
            bs = bset[0]
        if (np.argmin(err) != 9):
            be = bset[np.argmin(err) + 1]
        else:
            be = bset[9]
    b = (bs + be) / 2.0
    psf = shapproxw(xp, yp, b, nmax, boxsz, J,W)
    dx, dy, f, B = shFXY(b, nmax, psf)
    xp,yp = xp+dx,yp+dy
    psf = psf / f
    psf[0]=0
    AP = int(5 * b)
    return xp,yp,f,B,AP,b,psf

def onepoint(x,y,B,f,b,nmax,psf,J):
    global _cpy
    box = np.shape(J)[0]
    P = np.array([B, f, x, y])
    def of(p):
        I = p[0]+p[1]*shbox(p[2],p[3],b,nmax,psf,box)
        return np.sum((J-I)**2)
    return minimize(of, P, method='Nelder-Mead', tol=1e-3)

def twopoints(B,x_p,y_p,f_p,x_s,y_s,f_s,b,nmax,psf,J):
    global _cpy
    box = np.shape(J)[0]
    P = np.array([B,x_p,y_p,f_p,x_s,y_s,f_s])
    def of(p):
        I = p[0]+p[3]*shbox(p[1],p[2],b,nmax,psf,box)+p[6]*shbox(p[4],p[5],b,nmax,psf,box)
        return np.sum((J-I)**2)
    return minimize(of, P, method='Nelder-Mead', tol=1e-3)

def one_point(x,y,B,f,b,nmax,psf,J):
    global _cpy
    box = np.shape(J)[0]
    P = np.array([B, f, x, y])
    X,Y = np.arange(0, box, 1),np.arange(0, box, 1)
    X,Y = np.meshgrid(X, Y)
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = J.ravel()
    def of(xdata, *p):
        Is = p[0] + p[1] * shbox(p[2], p[3], b, nmax, psf, box)
        return np.ravel(Is)
    # p,pcov = curve_fit(f=of, p0=P, xdata=xdata,ydata=ydata)
    return curve_fit(f=of, p0=P, xdata=xdata,ydata=ydata)

def sh_set_to_value(x,y,r,v,J):
    global _cpy
    box = np.shape(J)[0]
    _cpy.csh_set_to_value(ctypes.c_int(x),
                          ctypes.c_int(y),
                          ctypes.c_int(r),
                          ctypes.c_double(v),
                          ctypes.c_int(box),
                          J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return J
