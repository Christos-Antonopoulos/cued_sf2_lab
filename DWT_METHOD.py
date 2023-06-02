import numpy as np
from typing import List, Tuple, NamedTuple, Optional
import numpy as np
from cued_sf2_lab.laplacian_pyramid import rowdec,rowint, quantise, quant1, quant2, bpp
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.lbt import pot_ii, dct_ii
from cued_sf2_lab.dwt import dwt, idwt
from scipy import optimize
from cued_sf2_lab.jpeg import diagscan, dwtgroup, huffdflt, HuffmanTable, huffenc, huffgen, huffdes, runampl
from encoding import huffblockhist, huffencopt
from quantisation import quant, inv_quant

from time import perf_counter


def img_size(img):
    return bpp(img) * img.shape[0] * img.shape[1]

class DWTCompression:
    
    def __init__(self, n: int, log: bool = True):
        self.name = "DWT"
        self.n = n
        self.log = log
        self.img_size = (256, 256)
#This is the constructor of the class. It initializes the object with several member variables. self.n is used as the level of DWT transform. 
#self.log is a boolean variable for logging purposes. self.img_size represents the size of the image.


    def compress(self, X: np.ndarray):
        Y = X.copy()

        for i in range(self.n):
            m = 256//(2**i)
            Y[:m,:m] = dwt(Y[:m,:m])
            
        return Y
    
#This method performs the wavelet compression. It loops over n levels and applies the dwt function on the image array 

    def decompress(self, Y: np.ndarray):
        Yr = Y.copy()

        for i in range(self.n):
            m = 256//(2**(self.n - i - 1))
            Yr[:m,:m] = idwt(Yr[:m,:m])

        return Yr

    def estimate_entropy(self, Y):
        """Estimate the entropy of the image by considering coding each block"""
        dwtent = 0

        for i in range(self.n):

            m = 256//(2**i) # 256, 128, 64 ... 
            h = m//2 # Midpoint: 128, 64, 32 ...

            dwtent += img_size(Y[:h, h:m]) + img_size(Y[h:m, :h]) + img_size(Y[h:m, h:m])

        # Final low pass image
        dwtent += img_size(Y[:m, :m])

        return dwtent

    def constant_steps(self, step: float = 1.):
        dwtstep = np.ones((3, self.n)) * step
        dwtstep = np.concatenate((dwtstep, np.ones((3, 1))), axis=1)

        return dwtstep

    def equal_mse_steps(self, initial: float = 1.,ratio: float = 2., root2: bool = True):
        """Return a (3 x N+1) np array of step sizes with constant ratio"""

        if root2:
            const_ratio = np.logspace(start=self.n, stop=0, num=self.n, base=ratio) * initial
            dwtstep = np.stack((const_ratio, const_ratio, const_ratio * np.sqrt(2)))
            # append ones for DC componenta
            dwtstep = np.concatenate((dwtstep, np.ones((3, 1))), axis=1)
        else:
            dwtstep = np.array([np.ones((1, 3))[0]*initial*(0.5**i) for i in range(self.n + 1)]).T

        return dwtstep

    def quantise(self, Y, steps: np.ndarray, rise_ratio=None):
        """Quantise as integers"""
        Yq = np.zeros_like(Y)
        if rise_ratio is None: rise_ratio = 0.5
        for i in range(self.n):

            m = 256//(2**i) # 256, 128, 64 ... 
            h = m//2 # Midpoint: 128, 64, 32 ...

            # Quantising
            s_tr = max(steps[0, i], 1)
            s_bl = max(steps[1, i], 1)
            s_br = max(steps[2, i], 1)

            Yq[:h, h:m] = quant(Y[:h, h:m],s_tr, s_tr * rise_ratio) # Top right 
            Yq[h:m, :h] = quant(Y[h:m, :h], s_bl, s_bl * rise_ratio) # Bottom left
            Yq[h:m, h:m] = quant(Y[h:m, h:m], s_br, s_br * rise_ratio) # Bottom right

        # Final low pass image

        # Yq[128:, 128:] = 0

        m = 256//(2**self.n)
        s_tr = max(steps[0, self.n], 1)
        Yq[:m, :m] = quant(Y[:m, :m], s_tr, s_tr * rise_ratio)

        return Yq.astype(int)

    def inv_quantise(self, Y, steps: np.ndarray, rise_ratio=None):
        """Quantise as integers"""
        Yq = np.zeros_like(Y)
        if rise_ratio is None: rise_ratio = 0.5
        for i in range(self.n):
            m = 256//(2**i) # 256, 128, 64 ... 
            h = m//2 # Midpoint: 128, 64, 32 ...

            s_tr = max(steps[0, i], 1)
            s_bl = max(steps[1, i], 1)
            s_br = max(steps[2, i], 1)

            Yq[:h, h:m] = inv_quant(Y[:h, h:m],s_tr, s_tr * rise_ratio) # Top right 
            Yq[h:m, :h] = inv_quant(Y[h:m, :h], s_bl, s_bl * rise_ratio) # Bottom left
            Yq[h:m, h:m] = inv_quant(Y[h:m, h:m], s_br, s_br * rise_ratio) # Bottom right

        # Final low pass image
        m = 256//(2**self.n)
        s_tr = max(steps[0, self.n], 1)
        Yq[:m, :m] = inv_quant(Y[:m, :m], s_tr, s_tr * rise_ratio)
        return Yq.astype(int)

    def encode(self, Y: np.ndarray, qstep: Optional[int] = None, M: Optional[int] = None, dcbits: int = 16, rise_ratio=None, root2: bool=True) -> Tuple[np.ndarray, HuffmanTable]:
        """Pass in a transformed image, Y, and
         - regroup
         - quantise
         - generate huffman encoding"""

        if qstep is None:
            dwtsteps = self.constant_steps()
        else:
            dwtsteps=self.equal_mse_steps(qstep, root2=root2)

        Yq = self.quantise(Y, dwtsteps, rise_ratio=rise_ratio)

        Yq = dwtgroup(Yq, self.n)
        self.A = Yq


        N = np.round(2**self.n)
        if M is None:
            M = N

        # Generate zig-zag scan of AC coefs.
        scan = diagscan(M)

        huffhist = np.zeros(16 ** 2)
        t = perf_counter()
        # First pass to generate histogram
        for r in range(0, Yq.shape[0], M):
            for c in range(0, Yq.shape[1], M):
                yq = Yq[r:r+M,c:c+M]
                if M > N:
                    yq = regroup(yq, N)
                yqflat = yq.flatten('F')
                dccoef = yqflat[0] + 2 ** (dcbits-1)

                ra1 = runampl(yqflat[scan])
                huffhist += huffblockhist(ra1)

        # Use histogram
        vlc = []
        dhufftab = huffdes(huffhist)
        huffcode, ehuf = huffgen(dhufftab)

        for r in range(0, Yq.shape[0], M):
            for c in range(0, Yq.shape[1], M):
                yq = Yq[r:r+M,c:c+M]
                # Possibly regroup
                if M > N:
                    yq = regroup(yq, N)
                yqflat = yq.flatten('F')
                # Encode DC coefficient first
                dccoef = yqflat[0] + 2 ** (dcbits-1)
                if dccoef > 2**dcbits:
                    raise ValueError(
                        'DC coefficients too large for desired number of bits')
                vlc.append(np.array([[dccoef, dcbits]]))
                # Encode the other AC coefficients in scan order
                # huffenc() also updates huffhist.
                ra1 = runampl(yqflat[scan])
                vlc.append(huffencopt(ra1, ehuf))
        # (0, 2) array makes this work even if `vlc == []`
        vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

        return (vlc, dhufftab) # "variable length code" and header (huffman table "hufftab" or other)

    def decode(self, vlc: np.ndarray, qstep: Optional[int] = None, hufftab: Optional[HuffmanTable] = None, N: int = 8, M: int = 8, dcbits: int = 16, rise_ratio=None, root2: bool=True) -> np.ndarray:

        N = np.round(2**self.n)

        if M is None:
            M = N

        if qstep is None:
            dwtsteps = self.constant_steps()
        else:
            dwtsteps = self.equal_mse_steps(qstep, root2=root2)

        scan = diagscan(M)

        # Define starting addresses of each new code length in huffcode.
        huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
        huffcode, ehuf = huffgen(hufftab) # Set up huffman coding arrays.

        k = 2 ** np.arange(17) # Define array of powers of 2 from 1 to 2^16.

        # For each block in the image:

        # Decode the dc coef (a fixed-length word)
        # Look for any 15/0 code words.
        # Choose alternate code words to be decoded (excluding 15/0 ones).
        # and mark these with vector t until the next 0/0 EOB code is found.
        # Decode all the t huffman codes, and the t+1 amplitude codes.

        eob = ehuf[0]
        run16 = ehuf[15 * 16]
        i = 0
        Zq = np.zeros(self.img_size)

        W, H = self.img_size
        for r in range(0, H, M):
            for c in range(0, W, M):
                yq = np.zeros(M**2)

                # Decode DC coef - assume no of bits is correctly given in vlc table.
                cf = 0 
                if vlc[i, 1] != dcbits:
                    raise ValueError('The bits for the DC coefficient does not agree with vlc table')

                yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
                i += 1

                while vlc[i, 0] != eob[0]: # Loop for each non-zero AC coef.
                    run = 0

                    while vlc[i, 0] == run16[0]: # Decode any runs of 16 zeros first.
                        run += 16
                        i += 1

                    # Decode run and size (in bits) of AC coef.
                    start = huffstart[vlc[i, 1] - 1]
                    res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                    run += res // 16
                    cf += run + 1
                    si = res % 16
                    i += 1
                    
                    # Assume no problem with Huffman table/decoding
                    ampl = vlc[i, 0]

                    thr = k[si - 1] # Adjust ampl for negative coef (i.e. MSB = 0).
                    yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)
                    i += 1

                i += 1 # End-of-block detected, save block.

                yq = yq.reshape((M, M)).T

                if M > N: # Possibly regroup yq
                    yq = regroup(yq, M//N)

                Zq[r:r+M, c:c+M] = yq

        Zq = dwtgroup(Zq, -self.n)
        Z = self.inv_quantise(Zq, dwtsteps, rise_ratio=rise_ratio)
        return Z

    def opt_encode(self, Y: np.ndarray, size_lim=40960, M: int = 8, root2: bool=True, rise_ratio=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            
        def error(qstep: int) -> int:

            Z, h = self.encode(Y, qstep=qstep, M=M, rise_ratio=rise_ratio, root2=root2)
            size = Z[:, 1].sum()

            return np.sum((size - size_lim)**2)

        opt_step = optimize.minimize_scalar(error, method="bounded", bounds=(0.1, 64)).x
        vlc, hufftab = self.encode(Y, qstep=opt_step, M=M, rise_ratio=rise_ratio, root2=root2)

        return (vlc, hufftab), opt_step


