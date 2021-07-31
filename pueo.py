import numpy as np
from scipy import signal
from scipy import ndimage

class PUEO:
    def __init__(self, impResp='impResp2.csv'):
        ir = np.genfromtxt(impResp)
        ir[0] = 0.119775
        # Resample 10 GSa/s to 3 GSa/s
        ir3 = signal.resample_poly(ir, 3, 10)
        # Create our filters
        self.hp = signal.butter(5, 0.1666, btype='highpass', output='sos')
        self.lp = signal.butter(5, 0.8, btype='lowpass', output='sos')
        # and filter the impulse response, and normalize
        ir3 = signal.sosfilt(self.hp, ir3)
        ir3 = signal.sosfilt(self.lp, ir3)        
        p2p = np.amax(ir3)-np.amin(ir3)
        self.ir = ir3/(p2p/2)
        # this is from TFilter, 0-650 Hz 1 dB ripple 750-1500 Hz 30 dB rejection
        self.trig_lp = [ 0.008221498626438355
                        ,0.022467600608263687
                        ,-0.005735806817546407
                        ,-0.011305635285472695
                        ,0.000048253848191315944
                        ,0.016226723878188614
                        ,0.0030049330597322955
                        ,-0.01985335970710756
                        ,-0.008747581736113496
                        ,0.023662964861940212
                        ,0.017313676075751187
                        ,-0.027161918283047173
                        ,-0.03041376298199074
                        ,0.030156958746321648
                        ,0.052323744831867386
                        ,-0.0324770203662481
                        ,-0.09907231876330876
                        ,0.03391802170142495
                        ,0.3159219034492574
                        ,0.46558749845556885
                        ,0.3159219034492574
                        ,0.03391802170142495
                        ,-0.09907231876330876
                        ,-0.0324770203662481
                        ,0.052323744831867386
                        ,0.030156958746321648
                        ,-0.03041376298199074
                        ,-0.027161918283047173
                        ,0.017313676075751187
                        ,0.023662964861940212
                        ,-0.008747581736113496
                        ,-0.01985335970710756
                        ,0.0030049330597322955
                        ,0.016226723878188614
                        ,0.000048253848191315944
                        ,-0.011305635285472695
                        ,-0.005735806817546407
                        ,0.022467600608263687
                        ,0.008221498626438355
                        ]
        # "FIR" for average
        self.avgfir = [ (1/8) ]*8
        self.avgfir16 = [ (1/16) ]*16
        # Our magic IIR. This is an IIR with a double-zero at -1,
        # and poles at P=(1/sqrt(2)) and theta=+/-pi/6.
        # It's "magic" because when converted by adding 6 cancelling zeros
        # the denominator becomes [1, -1/4, 1/16], and
        # most of the coefficients become dumb.
        # g6 = 1/8
        # g5 = 1/2 sqrt(2)cos(pi/6)
        # g4 = 3/4
        # g3 =     sqrt(2)cos(pi/6)
        # g2 = 3/2
        # g1 = 1        
        # Meaning in the *entire* IIR there are only 2 real multiplies after the FIR. Everything else
        # has extremely simple partial product decomposition (either 1 or 2 adds). And the single
        # multiply for the FIR (for k) can be buried in the squaring conversion if I do it right.
        #
        # Note: this math is *exact*. There are no coefficient quantization issues here outside
        # of g5/g3 and the overall gain (which is unimportant anyway).
        k = 0.06483475022792895
        # This is stored as a second-order section.
        self.optiir = [ [ k, 2*k, k, 1, -1*np.cos(np.pi/6)*np.sqrt(2), 0.5 ] ]
        self.optiir_zi = signal.sosfilt_zi(self.optiir)
        # This is now a first-order IIR, decimated
        cb, ca = signal.butter(1, 0.45)
        self.cb = [ cb[0], 0, 0, 0, cb[1], 0, 0, 0 ]
        self.ca = [ ca[0], 0, 0, 0, ca[1], 0, 0, 0 ]
                
        # 1st stage halfband FIR for lowpass version
        self.lp_hb1 = signal.firwin(9, 0.5)
        # 2nd stage halfband FIR for lowpass version
        self.lp_hb2 = signal.firwin(13, 0.5)
        # Alternate 2nd stage filter
        self.lp_hb2alt = signal.cheby1(2, 0.5, 0.2, output='sos')
        self.lp_hb2altzi = signal.sosfilt_zi(self.lp_hb2alt)
        # "zeroth" stage halfband FIR for non-filtered
        self.lp_hb0 = signal.firwin(5, 0.5)
        self.notch = None
        
    def setNotch(self, on=False, freq=380., q=5):
        if on:
            b, a = signal.iirnotch(freq, q, fs=3000)        
            self.notch = signal.tf2sos(b, a)
            self.notchzi = signal.sosfilt_zi(self.notch)
        else:
            self.notch = None
            self.notchzi = None
        
    def get(self, snr, cwsnr=0, cwfreq=380., length=1024, front=128):
        # give ourselves some room
        fullPad = length+256
        sig = np.pad(self.ir, [front, fullPad - len(self.ir) - front], mode='constant')
        # Figure out random phase
        phase = np.random.random_sample()
        # Shift.
        this_sig = ndimage.shift(sig, phase)
        # Generate noise
        noise = np.random.normal(0, 1, sig.shape)
        noise = signal.sosfilt(self.hp, noise)
        noise = signal.sosfilt(self.lp, noise)
        # Scale
        rms = np.std(noise)
        noise /= rms
        # Generate CW
        # we do sin(2pi*i*(freq/3000)). So at 375
        phase = np.random.random_sample()*np.pi*2
        x = np.arange(0, fullPad)
        sin = np.sin(2*np.pi*(cwfreq/3000.)*x + phase)
        this_sig = this_sig*snr + noise + cwsnr*sin
        return this_sig
    
    def trigger_filter(self, sig, phase=None):
        if phase == None:
            phase=np.random.randint(0,2)
            
        afterlp = signal.lfilter(self.trig_lp, [1], sig)
        # correct for impulse delay of filter
        afterlp = afterlp[int((len(self.trig_lp)-1)/2):]
        # and decimate by 2 with random phase
        afterlp = afterlp[phase::2]
        return afterlp

    def average16(self, power, phase=None):
        if phase == None:
            phase = np.random.randint(0,8)
        power = signal.lfilter(self.avgfir16, [1], power)[phase::8]
        return power
    
    def average8(self, power, phase=None):
        if phase == None:
            phase=np.random.randint(0,4)
        power = signal.lfilter(self.avgfir, [1], power)[phase::4]
        return power

    def decimate8(self, power, phase=None, alt=False):
        if phase == None:
            phase = np.random.randint(0,8)
        # Zeroth filter
        lp0 = signal.lfilter(self.lp_hb0, [1], power)[(phase % 2)::2]
        phase = int(phase/2)
        return self.decimate4(lp0, phase, alt)

    def decimate4(self, power, phase=None, alt=False):
        if phase == None:
            phase=np.random.randint(0,4)
        # Filter first halfband. If phase=1 or 3 we start from second here
        lp1 = signal.lfilter(self.lp_hb1, [1], power)[(phase % 2)::2]
        if alt == True:
            lp2, lp2zf = signal.sosfilt(self.lp_hb2alt, lp1, zi=self.lp_hb2altzi)
            lp2 = lp2[int(phase/2)::2]
        else:
            # Filter second halfband. If phase = 3 or 4 we start from second here
            lp2 = signal.lfilter(self.lp_hb2, [1], lp1)[int(phase/2)::2]
        return lp2
    
    def optimize4(self, power, phase=None):
        if phase == None:
            phase=np.random.randint(0,4)
        # Optimized IIR first
        lp1, lp1zf = signal.sosfilt(self.optiir, power, zi=self.optiir_zi)
        # Now first-order IIR and grab a random phase
        lp2 = signal.lfilter(self.cb, self.ca, lp1)[phase::4]
        
        return lp2
    
