##### use by:
## import sys
## sys.path.append('/home/speik/soridat/pythonlib')
## import mwave as mw
##
'''
  Module with useful Microwave methods and definitions
'''
from numpy import  array,sqrt,pi,log,matrix, conj, angle, exp, abs, log10, arange, tan, isnan, nan, cosh, sinh
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/speik/soridat/pythonlib/mwlab') 
import smith as smi
import types
from scipy.optimize import fsolve, brentq

#### some constants

mu0  = 4*pi*1e-7    # Permeability of free space
c    = 299792458.0  # velocity of light
eps0 = 1/mu0/c/c    # Permittivity of free space
k    = 1.3806488e-23 # Boltzman Constant
eta0 = sqrt(mu0/eps0) # Free Space impedance
 

def lineinputimpedance(Z0,Zl,betal):
    r'''
    Calculates input impedance of a terminated line

    Args:
       :Z0: Line impedance in Ohm (type complex)
       :Zl: Load impedance in Ohm (type complex)
       :betal: Electrical length :math:`\beta l` in radians (type float)
    Returns: 
       complex: Input impedance in Ohm 
    '''
    Zin = Z0 * (Zl+1j*Z0*tan(betal)) / (Z0+1j*Zl*tan(betal))
    return Zin

###########################################################################
def msimpedance(w,h,er):
  '''
  Calculates microstrip line impedance :math:`Z_0` and :math:`\epsilon_{eff}` from Wheeler formula 

    :w: width of MS-line (type float)
    :h: height of substrate (type float)
    :er: Epsilon relative of substrate (type float)
    returns:  Z0, eps_eff  (tuple)
  '''

  eta0=377
  if(w/h<=1):
      F=1/sqrt(1+12*h/w)+0.004*(1-w/h)**2
  else:
      F=1/sqrt(1+12*h/w)
  e_eff=0.5*(er+1+(er-1)*F)

  if(w/h<=1): 
      Z0=eta0/sqrt(e_eff)*1/2/pi*log(8*h/w+0.25*w/h)
  else:
      Z0=eta0/sqrt(e_eff)*1/(w/h+2.46-0.49*h/w+(1-h/w)**6)
  return Z0,e_eff

###########################################################################
def msdimension(Z0wanted,elen,f,h,epsr):
    '''
    Calculates microstrip line dimensions from the impedance, elec. length
    
        :Z0wanted: Impedance of MS-line (type float)
        :elen: elec length of line in :math:`\lambda` (type float)
        :f: frequency (type float)
        :h: height of substrate in mm (type float)
        :epsr: Epsilon relative of substrate (type float)
        returns:  w,l, epseff (float) in mm
    '''

    lam0=3e8/f
    imp= lambda w:  msimpedance(w,h,epsr)[0]-Z0wanted
    try:
        result = brentq(imp,0.02*h,5.0*h) 
    except:
        raise ValueError('could not find solution in msdimension for %f Ohms' % (Z0wanted))
    w=round(result,2)
    Z0,epseff=msimpedance(w,h,epsr)
    lamms=lam0/sqrt(epseff)*1000
    l = elen * lamms
    return (w,l,epseff)

### ABCD Matrix for Series Element #########################
def ABCDseries(Z):
    return matrix([[1,Z],[0,1]])

### ABCD Matrix for Shunt Element #########################
def ABCDshunt(Y):
    return matrix([[1,0],[Y,1]])

### ABCD Matrix for Line Element #########################
def ABCDline(beta, length,Z0,alpha=0.0):
    gammal = (alpha+1j*beta)*length
    return matrix([[cosh(gammal), Z0*sinh(gammal)],[ 1./Z0*sinh(gammal), cosh(gammal)]])


############################################################
def cascade(ABCDlist):
    p = matrix([[1,0],[0,1]])
    for ABCD in ABCDlist:
        p = p * ABCD
    return p

############################################################
def ABCDJInverter(J):
    return(matrix([[0,-1/(1j*J)],[1j*J,0]]))

def ABCDKInverter(K):
    return(matrix([[0,1j*K],[-1/1j/K,0]]))

############################################################
def ABCDtoS(ABCD,Z0):
    A=ABCD[0,0]
    B=ABCD[0,1]
    C=ABCD[1,0]
    D=ABCD[1,1]
    S11=(A+B/Z0-C*Z0-D)/(A+B/Z0+C*Z0+D)
    S12=2*(A*D-B*C)/(A+B/Z0+C*Z0+D)
    S21=2/(A+B/Z0+C*Z0+D)
    S22=(-A+B/Z0-C*Z0+D)/(A+B/Z0+C*Z0+D)
    S=matrix([[S11,S12],[S21,S22]])
    return S

############################################################
def StoABCD(S,Z0):
    S11=S[0,0]
    S12=S[0,1]
    S21=S[1,0]
    S22=S[1,1]
    A =      ((1+S11)*(1-S22)+S12*S21) / 2/S21
    B = Z0*  ((1+S11)*(1+S22)-S12*S21) / 2/S21
    C = 1/Z0*((1-S11)*(1-S22)-S12*S21) / 2/S21
    D =      ((1-S11)*(1+S22)+S12*S21) / 2/S21
    ABCD=matrix([[A,B],[C,D]])
    return ABCD

#############################################################
def ZtoS(Z,Z0):
    Z11=Z[0,0]
    Z12=Z[0,1]
    Z21=Z[1,0]
    Z22=Z[1,1]
    S11 = ( (Z11-Z0)*(Z22+Z0)-Z12*Z21 ) / ((Z11+Z0)*(Z22+Z0)-Z12*Z21)
    S12 = ( 2*Z12*Z0 ) / ((Z11+Z0)*(Z22+Z0)-Z12*Z21)
    S21 = ( 2*Z21*Z0 ) / ((Z11+Z0)*(Z22+Z0)-Z12*Z21)
    S22 = ( (Z11+Z0)*(Z22-Z0)-Z12*Z21 ) / ((Z11+Z0)*(Z22+Z0)-Z12*Z21)
    S=matrix([[S11,S12],[S21,S22]])
    return S


################################################################################
################################################################################
##    AMP DESIGN 
################################################################################
################################################################################


### Return a complex type from a number given in magitude and phase (Degrees)
def magphase(A,phi):
    '''Returns a complex number from  magnitude and phase (in degrees)
    '''
    return A*exp(1j*phi*pi/180.0)

### Return a string formatted from a complex in the form Magn /__ Phase deg

################################################################################
def magphase_str(c):
    ''' Returns a nicely formatted string to print complex numbers in ampl. and phase
    '''
    return u'{0:6.3f}\u2220{1:5.1f}\u00B0'.format(abs(c),angle(c)*180/pi)

################################################################################
def magphase_tuple(c):
    ''' Returns a tuple with (magn,phase) to print complex numbers in ampl. and phase
    '''
    return ( abs(c) , angle(c)*180/pi )

################################################################################
def splitmatrixarray(S):
    '''
....splits list of matrices into  lists of the individual elements
    currently two by two matirces only
    '''
    S11 = array([x[0,0] for x in S])
    S12 = array([x[0,1] for x in S])
    S21 = array([x[1,0] for x in S])
    S22 = array([x[1,1] for x in S])
    return S11,S12,S21,S22


### Load touchstone formatted  S-parameter files ###############################
def load_touchstone(filename):
    '''
    Loads a touchstone file in two lists 
    
        :filename: Touchstone filename including path (type string)
        returns: tuple with: frequency list (type flaot)  S-matrix list (2x2 Matrix list of S Parameters)
       
    currently works with 2y2 matrices only
    '''
    print("Load Touchstone file ",filename)
    f=open(filename,'r', encoding = "ISO-8859-1")
    noise=False
    if filename[-2] == '1': 
        Twoport = False
    elif filename[-2] == '2': 
        Twoport = True
    else:
        print('Load Touchstone: Neither extension s1p or s2p , Exit')
        exit(1)

   
    Slist=[];flist=[]
    rad=pi/180.0
    for line in f:
        #print(line.strip())
        if line[0]=='!': 
            if line.find('Fmin')>0:
                noise=True
                #print("----- Here Noise Data start ------>")
            continue
        if line[0]=='#':
            #print("Format is ",line)
            if 'HZ' in line.upper(): factor=1e0
            if 'KHZ' in line.upper(): factor=1e3
            if 'MHZ' in line.upper(): factor=1e6
            if 'GHZ' in line.upper(): factor=1e9
            if 'MA' in line.upper():
                sform ='MA'
            elif 'RI' in line.upper(): 
                sform = 'RI'
            elif 'DB' in line.upper(): 
                sform ='DB'
            else:
                print("Data not in MA or RI Format")
                raise RuntimeError("Data not in MA or RI Format")
                return
            continue
        if len(line) <10: continue ## empty line
        if not(noise): ##### Spara Info
            p=line.split()
            p=[float(x) for x in p]
            #print("f=",p[0],"S11=",p[1], ".....")
            flist.append(float(p[0])*factor)
            if sform=='MA':
                S11=p[1]*exp(1j*p[2]*rad)
                S=S11
                if Twoport:
                    S21=p[3]*exp(1j*p[4]*rad)
                    S12=p[5]*exp(1j*p[6]*rad)
                    S22=p[7]*exp(1j*p[8]*rad)
                    S=matrix([[S11,S12],[S21,S22]])
                Slist.append(S)
            if sform=='RI':
                S11=p[1]+p[2]*1j
                S=S11
                if Twoport:
                    S21=p[3]+p[4]*1j
                    S12=p[5]+p[6]*1j
                    S22=p[7]+p[8]*1j
                    S=matrix([[S11,S12],[S21,S22]])
                Slist.append(S)
            if sform=='DB':
                S11=10**(p[1]/20)*exp(1j*p[2]*rad)
                S=S11
                if Twoport:
                    S21=10**(p[3]/20)*exp(1j*p[4]*rad)
                    S12=10**(p[5]/20)*exp(1j*p[6]*rad)
                    S22=10**(p[7]/20)*exp(1j*p[8]*rad)
                    S=matrix([[S11,S12],[S21,S22]])
                Slist.append(S)
            #print S
        if (noise): ##### Noise Info
            pass
    flist = array(flist)
    return flist,Slist



#
# mdif load
#
import re
rad = pi/180.0

######################################################################
def mdifbiaslist(filename):
    f=open(filename,'r')
    line = f.readlines()
    i=0
    biaslist = []
    while i< len(line):
        if 'VAR Vc' in line[i]:
            if not 'Ic' in line[i+1]: 
                raise valueerror('No Vc,Ic VAR defined in mdif')
            valueV = re.findall("\d+\.\d+", line[i])[0]
            valueI = re.findall("\d+\.\d+", line[i+1])[0]
            biaslist.append((float(valueV),float(valueI)))
            i += 1   
        i += 1
    if biaslist == []: raise valueerror('No Vc,Ic VAR defined in mdif')
    return biaslist
  
##########Load MDIF Spara #############################################################
def mdifsparlist(filename,Vc,Ic):
    f=open(filename,'r')
    line = f.readlines()
    i=0
    biaslist = []
    while i< len(line):
        if 'VAR Vc' in line[i]:
            valueV = float(re.findall("\d+\.\d+", line[i])[0])
            valueI = float(re.findall("\d+\.\d+", line[i+1])[0])
            if valueV == Vc and valueI == Ic:
                #print("Biaspoint found", valueV, valueI)
                if not ('BEGIN ACDATA' in line[i+2]): raise ValueError('MDIF Wrong Format no BEGIN ACDATA found ')
                i +=3
                #print(line[i])
                if not '#' in line[i]: raise ValueError('MDIF Wrong Format no # Format found found ')
                if 'HZ'  in line[i]: factor=1e0
                if 'MHZ' in line[i]: factor=1e6
                if 'GHZ' in line[i]: factor=1e9
                if 'MA' in line[i]:
                    sform ='MA'
                elif 'RI' in line[i]: 
                    sform = 'RI'
                else:
                    raise RuntimeError("MDIF Data not in MA or RI Format")
                #print(sform, factor)
                i += 2
                
                ##### Start of spar found reading data ###################
                flist = []
                Slist = []
                while not 'END' in line[i]: 
                    p=line[i].split()
                    p=[float(x) for x in p]
                    #print("f=",p[0],"S11=",p[1], ".....")
                    flist.append(float(p[0])*factor)
                    if sform=='MA':
                        S11=p[1]*exp(1j*p[2]*rad)
                        S21=p[3]*exp(1j*p[4]*rad)
                        S12=p[5]*exp(1j*p[6]*rad)
                        S22=p[7]*exp(1j*p[8]*rad)
                    if sform=='RI':
                        S11=p[1]+p[2]*1j
                        S21=p[3]+p[4]*1j
                        S12=p[5]+p[6]*1j
                        S22=p[7]+p[8]*1j
                    S=matrix([[S11,S12],[S21,S22]])
                    Slist.append(S)
                    i += 1
                return flist, Slist
                ### end of spar data read 
                
        i += 1
    raise ValueError('Specific Vc,Ic not defined in mdif')
    return 

###### Load MDIF Noise #############################################
def mdifnoiselist(filename,Vc,Ic):
    f=open(filename,'r')
    line = f.readlines()
    i=0
    biaslist = []
    while i< len(line):
        if 'VAR Vc' in line[i]:
            valueV = float(re.findall("\d+\.\d+", line[i])[0])
            valueI = float(re.findall("\d+\.\d+", line[i+1])[0])
            if valueV == Vc and valueI == Ic:
                #print("Biaspoint found", valueV, valueI)
                i+=  2
                while i< len(line):
                    if ('BEGIN NDATA' in line[i]): break
                    i += 1
                if i == len(line): raise ValueError('MDIF no BEGIN NDATA found ')
                i += 1
                if not '#' in line[i]: raise ValueError('MDIF Wrong Format no # Format found found ')
                if 'HZ'  in line[i]: factor=1e0
                if 'MHZ' in line[i]: factor=1e6
                if 'GHZ' in line[i]: factor=1e9
                if 'MA' in line[i]:
                    sform ='MA'
                elif 'RI' in line[i]: 
                    sform = 'RI'
                else:
                    raise RuntimeError("MDIF Data not in MA or RI Format")
            
                i += 2
                ##### Start of spar found reading data ###################
                flist = []
                Nfminlist = []
                Gamoptlist = []
                Rnlist = []
                while not ('END' in line[i]): 
                    p=line[i].split()
                    p=[float(x) for x in p]
                    #print("f=",p[0],"S11=",p[1], ".....")
                    flist.append(p[0]*factor)
                    Nfminlist.append(p[1])    ### min Noisefigure
                    if sform=='MA':
                        Gamoptlist.append(p[2]*exp(1j*p[3]*rad)) ## Gamma Opt. 
                    if sform=='RI':
                        Gamoptlist.append(p[2]+p[3]*1j)
                    Rnlist.append(p[4])
                    i += 1
                if flist == []:
                    print('MDIF: No Noise data defined')
                return flist, Nfminlist, Gamoptlist, Rnlist
                ### end of spar data read 
                
        i += 1
    raise ValueError('Specific Vc,Ic not defined in mdif')
    return 

### Splits a matrix S-function into its elements for printing etc.
def SSplit(Sfct):
    ''' Splits a matrix S-function into its elements for printing etc.
    '''
    S11= lambda f: Sfct(f)[0,0]
    S12= lambda f: Sfct(f)[0,1]
    S21= lambda f: Sfct(f)[1,0]
    S22= lambda f: Sfct(f)[1,1]      
    return S11,S12,S21,S22

### Plot S-Parameter in Cart Plot
def PlotSpar(flist,Slist):
    fig,ax = plt.subplots(figsize=(7,5))
    flist=array(flist)
    S11list=array([Slist[i][0,0] for i in range(len(Slist))])
    S12list=array([Slist[i][0,1] for i in range(len(Slist))])
    S21list=array([Slist[i][1,0] for i in range(len(Slist))])
    S22list=array([Slist[i][1,1] for i in range(len(Slist))])
    ax.plot(flist/1e9,20*log10(abs(S11list)),label='$S_{11}$')
    ax.plot(flist/1e9,20*log10(abs(S21list)),label='$S_{21}$')
    ax.plot(flist/1e9,20*log10(abs(S12list)),label='$S_{12}$')
    ax.plot(flist/1e9,20*log10(abs(S22list)),label='$S_{22}$')
    plt.legend(loc=4)
    plt.xlabel('Freq in GHz')
    plt.ylabel('S in dB')
    plt.grid()
    plt.tight_layout()
    return fig,ax
    

###### determine Stability Circles and mu factor
def AmpStabilityCircle(S,plotit=False): 
    S11=S[0,0];S12=S[0,1];S21=S[1,0];S22=S[1,1]
    #print(u"\n \u25AD\u25AD\u25AD\u25AD Stability \u25AD\u25AD\u25AD\u25AD")
    Delta=S11*S22-S12*S21
    #print("|Delta|=",abs(Delta))
    Cl=conj(S22-Delta*conj(S11))/(abs(S22)**2-abs(Delta)**2)
    Rl=abs(S12*S21/(abs(S22)**2-abs(Delta)**2))
    Cs=conj(S11-Delta*conj(S22))/(abs(S11)**2-abs(Delta)**2)
    Rs=abs(S12*S21/(abs(S11)**2-abs(Delta)**2))
    mu1=(1-abs(S11)**2)/(abs(S22-Delta*conj(S11))+abs(S12*S21));
    mu2=(1-abs(S22)**2)/(abs(S11-Delta*conj(S22))+abs(S12*S21));
    k=(1-abs(S11)**2-abs(S22)**2+Delta**2)/(2*abs(S12*S21));

    fig, ax = (0,0)
    if plotit:
        fig, ax = plt.subplots() 
        plt.tight_layout()
        ax.set_title('Stability Circles', fontsize=15) 
        fig.set_facecolor('white')
        Z0=1 
        mysmith=smi.smith(ax,'smith',Z0,0.5)
        mysmith.addcircle(Cl,Rl)
        mysmith.addcircle(Cs,Rs,'r')
        #plt.savefig('stabcircles.pdf')
        return Cs,Rs,Cl,Rl,mu1, mu2, fig, ax
    return Cs,Rs,Cl,Rl,mu1, mu2, 0, 0


##### determine Noise Circle and Noise Number N
def AmpNoiseCircle(S,FmindB,Gamopt,rn,FsetdB,plotit): 
    S11=S[0,0];S12=S[0,1];S21=S[1,0];S22=S[1,1]
    Delta=S11*S22-S12*S21
    F=10**(FsetdB/10)
    Fmin=10**(FmindB/10)
    N=(F-Fmin)/4/rn*abs(1+Gamopt)**2
    Cf=Gamopt/(N+1)
    Rf=sqrt(N*(N+1-abs(Gamopt)**2))/(N+1)
    fig, ax = (0,0)
    if plotit:
        fig, ax = plt.subplots() 
        plt.tight_layout()
        ax.set_title('Noise Circle', fontsize=15) 
        fig.set_facecolor('white')
        Z0=1 
        mysmith=smi.smith(ax,'smith',Z0,0.5)
        mysmith.addcircle(Cf,Rf)
    return Cf, Rf, N, fig, ax

##### determine Gain Circle 
def AmpGainCircleSource(S,GaindB): 
    S11=S[0,0];S12=S[0,1];S21=S[1,0];S22=S[1,1]
    if S12 != 0:
        raise ValueError('Device is not Unilateral!')
        return
    Delta=S11*S22-S12*S21
    Gain=10**(GaindB/10);
    Gsmax=1/(1-abs(S11)**2);
    GsmaxdB=10*log10(Gsmax);
    gs=10**(GaindB/10)/Gsmax;
    if 1-gs <0:
        return nan,nan,nan
    Cs=gs*conj(S11)/(1-(1-gs)*abs(S11)**2);
    Rs=sqrt(1-gs)*(1-abs(S11**2))/(1-(1-gs)*abs(S11)**2);
    return Cs, Rs, GsmaxdB

##### determine Gain Circle 
def AmpGainCircleLoad(S,GaindB): 
    S11=S[0,0];S12=S[0,1];S21=S[1,0];S22=S[1,1]
    if S12 != 0:
        raise ValueError('Device is not Unilateral!')
        return
    Delta=S11*S22-S12*S21
    Gain=10**(GaindB/10);
    Gsmax=1/(1-abs(S22)**2);
    GsmaxdB=10*log10(Gsmax);
    gs=10**(GaindB/10)/Gsmax;
    if 1-gs <0:
        return nan,nan, nan
    Cs=gs*conj(S22)/(1-(1-gs)*abs(S22)**2);
    Rs=sqrt(1-gs)*(1-abs(S22**2))/(1-(1-gs)*abs(S22)**2);
    return Cs, Rs, GsmaxdB


####### Bilateral Max. Gain Design
def AmpMaxgain(S):
    '''
    Calculates Maximum Gain input and output loads
    
    Args:
       :S: S-Matrix containing linear transistor S data (2x2 Matrix)
    
    Returns:
       Tuple: :math:`\Gamma_s` (type complex), :math:`\Gamma_l` (type complex), GtdB (type float)
    '''
 
    S11=S[0,0];S12=S[0,1];S21=S[1,0];S22=S[1,1]
    Delta=S11*S22-S12*S21
    #print(u"\n\u25AD\u25AD\u25AD\u25AD Conj. Matching \u25AD\u25AD\u25AD\u25AD\n")
    B1=1+abs(S11)**2-abs(S22)**2-abs(Delta)**2
    B2=1+abs(S22)**2-abs(S11)**2-abs(Delta)**2
    C1=S11-Delta*conj(S22)
    C2=S22-Delta*conj(S11)
    stablegain=0
    #print(B1,B2,C1,C2,Delta)
    #print("Solution 1:")
    Gams1=(B1-sqrt(B1**2-4*abs(C1)**2+0j))/2/C1
    Gaml1=(B2-sqrt(B2**2-4*abs(C2)**2+0j))/2/C2
    #print(u"   \u0393s1=",magphase_str(Gams1))
    #print(u"   \u0393l1=",magphase_str(Gaml1))
    #print("Solution 2:")
    Gams2=(B1+sqrt(B1**2-4*abs(C1)**2+0j))/2/C1
    Gaml2=(B2+sqrt(B2**2-4*abs(C2)**2+0j))/2/C2
    #print(u"   \u0393s2=",magphase_str(Gams2))
    #print(u"   \u0393l2=",magphase_str(Gaml2))
    if (abs(Gams1)<0.99 and abs(Gaml1)<0.99):
        #print(">> Choosen Sol.1:")
        #print( u"   \u0393s=",magphase_str(Gams1))
        #print( u"   \u0393l=",magphase_str(Gaml1))
        Gaml=Gaml1
        Gams=Gams1
        stablegain=1;
    if (abs(Gams2)<0.99 and abs(Gaml2)<0.99):
        #print(">> Choosen Sol.2:")
        #print(u"   \u0393s=",magphase_str(Gams2))
        #print(u"   \u0393l=",magphase_str(Gaml2))
        Gaml=Gaml2
        Gams=Gams2
        stablegain=1
    if not(stablegain):
        # disp(" ***** Geht Nicht Unstable *****")
        return 0,0,0
    else:
        #disp("\n=== Transducer Power Gain: ===")
        Gamin=S11+(S12*S21*Gaml)/(1-S22*Gaml)
        Gamout=S22+(S12*S21*Gams)/(1-S11*Gams)
        Gs=(1-abs(Gams)**2)/abs(1-Gamin*Gams)**2
        G0=abs(S21)**2
        Gl=(1-abs(Gaml)**2)/abs(1-S22*Gaml)**2
        Gt=Gs*G0*Gl
        GsdB=10*log10(Gs)
        G0dB=10*log10(G0)
        GldB=10*log10(Gl)
        GtdB=GsdB+G0dB+GldB
        #print(u"Gt={0:.2f}*{1:.2f}*{2:.2f}={3:.2f}".format(Gs,G0,Gl,Gt))
        #print(u"GtdB={0:.2f}dB+{1:.2f}dB+{2:.2f}dB={3:.2f}dB".format(GsdB,G0dB,GldB,GtdB) )
        return Gams,Gaml,GtdB
    

#### Matching Circuits ##############
def AmpStubmatching(Gammamatch,plotit=False):
    '''
    Performs a complete open stub - line matching for a given desired input :math:`\Gamma`
    
    Plots the smith chart with the constructed matching network 
    
    Args:
       :Gammamatch: desired input :math:`\Gamma` (type complex)
       :plotit: Flag for plotting (type Boolean)
    
    Returns:
       Tuple: line length of stub (float), line length of line (float), fig, ax (handles of smith plot)
    '''

    Z0=1
    Z1=Z0
    ### finding length of open stub line
    for len1 in arange(0.001,1,0.001):
        betal=2*pi*len1
        Zstub=-1j*Z0/tan(betal)
        Z2=Z1*Zstub/(Z1+Zstub)
        Gam=(Z2-Z0)/(Z2+Z0)
        if abs(Gam)>abs(Gammamatch): break
    ### finding length of inserted line
    if(angle(Gam)>angle(Gammamatch)):
        betal=(angle(Gam)-angle(Gammamatch))/2
    else:
        betal=(angle(Gam)-angle(Gammamatch))/2+pi
    len2=round(betal/2/pi,3)
    #print(u"stub length={0:.3f} \u03BB inserted line length={1:.3f}\u03BB".format(len1,len2))
    if plotit:
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
        plt.tight_layout()
        fig.set_facecolor('white')
        plt.clf()
        ax = plt.axes() 
        Z0=1 
        mysmith=smi.smith(ax,'both',Z0,0.5) 
        #mysmith.addanglering()
        Zl=mysmith.addstart(1) 
        Z2=mysmith.addstubopen(1,len1,1)    
        Z3=mysmith.addline(Z2,len2,1)
        mysmith.addpoint(1,'$Z_{50}$','NE')
        mysmith.addpoint(Z2,'$Z_2$','NE') 
        mysmith.addpoint(Z3,'$Z_3$','NE')
        return len1,len2, fig, ax
    else:
        return len1,len2,0,0



