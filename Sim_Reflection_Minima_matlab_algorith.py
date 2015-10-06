#########################################################################################################
# Philipp Liehm  01/01/2013                                                                             #
#                                                                                                       #
# Script to simulate the reflectivity of a layer structure with a transfer matrix algorithm. The minima #
# positions of the reflectivity will be fitted                                                          #
#########################################################################################################


#########
# INPUT #
#########

# minimum, maximum, and step size of thickness of elastomer in nanometer
min_d = 7500    
max_d = 8500
thickness_step = 1

# wavelength range for the simulation, usually 540nm - 760nm, not sure if 550nm - 750nm would be enough es well --> hard to check, I guess it wouldn't matter too much

wave_start = 540
wave_end = 760

# step size in which the simulation will be performed --> if you have time simulate in 0.01 steps, otherwise 0.1nm steps do not introduce a big error (~1nm)
wave_step = 0.01

# define the output file name, it helps to put the layer structure in the file name
output_file_name = "test.txt"

#########
# START #
#########


# this module has to be installed separately from:  
import openTMM as tmm
import scipy as sp
import numpy as np

# For plotting only
import matplotlib.pyplot as plt 
import time

ta=time.time()

# define a class for the layers to easily access, material, thickness, optical constants
class Layer:    
    "A layer with thickness and permittivity"
    def __init__(self, material, thickness, e_r):
        self.material= material
        self.thickness=thickness*1e-9
        self.permittivity=e_r

    # function to read the n&k values from data files (space separated with  lines) and convert it to permitivity
    def e_r_read(self,waves):
        p = open('MaterialsDataFiles/'+self.material + '.txt','r')
        string=p.read()
        p.close()
        e_r=[]
        w=[]
        n=[]
        k=[]
        linecounter=0
        for thisline in string.split('\n'):
            for x in thisline.split():
                if linecounter==0 and len(x)>0:
                    w.append(float(x))
                if linecounter==1 and len(x)>0:
                    n.append(float(x))
                if linecounter==2 and len(x)>0:
                    k.append(float(x))
            linecounter+=1
        
        # interpolate n&k walues 
        n_new=sp.interp(waves,w,n)
        k_new=sp.interp(waves,w,k)
        e_r_new=[]        
        # calculate epsilon from n&k for every wavelength
        for i in range(len(waves)):
            e_r_new.append(n_new[i]**2-k_new[i]**2+2j*n_new[i]*k_new[i]) # calculate the complex epsilon from n&k

        # not sure anymore why e_r (permittivity) gets appended the waves and e_r_new (complex epsilon)  
        e_r.append(waves)
        e_r.append(e_r_new)    
        
        self.permittivity=e_r


############################
## Define Layer structure ##
############################

## write in the order how your multilayer stack should look, the light will come from the side oft the last layer
## write it like: Layers.append(Layer('material', thickness, []), 'material' must have the name of the txt file where the material data is stored without ".txt" 
## --> thickness in nanometers

Layers=[]
# air
Layers.append(Layer('Au',15,[]))
Layers.append(Layer('Elastomer_RT601',3000,[]))
Layers.append(Layer('SIO2',50,[]))
Layers.append(Layer('Au',10,[]))
Layers.append(Layer('Cr',0.5,[]))
# substrate


# define host media, air, and glass

hm_eps = sp.array([1.5*1.5, 1.0]) 
hm_mu = sp.array([1.0,1.0])
hm_dict = {'epsilonRelative': hm_eps, 'muRelative': hm_mu}


#################################################### 
## Function to convert Frequencies to wavelengths ##
####################################################

# takes Freq in "Hz" and gives waves in "nm"
def Freq_to_wave(f):
    cLight=299792458.0
    wave=cLight/f/1e-9
    return wave

#################################################
## Function to convert wavelengths Frequencies ##
#################################################

# INPUT: waves in "nm", OUTPUT: Freq in "Hz"  
def wave_to_Freq(wave):
    wave=wave*1e-9
    cLight=299792458.0
    Freq=cLight/wave
    return Freq
        


####################################################################
## Function for the wave dependent simulation of the reflectivity ##
####################################################################

def tmm_waves(wave_min, wave_max, wave_step):
    "calculates reflection and transmission for different wavelengths \n INPUT:\n 1. Minimum wavelength (in nm) \n 3. Maximum wavelength (in nm) \n\n OUTPUT:\n wavelength variation (list), transmission (list), reflection (list)"
    f_par=[]
    R_par=[]
    T_par=[]
    waves=[]
    
    # make list of waves
    i=0
    welle = wave_min
    while welle <= wave_max:
        waves.append(welle)
        i +=wave_step
        welle=wave_min+i

    #for i in range(wave_max-wave_min+1):
    #    waves.append(wave_min+i)
    for x in Layers:
        x.e_r_read(waves)
    for i in range(len(waves)):
        Freq_min=wave_to_Freq(waves[i])
        Freq_max=Freq_min
        eps=sp.ones(len(Layers),dtype=complex) # its important to generate a complex array, otherwise the complex permittivity will be reduced to its real part --> you get wrong results
        mu=sp.ones(len(Layers))
        height=sp.ones(len(Layers))
        for x in range(len(Layers)):
            eps[x]=Layers[x].permittivity[1][i]
            height[x]=Layers[x].thickness
        stack = {'epsilonRelative': eps, 'muRelative': mu, 'height': height, 'hostMedia': hm_dict }
        mls= tmm.Layer(stack)
        # call the function to calculate transmission and reflection from the tmm library
        f1, T1, R1 = mls.TRvsFreq(Freq_min, Freq_max,1, 0, 'parallel')
        f_par.append(Freq_to_wave(f1[0]))
        T_par.append(T1[0])
        R_par.append(R1[0])

    return f_par, T_par, R_par


###############################
# Function to find the minima #
###############################

def peakdetect(y_axis, x_axis = None, lookahead_min = 10, lookahead_max = 9,  delta=0.05):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200) 
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    #x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    #if lookahead < 1:
     #   raise ValueError, "Lookahead must be '1' or above in value"
    #if not (np.isscalar(delta) and delta >= 0):
     #   raise ValueError, "delta must be a positive number"
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead_min], 
                                        y_axis[:-lookahead_min])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead_max].max() < mx:
                #print 'max'
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead_min >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead_min].min() > mn:
                #print 'min'
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead_min >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

   #print min_peaks
    return min_peaks


# make list of thickness values
d_values = []
d = min_d
for i in range(max_d - min_d+1):
    d_values.append(d)
    d += thickness_step


# output file, define name for yourself at top of the script
# format should be like: Sim + Stack + wavelength-range .txt
p = open(output_file_name, "a")


#################################################
## Loop over different thickness for one layer ##
#################################################

for i in range(len(d_values)):    
    # output file, define name for yourself at top of the script
    # format should be like: Sim + Stack + wavelength-range .txt
    p = open(output_file_name, "a")
    
    p.write(str(d_values[i]) + "\n")
    print d_values[i]
    Layers[1].thickness=d_values[i]*1e-9  

    # perform simulation for a certain wavelength range
    f, T, R = tmm_waves(wave_start,wave_end,wave_step)

    # start matlab minima finding algorithm
    found_minima = np.array(peakdetect(np.array(R),np.array(f)))

    ## Old minima finding algorithm, less reliable as kinks in the n&k files can lead to wrong local minima
    # if d_values[i] <=5000:
    #     for i in range(len(f)-100):
    #         x = R[i+50]
    #         if x]<1.0 and R[i+49]>= x] and R[i+51] >= x] and R[i+48] > x] and R[i+52] > x] and R[i]>x] and R[i+80]>x] and R[i+30]>x] and R[i+20]>=x] and R[i+100]>x] and R[i+40]>x] and R[i+70]>x]: # check if 2 points before and after are larger than minimum
    #             p.write(str(f[i+50]) + "\t" + str(x]) + "\n")
    #     p.write("\n")

    # else:

    #     for i in range(len(f)-12):
    #         x = R[i+6]
    #         if x<1.0 and R[i]>x and R[i+8]>x and R[i+4]>x and R[i+2]>=x and R[i+12]>x and R[i+5]>x and R[i+7]>x: # check if 2 points before and after are larger than minimum
    ##

    # write minima positions to file if a minimum was found for the thickness
    if len(found_minima)>0:
        for i in range(len(found_minima[:,0])):
            p.write(str(found_minima[:,0][i]) + "\t" + str(found_minima[:,1][i]) + "\n")
        p.write("\n")
    else:
        p.write("\n")
    # close file
    p.close()    

        
te=time.time()

print "Total time the program needed is: ", str(te-ta), "seconds"
