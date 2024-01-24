import numpy as np
import pandas as pd
from tqdm import tqdm

# Define a SSID_4ptFilter function that takes data, tcrit_u, tcrit_l, n, ewma, and stp as parameters
def SSID_4ptFilter(data, tcrit_u, tcrit_l, n, ewma, stp):
    """
    Title: 4 Points Filter Function
    Auth: Max Horner & Shivam Chopra
    Description: Statistical filter for SS and TS indication of PV. 
    Uses 4 corner approach where 4 points in a moving window are assessed
    by taking the ratio of the difference between the min/max filtered values
    and the filtered variance for the window. 4 points selected as a balance for
    computing performance vs ability to identify periodicity in data. 
    References: Dr. R Rhinehard 2019, OSU 
    """
    pbar = tqdm(total=100)                      # Initialize a progress bar
    # Create an empty data frame with 11 columns and the same number of rows as data
    SSData = pd.DataFrame(np.zeros((len(data), 11)))
    # Rename the columns
    SSData.columns = ["data","tstat", "ss","iread1","iread2","iread3","iread4","y1filt","y2filt","y3filt","y4filt"]
    if stp < 1:     stp = 1                     # Set stp to 1 if it is less than 1
    y = np.zeros(n)                             # Create a numpy array of zeros with length n
    yold, data_var, noise_ampl, random = 0      # Initialize yold, data_var, noise_ampl and random as 0
    SS = 0.5                                    # SS is indeterminate
    cewma = 1 - ewma                            # Calculate cewma 
    g = ((5**0.5-1)/2)                          # Calculate g
    iput, iread4 = 1, n                         # Initialize iput, iread2, iread3, and iread4
    iread2 = int(g*n)
    iread3 = int((1-g)*n)
    y1filt, y2filt, y3filt, y4filt = 0          # Initialize y1filt, y2filt, y3filt, and y4filt to 0
    
    for i in range(1, len(data)):                           # Loop through the data, filter, data, find SS
        
        pbar.update((i/len(data))*100 - pbar.n)             # Progress bar so user doesn't rage quit because it takes ~10-15 minutes to process large datasets
        y[iput-1] = data[i] + noise_ampl*random*0           # Pull PV from input dataset, noise amplitude currently ignored
        SSData["data"][i] = data[i]                         # Add PV to output dataframe for debugging
        # Calculate filtered variance, filter value of 0.05 seleced based on advice from Russell Rhinehart 2019 (OK State)
        data_var = (0.05) * 0.5 * (y[iput-1] - yold)**2 + 0.95*data_var      
        yold = y[iput-1]                                    # Add current PV to yold for next variance calculation
        
        y1filt = ewma * y[iput-1] + cewma * y1filt          # Calculate EWMA value at first index
        y2filt = ewma * y[iread2-1] + cewma * y2filt        # Calculate EWMA value at second index
        y3filt = ewma * y[iread3-1] + cewma * y3filt        # Calculate EWMA value at third index
        y4filt = ewma * y[iread4-1] + cewma * y4filt        # Calculate EWMA value at fourth index
        
        SSData["y1filt"][i] = y1filt                        # Add PV to output dataframe for debugging
        SSData["y2filt"][i] = y2filt                        # Add PV to output dataframe for debugging
        SSData["y3filt"][i] = y3filt                        # Add PV to output dataframe for debugging
        SSData["y4filt"][i] = y4filt                        # Add PV to output dataframe for debugging
        
        SSData["iread1"][i] = iput                          # Add PV to output dataframe for debugging
        SSData["iread2"][i] = iread2                        # Add PV to output dataframe for debugging
        SSData["iread3"][i] = iread3                        # Add PV to output dataframe for debugging
        SSData["iread4"][i] = iread4                        # Add PV to output dataframe for debugging
        
        iput += 1                                           # Incrementing each corner variable
        iread2 += 1
        iread3 += 1
        iread4 += 1
        
        if iput > n:    iput = 1                            # Reset iput, iread2, iread3, and iread4 if they exceed n
        if iread2 > n:  iread2 = 1                          # When an index exceeds the window size, re-initialize the index
        if iread3 > n:  iread3 = 1
        if iread4 > n:  iread4 = 1
        
        maxfilt, minfilt = y1filt                           # Initialize maxfilt and minfilt to hold min and max filtered values
        if y2filt > maxfilt:    maxfilt = y2filt            # Identify minimum/maximum filtered values
        if y2filt < minfilt:    minfilt = y2filt
        if y3filt > maxfilt:    maxfilt = y3filt
        if y3filt < minfilt:    minfilt = y3filt
        if y4filt > maxfilt:    maxfilt = y4filt
        if y4filt < minfilt:    minfilt = y4filt
        
        if data_var < 0.01:     data_var = 0.01             # Set the minimum data_var to 0.01
        t = (maxfilt - minfilt) / (data_var)**0.5           # Calculate the t-statistic
        if t > 5:               t = 5                       # Set the maximum t-statistic to 5
        
        for ii in range(i, i+stp):                          # Add T-statistic to output dataframe for later troubleshooting
            if ii < len(data):  SSData["tstat"][ii] = t
        
        if i >= n:
            if t <= tcrit_l:    SS = 1                      # If the t-statistic is lower than the lower threshold, probable SS               
            if t > tcrit_u:     SS = 0                      # If the t-statistic is higher than the upper threshold, probable TS
        
        for ii in range(i, i+stp):                          # Add the SS indicator to SSData["ss"]
            if ii < len(data):  SSData["ss"][ii] = SS       # Add steady-state indicator to output dataframe
    
    pbar.close()                            # Close the progress bar
    return SSData                           # Return SSData
