def eqdistant_global_quantization(data,bin_num):
    import pandas as pd
    import numpy as np
    max_value=data.max().max()
    min_value=data.min().min()
    bin_width=(max_value-min_value)/bin_num
    eq_global_binedges=np.arange(min_value, max_value, bin_width)
    eq_global_binedges = np.append(eq_global_binedges, max_value)
    eq_global_center= [ (eq_global_binedges[i]+bin_width/2) for i in np.arange(len(eq_global_binedges))]
    inds = np.digitize(data, eq_global_binedges)
    # np.digitize discritize x for bins[i-1] <= x < bins[i]
    inds = np.where(inds > bin_num,bin_num,inds) 
    # ==> if x==max_value, it put x on new bins(bin_num+1) ==>np.where is used to put x==max_value in the last bin(bin_num)
    columnns_name=data.columns
    eq_global_quantized= pd.DataFrame(inds,columns=columnns_name)
    eq_global_quantized_value=eq_global_quantized.replace(to_replace = (np.arange(len(eq_global_center))+1), value =eq_global_center) 
    return eq_global_quantized,eq_global_quantized_value, eq_global_center, eq_global_binedges 

def eqdistant_local_quantization(data,bin_num):
    import pandas as pd
    import numpy as np
    max_value=data.max()
    min_value=data.min()
    bin_width=(max_value-min_value)/bin_num
    eq_local_binedges=np.arange(min_value, max_value, bin_width)
    eq_local_binedges = np.append(eq_local_binedges, max_value)
    eq_local_center= [ (eq_local_binedges[i]+bin_width/2) for i in np.arange(len(eq_local_binedges))]
    inds = np.digitize(data, eq_local_binedges)
    # np.digitize discritize x for bins[i-1] <= x < bins[i]
    inds = np.where(inds > bin_num,bin_num,inds) 
    # ==> if x==max_value, it put x on new bins(bin_num+1) ==>np.where is used to put x==max_value in the last bin(bin_num)
    eq_local_quantized= pd.DataFrame(inds, columns=['quantized'])
    eq_local_quantized_value=eq_local_quantized.replace(to_replace = (np.arange(len(eq_local_center))+1), value =eq_local_center) 
    return eq_local_quantized,eq_local_quantized_value, eq_local_center, eq_local_binedges

def km_global_quantization(data,bin_num):
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    max_value=data.max().max()
    min_value=data.min().min()
    array=data.values
    array=np.reshape(array, (array.shape[0]*array.shape[1],1))
    kmeans = KMeans(n_clusters=bin_num,random_state=0).fit(array)
    global_centers=kmeans.cluster_centers_
    global_centers=np.sort(global_centers, axis=0) 
    global_binedges=[min_value]
    for i in np.arange(len(global_centers)):
        if i<len(global_centers)-1:
            global_binedges.append(np.squeeze((global_centers[i]+global_centers[i+1])/2).tolist())
        else:
            global_binedges.append(max_value)
    inds = np.digitize(data, global_binedges)
    # np.digitize discritize x for bins[i-1] <= x < bins[i]
    inds = np.where(inds > bin_num,bin_num,inds) 
    # ==> if x==max_value, it put x on new bins(bin_num+1) ==>np.where is used to put x==max_value in the last bin(bin_num)
    columnns_name=data.columns
    km_global_quantized= pd.DataFrame(inds,columns=columnns_name)
    km_global_quantized_value=km_global_quantized.replace(to_replace = (np.arange(len(global_centers))+1), value =global_centers)         
    return km_global_quantized,km_global_quantized_value, global_centers, global_binedges

def km_local_quantization(data,bin_num):
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    max_value=data.max()
    min_value=data.min()
    array=data.values
    array=np.reshape(array, (array.shape[0],1))
    kmeans = KMeans(n_clusters=bin_num,random_state=0).fit(array)
    local_centers=kmeans.cluster_centers_
    local_centers=np.sort(local_centers, axis=0) 
    local_binedges=[min_value]
    for i in np.arange(len(local_centers)):
        if i<len(local_centers)-1:
            local_binedges.append(np.squeeze((local_centers[i]+local_centers[i+1])/2).tolist())
        else:
            local_binedges.append(max_value)
    inds = np.digitize(data, local_binedges)
    # np.digitize discritize x for bins[i-1] <= x < bins[i]
    inds = np.where(inds > bin_num,bin_num,inds) 
    # ==> if x==max_value, it put x on new bins(bin_num+1) ==>np.where is used to put x==max_value in the last bin(bin_num)
    km_local_quantized= pd.DataFrame(inds,columns=['quantized'])
    km_local_quantized_value=km_local_quantized.replace(to_replace = (np.arange(len(local_centers))+1), value =local_centers)         
    return km_local_quantized,km_local_quantized_value, local_centers, local_binedges

def floor_global_quantization(data,a_f):
    import pandas as pd
    import numpy as np
    floor_global_quantized= 1+((2*data+a_f)/(2*a_f)).apply(np.floor)
    max_value=data.max().max()
    min_value=data.min().min()
    binedges=[min_value, a_f/2]
    i=0
    max_binedges_floor=max(binedges)
    while max_binedges_floor <max_value:
      binedges.append(binedges[i+1]+a_f)
      max_binedges_floor=max(binedges)
      i+=1
    global_centers =[]
    for i in np.arange(len(binedges)):
            if i<len(binedges)-1:
                global_centers.append(np.squeeze((binedges[i]+binedges[i+1])/2).tolist())
    floor_global_quantized_value=floor_global_quantized.replace(to_replace = (np.arange(len(global_centers))+1), value =global_centers)  
    return floor_global_quantized,floor_global_quantized_value, global_centers, binedges

def likelihood_calculator_original_data(data_original):
    import pandas as pd
    import numpy as np
    x = data_original
    x=x.to_numpy()
    # bins="auto"
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    density,bins= np.histogram(x,bins = "auto",density=1) 
    counts_per_bin,bins= np.histogram(x,bins = "auto") # vector of bin_use
    density=density[density != 0] # delete zero element(log(0) would make problem)
    counts_per_bin=counts_per_bin[counts_per_bin != 0]
    log_likelihood=np.log(density) 
    sum_log_likelihood=log_likelihood*counts_per_bin
    sum_log_likelihood=np.sum(sum_log_likelihood)
    sum_log_likelihood=-sum_log_likelihood
    return sum_log_likelihood

def likelihood_calculator_quantized_data(data,bin_edges):
    import pandas as pd
    import numpy as np
    x = data
    x=x.to_numpy()
    binx=bin_edges
#   binx=binx.to_numpy()
    binx=np.squeeze(binx) # convert (n,1) to (n,)
    density,bins= np.histogram(x,bins = binx,density=1) 
    counts_per_bin,bins= np.histogram(x,bins = binx) # vector of bin_use
    density=density[density != 0] # delete zero element(log(0) would make problem)
    counts_per_bin=counts_per_bin[counts_per_bin != 0]
    log_likelihood=np.log(density) 
    sum_log_likelihood=log_likelihood*counts_per_bin
    sum_log_likelihood=np.sum(sum_log_likelihood)
    sum_log_likelihood=-sum_log_likelihood
    return sum_log_likelihood
#
def projection_to_fine_prob_grid(signal_qauntized,old_binedges,max_value,min_value,num_new_bin):
    import numpy as np
    hist,bins= np.histogram(signal_qauntized,bins = old_binedges) 
    old_prob_dist=hist/sum(hist)
    new_bin_binwidth=(max_value-min_value)/num_new_bin
    new_binedges=np.arange(min_value, max_value, new_bin_binwidth)
    new_binedges=np.append(new_binedges, max_value)
    old_intervals=(np.delete(old_binedges, 0)-np.delete(old_binedges, -1))
    new_intervals=(np.delete(new_binedges, 0)-np.delete(new_binedges, -1))
    refrence_to_endof_oldedges=np.delete(old_binedges, 0) #cut the first edge
    new_prob=[]
    for i in np.arange(num_new_bin):
        cc=refrence_to_endof_oldedges[new_binedges[i]<refrence_to_endof_oldedges]
        loc1=old_intervals.shape[0]-cc.shape[0]+1 #this return the location(refrence to end of oldedges) of edge1 for a new bin
        cc2=refrence_to_endof_oldedges[new_binedges[i+1]<refrence_to_endof_oldedges]
        loc2=old_intervals.shape[0]-cc2.shape[0]+1 #this return the location of edge2 for a new bin

        if loc2<=len(old_intervals):
            if loc1==loc2:
                #this means new bin inside the old bin
                new_prob.append(np.squeeze(old_prob_dist[loc1-1]*new_bin_binwidth/old_intervals[loc1-1]).tolist())
            if loc1!=loc2:
                #this means new bin was split between two old bin
                new_prob.append(np.squeeze((old_prob_dist[loc1-1]*(refrence_to_endof_oldedges[loc1-1]-new_binedges[i])/old_intervals[loc1-1])+\
                                (old_prob_dist[loc1]*(new_binedges[i+1]-refrence_to_endof_oldedges[loc1-1])/old_intervals[loc1])).tolist())
        elif loc1<=len(old_intervals):
            #this means part of new bin (i.e, edge1 ) inside the old bin range but edge2 is beyond the range of old_ interval
            new_prob.append(np.squeeze(old_prob_dist[loc1-1]*(refrence_to_endof_oldedges[loc1-1]-new_binedges[i])/old_intervals[loc1-1]).tolist()) 
        else:
            new_prob.append(0) # this means new bin is beyond the range of old_ interval        
    return new_prob,new_binedges

def Kullback_Leibler_divergence_score(P,Q):
    import numpy as np
    P=P/np.sum(P)
    Q=Q/np.sum(Q)
    dDkl=[]
    for i in np.arange(len(P)):
        if (P[i]!=0 and Q[i]!=0):
            dDkl.append(np.squeeze(P[i]*(np.log2(P[i]/Q[i]))).tolist())
        else: 
            dDkl.append(0)
    
    Dkl = sum(dDkl)
    return Dkl

def entropy_1D(signal_qauntized):
    import numpy as np
    (unique, counts) = np.unique(signal_qauntized, return_counts=True)
    prob_dist=counts/sum(counts)
    H_x=[]
    for i in np.arange(len(prob_dist)):
            H_x.append(np.squeeze(-prob_dist[i]*(np.log2(prob_dist[i]))).tolist())
    Entropy=np.nansum(H_x)
    return Entropy

def entropy_terms_2D(joint_distribution):
    import numpy as np
    np.seterr(all="ignore")
    JD=joint_distribution
    nr_x=JD.shape[0]
    nr_y=JD.shape[1]
    prob=JD/sum(sum(JD))
    marginal_px=prob.sum(axis=1)
    marginal_py=prob.sum(axis=0)
    mutual_xy=np.zeros((nr_x, nr_y))
    joint_entropy=np.zeros((nr_x, nr_y))
    H_x=np.zeros(nr_x)
    H_y=np.zeros(nr_y)
    for i in np.arange(nr_x):
        for j in np.arange(nr_y):
            mutual_xy[i,j]=prob[i,j]*np.log2(prob[i,j]/(marginal_px[i]*marginal_py[j]))
            joint_entropy[i,j]=prob[i,j]*np.log2(prob[i,j])
    #####################################################
    for i in np.arange(nr_x):
        H_x[i]=marginal_px[i]*np.log2(marginal_px[i])
    for i in np.arange(nr_y):
        H_y[i]=marginal_py[i]*np.log2(marginal_py[i])
    
    H_x=-np.nansum(H_x)   
    H_y=-np.nansum(H_y)  
    mutual_xy=np.nansum(np.nansum(mutual_xy))
    joint_entropy=-np.nansum(np.nansum(joint_entropy)) 
    Hcond_x_y =H_x-mutual_xy
    Hcond_y_x =H_y-mutual_xy
    return joint_entropy,mutual_xy,Hcond_x_y,Hcond_y_x

def merging_operator_2varibles(signal1,signal2):
    import pandas as pd
    import numpy as np
    signal1=list(map(str, signal1))
    signal2=list(map(str, signal2))
    merg=list(map(''.join, zip(signal1, signal2)))
    merg=list(map(lambda num: int(num), merg))
    unique_elements=np.unique(merg)
    merg_signal = pd.DataFrame(merg)
    # relabeling
    merg_signal.replace(to_replace=unique_elements, value=(np.arange(len(unique_elements))+1))
    return merg_signal

def info_stats_merging_operator_multiple_varibles(data,order):
    import pandas as pd
    import numpy as np
    previous_merg=[]
    H_marg=[]
    previous_merg_2nd=[]
    for i in order:
        H_marg.append(entropy_1D(data.iloc[:,i-1]))
    for i in order:
        if i==order[0]:
            previous_merg=merging_operator_2varibles(data.iloc[:,i-1].astype('int64'),data.iloc[:,i-1].astype('int64')).squeeze()
        else:
            previous_merg_2nd=previous_merg
            previous_merg=merging_operator_2varibles(previous_merg,data.iloc[:,i-1].astype('int64')).squeeze()

    Joint_entrpoy=entropy_1D(previous_merg)
    total_correlation_last_added=sum(H_marg)-entropy_1D(previous_merg)
    mutual_last_added=H_marg[-1]-(entropy_1D(previous_merg)-entropy_1D(previous_merg_2nd))
    return Joint_entrpoy,mutual_last_added,total_correlation_last_added

def greedy_optimozer_maxJE(data):
    import numpy as np
    import itertools
    idx_left=list(np.arange(data.shape[1])+1)
    idx_selected=[]
    #  Finding First Station
    H_marg=[]
    for i in idx_left:
        H_marg.append(entropy_1D(data.iloc[:,i-1]))
    idx_new_select=idx_left[H_marg.index(max(H_marg))]
    idx_selected.append(idx_new_select)
    idx_left.remove(idx_new_select)
    #  Finding ranking for 2nd to end
    for j in np.arange(data.shape[1]-1):
        search=[]
        for i in idx_left :
            combination_order=[idx_selected,[i]]
            combination_order= list(itertools.chain.from_iterable(combination_order))
            search.append(info_stats_merging_operator_multiple_varibles(data,combination_order)[0])
        idx_new_select=idx_left[search.index(max(search))]
        idx_selected.append(idx_new_select)
        idx_left.remove(idx_new_select)
    Ranked_order=idx_selected
    return Ranked_order

def greedy_optimozer_minT(data):
    import numpy as np
    import itertools
    idx_left=list(np.arange(data.shape[1])+1)
    idx_selected=[]
    #  Finding First Station
    H_marg=[]
    for i in idx_left:
        H_marg.append(entropy_1D(data.iloc[:,i-1]))
    idx_new_select=idx_left[H_marg.index(max(H_marg))]
    idx_selected.append(idx_new_select)
    idx_left.remove(idx_new_select)
    #  Finding ranking for 2nd to end
    for j in np.arange(data.shape[1]-1):
        search=[]
        for i in idx_left :
            combination_order=[idx_selected,[i]]
            combination_order= list(itertools.chain.from_iterable(combination_order))
            search.append(info_stats_merging_operator_multiple_varibles(data,combination_order)[1])
        idx_new_select=idx_left[search.index(min(search))]
        idx_selected.append(idx_new_select)
        idx_left.remove(idx_new_select)
    Ranked_order=idx_selected
    return Ranked_order
