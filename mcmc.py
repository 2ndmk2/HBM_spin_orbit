import numpy as np
import spherecluster
from scipy.integrate import quad
import scipy
from numba import jit

count = 0
count2 = 0


    
def g_z(a_now=2):
    rand = np.random.rand() * (a_now**0.5 -(1/a_now)**0.5) + (1/a_now)**0.5
    return rand**2

def prior_spin(nx):
    lower= np.zeros(3*nx+1)
    upper = np.zeros(3*nx+1)
    lower[0:nx] = 0
    lower[nx:2*nx] = 0
    lower[2*nx:3*nx] = -np.pi
    upper[0:nx] = 1
    upper[nx:2*nx] =  1
    upper[2*nx:3*nx] =  np.pi
    lower[3*nx] = 0
    upper[3*nx] = 1000
    return lower, upper

def print_non_finite(a,b,c,d,e,f, g ):
    flag_0 = (np.isfinite(g)==False)
    if len(d[flag_0]):
        print(a[flag_0], b[flag_0], c[flag_0], d[flag_0], e[flag_0], f[flag_0])


def sin_set_correct_values(sin_PA, dmy=1-1e-7):

    flag = sin_PA > dmy
    sin_PA[flag] = dmy
    flag = sin_PA < -dmy
    sin_PA[flag] = -dmy
    return sin_PA

def w_set_correct_values(w, dmy=1-1e-7):

    flag = w > dmy
    w[flag] = dmy
    flag = w < 1-dmy
    w[flag] = 1-dmy
    return w

## u =cos_istar, v =  cosi_disk, w = cos psi
def jacobian_calc(u, v, delta_PA):
    w = u * v + np.sqrt((1-u)*(1+u)) *  np.sqrt((1-v)*(1+v)) * np.cos(delta_PA)
    w = w_set_correct_values(w, dmy=1-1e-7)
    sin_PA = np.sqrt(1-v**2) * np.sin(delta_PA)/np.sqrt(1-w**2)
    sin_PA = sin_set_correct_values(sin_PA)
    w_v = -v * np.sqrt(1-u**2) * np.cos(delta_PA)/np.sqrt(1-v**2) + u
    w_delPA = - np.sqrt(1-u**2) * np.sqrt(1-v**2) * np.sin(delta_PA)
    PA_v = -(1/np.sqrt((1-sin_PA)*(1+sin_PA))) *( -(v * np.sin(delta_PA)/(np.sqrt((1-v)*(1 + v)) * np.sqrt((1-w)*(1 + w)))) \
                                       + (w * np.sqrt((1-v)*(1 + v)) * np.sin(delta_PA)/((1-w)*(1 + w))**1.5)\
                                       *( -(v*np.sqrt((1-u)*(1 + u)) * np.cos(delta_PA)/(np.sqrt((1-v)*(1 + v)))) + u))
    PA_delta_PA = (1/np.sqrt((1-sin_PA)*(1+sin_PA))) *( (np.sqrt((1-v)*(1 + v)) * np.cos(delta_PA)/np.sqrt((1-w)*(1 + w))) - \
                                             w*np.sqrt((1-u)*(1 + u)) * ((1-v)*(1 + v))*(np.sin(delta_PA)**2)/(((1-w)*(1 + w))**1.5))
    



    return_sum = np.abs(w_v * PA_delta_PA  - w_delPA *PA_v )
    """
    if not np.isfinite(np.sum(return_sum)):
        test = w*np.sqrt((1-u)*(1 + u)) * ((1-v)*(1 + v))*(np.sin(delta_PA)**2)/(((1-w)*(1 + w))**1.5)
        test2 = (np.sqrt((1-v)*(1 + v)) * np.cos(delta_PA)/np.sqrt((1-w)*(1 + w)))
        test3 = 1/np.sqrt((1-sin_PA)*(1+sin_PA))



        #print("w_v:")
        #print_non_finite(u,v,delta_PA,test, test2, w_v)
        print("before")
        print("PA_delta_PA:")
        print_non_finite(test,test2,test3,u,v, sin_PA, PA_delta_PA)
        #print("w_delPA:")
        #print_non_finite(u,v,delta_PA,test, test2, w_delPA)
        print("PA_v :")
        print_non_finite(test,test2,test3,u,v, sin_PA, PA_v )

        dmy = 1-1e-5

        print("after")
        flag = w > dmy
        w[flag] = dmy
        sin_PA = np.sqrt(1-v**2) * np.sin(delta_PA)/np.sqrt(1-w**2)
        flag = sin_PA > dmy
        sin_PA[flag] = dmy
        flag = sin_PA < -dmy
        sin_PA[flag] = -dmy
        test = w*np.sqrt((1-u)*(1 + u)) * ((1-v)*(1 + v))*(np.sin(delta_PA)**2)/(((1-w)*(1 + w))**1.5)
        test2 = (np.sqrt((1-v)*(1 + v)) * np.cos(delta_PA)/np.sqrt((1-w)*(1 + w)))
        test3 = 1/np.sqrt((1-sin_PA)*(1+sin_PA))

            
        print("PA_delta_PA:")
        print_non_finite(test,test2,test3,u,v, sin_PA, PA_delta_PA)
        print("PA_v :")
        print_non_finite(test,test2,test3,u,v, sin_PA, PA_v )

        dmy = 1-1e-10
        print("after2")
        flag = w > dmy
        w[flag] = dmy
        sin_PA = np.sqrt(1-v**2) * np.sin(delta_PA)/np.sqrt(1-w**2)
        flag = sin_PA > dmy
        sin_PA[flag] = dmy
        flag = sin_PA < -dmy
        sin_PA[flag] = -dmy
        test = w*np.sqrt((1-u)*(1 + u)) * ((1-v)*(1 + v))*(np.sin(delta_PA)**2)/(((1-w)*(1 + w))**1.5)
        test2 = (np.sqrt((1-v)*(1 + v)) * np.cos(delta_PA)/np.sqrt((1-w)*(1 + w)))
        test3 = 1/np.sqrt((1-sin_PA)*(1+sin_PA))

            
        print("PA_delta_PA:")
        print_non_finite(test,test2,test3,u,v, sin_PA, PA_delta_PA)
        print("PA_v :")
        print_non_finite(test,test2,test3,u,v, sin_PA, PA_v )



    """
    return return_sum
    


def log_probs_spin(params, data):
    global count
    global count2
    data1,data2, err1, err2, prior_low, prior_up  = data
    nx = len(data1)

    log_prob = 0
    cosi_star = params[0:nx]
    cosi_disk = params[nx:2*nx]
    delta_PA = params[2*nx:3*nx]
    cos_psi = cosi_disk * cosi_star + np.sqrt(1-cosi_disk**2) *  np.sqrt(1-cosi_star**2) * np.cos(delta_PA)

    kappa = params[3*nx]
    flag =  (prior_low < params) *  (prior_up > params)
    index_now = np.arange(len(params))
    
    if len(params[flag]) !=3*nx+1:

        count += 1
        return -np.inf

    count2 += 1

    sum_1 = -0.5 * np.sum( ((data2 - cosi_star)/err2)**2) 
    sum_2 = -0.5 * np.sum( ((data1 - cosi_disk)/err1)**2)
    sum_3 = np.sum(-kappa + kappa * cos_psi + np.log( (kappa/(2*np.pi*(1 - np.exp(-2*kappa))))))

    jacob = np.sum(np.log(jacobian_calc(cosi_star, cosi_disk, delta_PA)))
    sum_now = sum_1 + sum_2 + sum_3 + jacob
    if not np.isfinite(sum_now):
        print(sum_1 + sum_2 + sum_3 + jacob, sum_1, sum_2, sum_3, jacob)

    return sum_1 + sum_2 + sum_3 + jacob


def initialize_walker(i_disk_obs, i_star_obs, n_walker):

    n_data = len(i_disk_obs)
    n_para = len(i_disk_obs) * 3 + 1
    walkers = np.zeros(( n_walker, n_para)) 

    for i in range(n_walker):
        walkers[i,0:n_data] = i_star_obs + np.random.randn(len(i_star_obs)) * 0.01
        walkers[i,n_data:2*n_data] = i_disk_obs + np.random.randn(len(i_star_obs)) * 0.01
        flag_0 = (walkers[i,0:2*n_data] <= 0) 
        flag_pi2 = (walkers[i,0:2*n_data] >= 1) 
        walkers[i,0:2*n_data][flag_0] = 1e-7
        walkers[i,0:2*n_data][flag_pi2] = 1 - 1e-7
        walkers[i,2*n_data:3*n_data] = -0.01 + 0.02 * np.random.rand(n_data) 
        walkers[i, 3 *n_data] = 50+np.random.rand() * 100
        
    return walkers

def delta_PA_to_correct_range(pa):


    pa_mod=  np.fmod(pa, 2 * np.pi)
    flag = pa_mod < -np.pi
    flag2 = pa_mod  > np.pi
    pa_mod[flag] += 2* np.pi
    pa_mod[flag2] -= 2 * np.pi
    return pa_mod

def init_walker_prior(walker, prior_low, prior_up):

    n_walk, npara = np.shape(walker)
    count_ng = 0

    for n_dmy in range(n_walk):
        flag =  (prior_low <= walker[n_dmy]) *  (prior_up >= walker[n_dmy])
    
        if len(walker[n_dmy][flag]) != len(walker[0]):
            print("NG")
            count_ng += 1
    print("Num of initial walkers, whose positions are out of prior ranges:", count_ng)


@jit    
def mcmc_gw10_spin(log_probabilty, data, run_num, n_walker, n_burns=1000):
    global count
    global count2

    i_disk, i_star, i_disk_err, i_star_err = data 
    n_para = len(i_disk) * 3  + 1
    n_data = len(i_disk)
    walkers_init = initialize_walker(i_disk, i_star, n_walker)
    walkers =np.copy(walkers_init)
    walkers_return = np.zeros(((run_num - n_burns)*n_walker, n_para))
    walker_sum = []
    lower_prior, upper_prior =  prior_spin(len(i_disk))
    data_for_fit = i_disk, i_star, i_disk_err, i_star_err, lower_prior, upper_prior
    init_walker_prior(walkers_init, lower_prior, upper_prior)
    sum_log = []

    for n in range(run_num):
        if n%100 == 0:
            print(n, count, count2)

        walkers_now = np.zeros(( n_walker, n_para)) 

        for k in range(n_walker):
            while(1):
                n_rand = np.random.randint(0, n_walker)
                if n_rand !=k:
                    break            
            
            z_now = g_z(1.2)
            proposed_x = walkers[n_rand] + z_now * (-walkers[n_rand] + walkers[k])
            proposed_x[2*n_data: 3*n_data] = delta_PA_to_correct_range(proposed_x[2*n_data: 3*n_data])
            upper_logprob =  log_probabilty(proposed_x,data_for_fit )
            lower_logprob = log_probabilty(walkers[k],data_for_fit)
            sum_log.append(lower_logprob)

            if not np.isfinite(upper_logprob):
                q = 0
            else:
                q = (z_now**(n_para-1)) * np.exp(upper_logprob-lower_logprob)
            r = np.random.rand()
            if r < q:   
                #print("success!")             
                walkers_now[k] = proposed_x
            else:

                walkers_now[k] = walkers[k]
                
        walkers = walkers_now
        i_now = n-n_burns
        if i_now >-1:

            walkers_return[i_now*n_walker:(i_now+1) * n_walker,:] = walkers

            
    return walkers_return, walkers_init, sum_log

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def reverse_minus(theta):
    if theta < np.pi/2.0:
        return theta
    else:
        return np.pi - theta

def data_make(n_obs, kappa=100, err = (0.3/180.0)*np.pi):
    i_disk = []
    i_star = []
    i_disk_err = np.ones(n_obs) * err
    i_star_err = np.ones(n_obs) * err
    
    cos_var = []
    
    i_obs_count = 0
    while (1):
        n_disk = sample_spherical(1)[:,0]
        n_star = spherecluster.sample_vMF(n_disk, kappa, 1)[0]
        psi_ang= np.arccos(np.dot(n_disk, n_star)) * 180/np.pi
        
        i_obs_count += 1
        i_disk.append(reverse_minus(np.arccos( n_disk[2]) + np.random.normal() * err))
        i_star.append(reverse_minus(np.arccos( n_star[2]) + np.random.normal() * err))
        
        if i_obs_count == n_obs:
            break
        
    return np.array(i_disk), np.array(i_star), i_disk_err, i_star_err

def set_zero_to_one(value):
    if value > 1:
        value = 1-1e-7
    if value < 0:
        value = 1e7
    return value
    
def data_make_cosi(n_obs, kappa=100, err = 0.01):
    cosi_disk = []
    cosi_star = []
    cosi_disk_err = np.ones(n_obs) * err
    cosi_star_err = np.ones(n_obs) * err
    
    cos_var = []
    cospsi_arr = []
    i_obs_count = 0
    
    while (1):
        n_disk = sample_spherical(1)[:,0]
        n_star = spherecluster.sample_vMF(n_disk, kappa, 1)[0]
        psi_ang= np.arccos(np.dot(n_disk, n_star)) * 180/np.pi
        cospsi_arr.append(np.dot(n_disk, n_star))
        
        i_obs_count += 1
        cosi_disk.append(set_zero_to_one(np.abs(n_disk[2] + np.random.normal() * err)))
        cosi_star.append(set_zero_to_one(np.abs(n_star[2] + np.random.normal() * err)))

        if i_obs_count == n_obs:
            break
        
    return np.array(cosi_disk), np.array(cosi_star),cosi_disk_err, cosi_star_err, np.array(cospsi_arr)


## HBM
def kappa_von_mises(kappa, cos_psi):
    return  np.sum(-kappa + kappa * cos_psi + np.log( (kappa/(2*np.pi*(1 - np.exp(-2*kappa))))))
   
    
def kappa_von_mises_2d(kappa, cos_psi):
    
    nx, ny = np.shape(kappa)
    cospsi_tile = np.tile(cos_psi,(ny,1)).T

    test = np.sum(-kappa + kappa * cospsi_tile + np.log( (kappa/(2*np.pi*(1 - np.exp(-2*kappa))))), axis = 0)
    
    return  np.mean(np.exp(-kappa + kappa * cospsi_tile + np.log( (kappa/(2*np.pi*(1 - np.exp(-2*kappa))))))   , axis = 0)


def kappa_von_mises_from_kappa(kappa, cos_psi_arr):
    
    return  np.exp(-kappa + kappa * cos_psi_arr + np.log( (kappa/(2*np.pi*(1 - np.exp(-2*kappa))))))
   
    

def kappa_dist(data, kappa_min=0.1, kappa_max=100, N_kappa_sample=1000, cosi_sample = 1000):
    
    cosi_star, cosi_star_err, cosi_disk, cosi_disk_err = data
    kappa_arr = np.linspace(kappa_min, kappa_max, N_kappa_sample)
    kappa_tile = np.tile(kappa_arr,(cosi_sample,1))
    prob_all = np.ones(len(kappa_arr))
    
    for i in range(len(cosi_star)):
            prob = 0
            cos_psi =  histogram_cos_psi_arr(cosi_star[i], cosi_star_err[i], cosi_disk[i], cosi_disk_err[i],cosi_sample)
            prob = kappa_von_mises_2d(kappa_tile, cos_psi)
            prob_all *= prob
    return prob_all, kappa_arr
        
def flag_cos(cos):
    
    flag_1 = cos>1-1e-8
    #cos[flag_1] = 2 - cos[flag_1] - 2e-8
    cos[flag_1] = 1-1e-8
    flag_2 = cos<-1+1e-8
    #cos[flag_2] =  -2 - cos[flag_2]  + 2e-8
    cos[flag_2] = -1 + 1e-8

    return cos
    
def random_mins_plus(num_vec):
    random_value = np.random.randn(num_vec)
    mins_flag = np.ones(num_vec)
    mins_flag[random_value <0] = -1
    return mins_flag
    
def gaussian_cosi(cos_star, cos_star_err, N):
    cos = np.random.normal(cos_star, cos_star_err, N)
    cos = flag_cos(cos)
    cos*= random_mins_plus(len(cos))
    sin_s = np.sqrt(1-cos**2)
    return cos, sin_s

def histogram_cos_psi(cos_star, cos_star_err, cos_disk, cos_disk_err, num_sample = 1000):
    cos_psi_sample =  cos_star + np.random.randn(len(cos_star)) *cos_star_err
    cos_psi_sample_star = flag_cos(cos_psi_sample)    
    cos_psi_sample_star *= random_mins_plus(len(cos_psi_sample))
    sin_psi_sample_star  =  np.sqrt(1- cos_psi_sample_star**2)
    cos_psi_sample =  cos_disk + np.random.randn(len(cos_star)) *cos_disk_err
    cos_psi_sample_disk = flag_cos(cos_psi_sample)    
    cos_psi_sample_disk *= random_mins_plus(len(cos_psi_sample))
    sin_psi_sample_disk =  np.sqrt(1- cos_psi_sample_disk**2)  
    PA_arr = np.random.rand(len(cos_star)) * np.pi*2 - np.pi 
    cos_psi = sin_psi_sample_star * sin_psi_sample_disk * np.cos(PA_arr) + cos_psi_sample_star * cos_psi_sample_disk
    return cos_psi

def histogram_cos_psi_arr(cos_star, cos_star_err, cos_disk, cos_disk_err, num_sample):
    cosstar_arr, sinstar_arr = gaussian_cosi(cos_star, cos_star_err, num_sample)
    cosdisk_arr, sindisk_arr = gaussian_cosi(cos_disk, cos_disk_err, num_sample)
    PA_arr = np.random.rand(num_sample) * np.pi * 2 - np.pi
    cos_psi = sinstar_arr * sindisk_arr * np.cos(PA_arr) + cosstar_arr * cosdisk_arr
    return cos_psi

def bin_make(binmin, binmax, num):
    bins = np.linspace(binmin, binmax, 100)
    bin_centers = 0.5 * (binmax - binmin)/(num-1.0) + bins[:len(bins)-1]
    return bins, bin_centers
    
def histogram_cos_psi_arr_prob(cos_star, cos_star_err, cos_disk, cos_disk_err, num_sample=100000):
    cosstar_arr, sinstar_arr = gaussian_cosi(cos_star, cos_star_err, num_sample)
    cosdisk_arr, sindisk_arr = gaussian_cosi(cos_disk, cos_disk_err, num_sample)
    PA_arr = np.random.rand(num_sample) * np.pi
    cos_psi = sinstar_arr * sindisk_arr * np.cos(PA_arr) + cosstar_arr * cosdisk_arr
    bins, bin_centers = bin_make(-1, 1,100)    
    hist = np.histogram(cos_psi, bins=bins)[0]
    return bin_centers, hist/np.sum(hist)

def prob_improved(kappa_arr, prob, n_kappa_sample, data):
    cosi_star, cosi_star_err, cosi_disk,cosi_disk_err = data
    psi_arr, psi_prob_data= histogram_cos_psi_arr_prob(cosi_star, cosi_star_err, cosi_disk,cosi_disk_err, 100000)   
    
    prob_all = np.zeros(len(psi_prob_data))
    
    for i in range(n_kappa_sample):
        kappa_now = np.random.choice(kappa_arr, size=None, replace=True, p=prob/np.sum(prob)) 
        prob_now = psi_prob_data * kappa_von_mises_from_kappa(kappa_now, psi_arr)   
        prob_all += prob_now/np.sum(prob_now)
    return psi_arr, prob_all, psi_prob_data
