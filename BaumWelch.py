import numpy as np
from numpy.linalg import matrix_power
from numpy import linalg as LA
from numba import njit, jit
import pdb
import sys
import psutil
import os
import pickle
from transition_matrix import *
from utils import *
from scipy.optimize import minimize, minimize_scalar, LinearConstraint
from joblib import Parallel, delayed
from scipy import linalg



class BaumWelch:
    def __init__(self,sequences_info,D,E,E_masked,lambda_A_values,lambda_B_values,gamma_fg,lambda_A_segs,lambda_B_segs,rho,theta,T_S,T_E,T_array,bin_size,j_max,cores,xtol=0.0001,ftol=0.0001,spread_1=0.1,spread_2=50,lambda_lwr_bnd=0.1,lambda_upr_bnd=10,gamma_lwr_bnd=0,gamma_upr_bnd=1,estimate_rho=True,output_path = None,verbosity=False,midpoint_transitions=False,optimisation_method="Powell",save_iteration_files=False,final_T_factor=None,midpoint_end=False,lambda_lwr_bnd_struct = 0.5, lambda_upr_bnd_struct = 2, recombnoexp = False):
        
        self.sequences_info = sequences_info
        self.num_files = len(sequences_info)
        self.D = D # number of states
        self.E = E # emission probabilities (taken as fixed)
        self.E_masked = E_masked
        self.rho = rho
        self.theta = theta
        self.estimate_rho = estimate_rho
        self.verbosity = verbosity 
        self.spread_1 = spread_1
        self.spread_2 = spread_2
        if self.estimate_rho:
            print('\tWill estimate rho.')
        else:
            print("\tNot estimating rho")

        self.T_array = T_array
        self.bin_size = bin_size
        self.xtol = xtol
        self.ftol = ftol
        self.j_max = j_max # the maximum number of mutations seen in the sequence
        self.T_S_index = T_S # fixed guess for T_S
        self.T_E_index = T_E # fixed guess for 
        self.LLs = []
        self.number_of_completed_iterations = 0
        self.midpoint_transitions = midpoint_transitions
        self.optimisation_method = optimisation_method
        self.save_iteration_files = save_iteration_files
        self.final_T_factor = final_T_factor
        self.midpoint_end = midpoint_end
        self.midpoints = np.array([(self.T_array[i]+self.T_array[i+1])/2 for i in range(0,len(self.T_array)-1)])
        self.cores = cores
        self.recombnoexp = recombnoexp
        if self.midpoint_end is False:
            self.midpoints[-1] = (self.T_array[-2]+self.T_array[-2]+3)/2

        # lambda_A_parameters initialisation
        self.lambda_A_current = parse_lambda_input(lambda_A_values,self.D,lambda_A_segs)
        self.lambda_A_segs, self.lambda_A_values, self.num_lambda_A_params = initialisation_lambda(lambda_A_values,lambda_A_segs,self.D)

        if T_E is not None: # check valid
            print('\tProblem! T_S is None, but T_E is not None. Aborting!')
            sys.exit()

        tm_first_guess = Transition_Matrix(D=self.D,spread_1=self.spread_1,spread_2=self.spread_2,midpoint_transitions=self.midpoint_transitions) # initialise transition matrix object
        self.Q_first_guess = tm_first_guess.write_tm(lambda_A=self.lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=self.rho,exponential=not self.recombnoexp) # initialise transition matrix object

        self.init_dist = get_stationary_distribution_theory(self.Q_first_guess)
        del tm_first_guess

        self.Q_new = np.zeros(shape=(self.D,self.D))
        self.output_path = output_path
        lambda_lwr_bnd = lambda_lwr_bnd
        lambda_upr_bnd = lambda_upr_bnd
        lambda_boundaries = (lambda_lwr_bnd,lambda_upr_bnd)
        self.gamma_lwr_bnd = gamma_lwr_bnd
        self.gamma_upr_bnd = gamma_upr_bnd
        self.lambda_bnds = ((lambda_boundaries,) *(self.num_lambda_A_params-1)) + ((lambda_lwr_bnd,10),)


    # function to generate sequence from succint sequence_memsaver (sequences_info[i][0])
    # input: which file to use (integer, for label)
    #  output: the observation(s) at position at given index(indices)
    # e.g. sequence(0) returns sequence[0], sequence(0,10) returns sequence[0:10]
    def sequence_fcn(self,file):
        
        het_sequence = np.zeros(int(self.sequences_info[file][3]/self.bin_size),dtype=int)
        mask_sequence = np.zeros(int(self.sequences_info[file][3]/self.bin_size),dtype=int)
        het_sequence[self.sequences_info[file][0][0]] = self.sequences_info[file][0][1]
        j_max = self.sequences_info[0][2]
        if self.bin_size <= j_max:
            print(f'bin_size={self.bin_size} < j_max ={j_max}')
            mask_sequence[self.sequences_info[file][1][0]] = self.sequences_info[file][1][1]*(j_max+1)
        else:
            mask_sequence[self.sequences_info[file][1][0]] = self.sequences_info[file][1][1]*self.bin_size
        sequence = het_sequence + mask_sequence
        B_sequence = self.sequences_info[file][6]
        B_vals = self.sequences_info[file][7]
        R_sequence = self.sequences_info[file][8]
        R_vals = self.sequences_info[file][9]
        # load Emissions here?
        if B_vals[0]==0:
            B_vals[0]=(B_vals[0]+B_vals[1])/2

        return sequence, B_sequence, B_vals, R_sequence, R_vals

    def save_output_files(self,iteration):
        LL_str = f'log likelihood for this iteration = {self.LLs[-1]}'
        if len(self.LLs)<2:
            change_LL_str = ''
        else:
            change_LL_str = f'change in log likelihood for this iteration = {self.LLs[-1] - self.LLs[-2]}'
        iterations_str = f'number of iterations taken = {self.number_of_completed_iterations}'
        theta_str = f'theta=4*N_E*mu = {self.theta}'
        rho_str = f'rho=4*N_E*r = {self.rho}'

        scaled_time = 0.5*self.T_array*(self.theta) # scale this to gens by dividing by mu. time_gens = scaled_time / mu
        scaled_inverse_lambda =  (4*self.lambda_A_current)/self.theta # scale this to inverse pop sizes with N_E = (1/scaled_inverse_lambda)/mu 
        ltb = scaled_time[0:self.D]
        rtb = scaled_time[1:self.D+1]

        info_strings = [LL_str,change_LL_str,iterations_str,theta_str,rho_str]
        scaletime_str = 'scale time by dividing by mu'
        scalelambda_str = 'scale lambda by taking its inverse then dividing by mu'
        info_strings.append(scaletime_str)
        info_strings.append(scalelambda_str)

        output_filename_params = self.output_path + 'params_iteration' + str(iteration) + '.txt'
        if self.T_S_index is None: # panmixia
            final_array = np.zeros(shape=(self.D,3))
            final_array[:,0] = ltb
            final_array[:,1] = rtb 
            final_array[:,2] = scaled_inverse_lambda 
            columns_str = 'col 0 is left time boundary; col 1 is right time boundary; col 3 is scaled_lambda_A'
            info_strings.append(columns_str)
            footer = ''
            header = "\n".join(info_strings)

            np.savetxt(output_filename_params,final_array, comments='# ', header=header,footer=footer)
        else: # structure
            final_array = np.zeros(shape=(self.D,4))
            final_array[:,0] = ltb
            final_array[:,1] = rtb 
            final_array[:,2] = scaled_inverse_lambda 
            scaled_inverse_lambda_B =  (4*self.lambda_B_current)/self.theta # scale this to inverse pop sizes with N_E = (1/scaled_inverse_lambda)/mu 
            final_array[:,3] = scaled_inverse_lambda_B 
            columns_str = 'col 0 is left time boundary; col 1 is right time boundary; col 3 is scaled_lambda_A; col 4 is scaled_lambda_B'
            footer = f'gamma is {self.gamma_current}'
            info_strings.append(columns_str)
            header = "\n".join(info_strings)
            np.savetxt(output_filename_params,final_array,comments='# ',header=header,footer=footer)
        print(f'\n\t\toutput saved to {output_filename_params}')
        return None


    def get_num_masks_hets(self,x):
        num_masks = abs(int(x/self.bin_size))
        num_hets = abs(x) - num_masks*self.bin_size
        return num_masks, num_hets

    def neg_objective_function_pan(self,params,A_evidence,tm_dummy,rho):
        # minimise with panmixia
        # no estimation of rho
        lambda_A = write_lambda_optimise(self.lambda_A_segs,params,self.lambda_A_current)
        tm_dummy.write_tm(lambda_A=lambda_A,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=self.rho,exponential=not self.recombnoexp) # initialise transition matrix object        
        if np.min(tm_dummy.Q)<0 or np.max(tm_dummy.Q)>1: #  or check_E>0: # this rho value returned an invalid result in matrix 
            F_obj = -np.inf
        else:
            log_Q = np.log(tm_dummy.Q)
            F_trans = np.sum(np.multiply(log_Q,A_evidence)) 
            F_obj = F_trans
        return F_obj*(-1)



    def neg_objective_function_pan_varR(self,params,A_evidence,tm_dummy,rho,Rvals_freqs,R_vals):
        # minimise with panmixia, with varying mut rate
        # no estimation of rho
        lambda_A = write_lambda_optimise(self.lambda_A_segs,params,self.lambda_A_current)
        tm_dummy.write_tm(lambda_A=lambda_A,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=self.rho,exponential=not self.recombnoexp) # initialise transition matrix object        
        zQ = write_Q_array_withR(tm_dummy.Q,R_vals,rho,self.D,self.spread_1,self.spread_2,lambda_A,self.midpoint_transitions)       
        zQ = tm_dummy.write_tm(lambda_A=lambda_A,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=self.rho,exponential=not self.recombnoexp) # initialise transition matrix object        
        if np.min(tm_dummy.Q)<0 or np.max(tm_dummy.Q)>1: #  or check_E>0: # this rho value returned an invalid result in matrix 
            F_obj = -np.inf
        else:
            log_Q = np.log(zQ)
            F_trans = np.sum(np.multiply(log_Q,A_evidence)) 
            F_obj = F_trans
        return F_obj*(-1)



    def neg_objective_function_pan_rho(self,params,A_evidence,tm_dummy):
        # minimise with panmixia
        # estimate rho too
        lambda_A = write_lambda_optimise(self.lambda_A_segs,params[0:-1],self.lambda_A_current)
        rho = params[-1]
        tm_dummy.write_tm(lambda_A=lambda_A,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=rho,exponential=not self.recombnoexp) # initialise transition matrix object
        if np.min(tm_dummy.Q)<0 or np.max(tm_dummy.Q)>1: #  or check_E>0: # this rho value returned an invalid result in matrix 
            F_obj = -np.inf
        else:
            log_Q = np.log(tm_dummy.Q)
            F_trans = np.sum(np.multiply(log_Q,A_evidence)) 
            F_obj = F_trans 
        return F_obj*(-1)
   
    def Powell_parameter_search(self,A_new,iteration):
        print(f'\t\tOptimising for new demographic parameters...',flush=True) 
        tm_dummy = Transition_Matrix(D=self.D,spread_1=self.spread_1,spread_2=self.spread_2,midpoint_transitions=self.midpoint_transitions) # initialise transition matrix object
        if self.estimate_rho is False: # panmixia and no rho
            params_initial_guess = self.lambda_A_values
            zmax = A_new.max()
            start = time.time()
            res = minimize(self.neg_objective_function_pan,params_initial_guess,args=(A_new,tm_dummy,self.rho),method=self.optimisation_method,bounds=self.lambda_bnds,options={'xtol': self.xtol,'ftol': self.ftol}) #flag220818
            end = time.time()
            time_taken = end - start
            print(f'\t\t\ttime taken to optimise: {time_taken}',flush=True)
            optimised_params = res.x
        elif self.estimate_rho is True: # panmixia and rho
            params_initial_guess = np.append(self.lambda_A_values,np.array([self.rho]))
            # TODO update lower and upper bound of rho
            rho_lwr_bnd = 1e-09*2*2*5000*self.bin_size # arbitrarily choose lowest p=1e-09 and N=5000 (p is per gen per bp recomb rate). Scale by bin size # TODO CHOOSE BETTER BOUNDS
            rho_upr_bnd = 2e-08*2*2*20000*self.bin_size # arbitrarily choose greatest p=2e-08 and N = 20000 
            rho_boundaries = ((rho_lwr_bnd,rho_upr_bnd),)
            bnds = (self.lambda_bnds + rho_boundaries )
            start = time.time()
            res = minimize(self.neg_objective_function_pan_rho,params_initial_guess,args=(A_new,tm_dummy),method="Powell",bounds=bnds) 
            end = time.time()
            time_taken = end - start
            print(f'\t\t\ttime taken to optimise: {time_taken}',flush=True)
            self.rho=res.x[-1]              
            res.x = res.x[0:-1] # remove rho from this array
            optimised_params = res.x
    


        tm_current = Transition_Matrix(D=self.D,spread_1=self.spread_1,spread_2=self.spread_2,midpoint_transitions=self.midpoint_transitions) # initialise transition matrix object

        # write matrix for next iteration
        # write new panmictic matrix
        self.lambda_A_values = optimised_params
        self.lambda_A_current = write_lambda_optimise(self.lambda_A_segs,self.lambda_A_values,self.lambda_A_current)
        self.Q_current = tm_current.write_tm(lambda_A=self.lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=self.rho,exponential=not self.recombnoexp) # initialise transition matrix object
        del tm_current
        # print(f'\tHave updated Q and current parameters')

        return None

    def Powell_parameter_search_Rmap(self,A_new,iteration):
    # def Powell_parameter_search(self,A_new,iteration,weightedrarray,R_vals):
        Rvals = self.sequence_fcn(0)[4]
        Rvals_counts = np.zeros(len(Rvals))
        for i in range(0,self.num_files):
            abba,baba = np.unique(self.sequence_fcn(i)[3],return_counts=True)
            Rvals_counts[abba]+=baba
        Rvals_freqs = Rvals_counts/Rvals_counts.sum()
        print(f'\t\tOptimising for new demographic parameters...',flush=True) 
        tm_dummy = Transition_Matrix(D=self.D,spread_1=self.spread_1,spread_2=self.spread_2,midpoint_transitions=self.midpoint_transitions) # initialise transition matrix object
        if self.estimate_rho is False: # panmixia and no rho
            params_initial_guess = self.lambda_A_values
            # zA_new = np.zeros(shape=A_new.shape)
            # for qq in range(0,zA_new.shape[0]):
            #     zA_new[qq,:,:] = A_new[qq,:,:]*(zmax/A_new[qq,:,:].max()) # make each A the same size
            start = time.time()
            # res = minimize(self.neg_objective_function_pan,params_initial_guess,args=(A_new,tm_dummy,R_vals,self.rho),method=self.optimisation_method,bounds=self.lambda_bnds,options={'xtol': self.xtol,'ftol': self.ftol}) #flag220818
            res = minimize(self.neg_objective_function_pan_varR,params_initial_guess,args=(A_new,tm_dummy,self.rho,Rvals_freqs,Rvals),method=self.optimisation_method,bounds=self.lambda_bnds,options={'xtol': self.xtol,'ftol': self.ftol}) #flag220818
            end = time.time()
            time_taken = end - start
            print(f'\t\t\ttime taken to optimise: {time_taken}',flush=True)
            optimised_params = res.x
        else:
            print(f'Rho must be fixed for Rmap=True')
            sys.exit() 

        tm_current = Transition_Matrix(D=self.D,spread_1=self.spread_1,spread_2=self.spread_2,midpoint_transitions=self.midpoint_transitions) # initialise transition matrix object
        # self.params_current = res.x # update parameters
        # self.params_current = optimised_params

        # write matrix for next iteration
            # write new panmictic matrix
        self.lambda_A_values = optimised_params
        self.lambda_A_current = write_lambda_optimise(self.lambda_A_segs,self.lambda_A_values,self.lambda_A_current)
        self.Q_current = tm_current.write_tm(lambda_A=self.lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=self.rho,exponential=not self.recombnoexp) # initialise transition matrix object
    
        del tm_current
        return None

    het_mask_convert = lambda self, a: -a % self.bin_size if a<0 else a 

    def BaumWelch(self,BW_iterations=20,BW_thresh=None,verbosity=False):
        iteration = 0
        old_ll = 0
        A_guess = np.zeros(shape=(self.D,self.D))      
        change_ll = 1000
        
        while iteration < BW_iterations and change_ll>BW_thresh:
            print(f'\n\tOn iteration {iteration} ',flush=True)

            A_new = np.zeros(shape=(self.D,self.D))
            E_new = np.zeros(shape=(self.E.shape[0],self.E.shape[1]))
            E_new_with_scales = np.zeros(shape=(self.E.shape[0],self.E.shape[1]))
            print(f'\t\tCalculating expectation...')
            start = time.time()

            sequence, B_sequence, B_vals,R_sequence, R_vals = self.sequence_fcn(0)
            if max(R_vals)==min(R_vals):
                Rmap=False
            else:
                Rmap=True
            
            tm_dummy = Transition_Matrix(D=self.D,spread_1=self.spread_1,spread_2=self.spread_2,midpoint_transitions=self.midpoint_transitions) # initialise transition matrix object
            tm_dummy.write_tm(lambda_A=self.lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=self.rho,exponential=not self.recombnoexp) # write transition matrix for different rho values
            # zQ_current_array = write_Q_array_withR_old(tm_dummy.Q,R_vals,R_vals[np.argmin(np.abs(R_vals-1))],self.D)
            Q_current_array = write_Q_array_withR(tm_dummy.Q,R_vals,self.rho,self.D,self.spread_1,self.spread_2,self.lambda_A_current,self.midpoint_transitions) # TODO use this!!!
            
            expectation_steps = Parallel(n_jobs=self.num_files, backend='loky')(delayed(calculate_transition_evidence)(self.sequence_fcn,file,self.D,self.init_dist,self.E_masked,Q_current_array,self.theta,self.rho,self.bin_size,self.j_max,self.midpoints,self.spread_1,self.spread_2,self.midpoint_transitions) for file in range(self.num_files))
            # expectation_steps = calculate_transition_evidence(self.sequence_fcn,0,self.D,self.init_dist,self.E_masked,self.Q_current,self.ram_limit,self.theta,self.bin_size,self.j_max,self.midpoints) 

            # expectation_steps = Parallel(n_jobs=self.num_files, backend='loky')(delayed(self.calculate_transition_evidence_copy)(file) for file in range(self.num_files))
            end = time.time()
            time_taken = end - start
            print(f'\t\t\ttime taken to calculate expectation: {time_taken}')
            A_evidence = np.zeros(shape=(self.D,self.D))
            new_ll = 0
            for i in range(self.num_files):
                A_evidence += expectation_steps[i][0]
                new_ll += expectation_steps[i][1]
            if iteration==0:
                change_ll = -new_ll
            else:
                change_ll = new_ll - old_ll
            print('\t\tlog likelihood is {}'.format( new_ll ),flush=True)
            print('\t\tchange in log likelihood is {}'.format(change_ll))
            self.LLs.append(new_ll)
            self.number_of_completed_iterations += 1
            old_ll = new_ll

            # this function searches for most likely parameters, then saves them to self (updates Q)
            if Rmap is False:
                self.Powell_parameter_search(A_evidence,iteration)
            if Rmap is True:
                self.Powell_parameter_search_Rmap(A_evidence,iteration)

            # self.Powell_parameter_search(zA_evidence,iteration,R_vals)
            # self.Powell_parameter_search(zA_evidence,iteration,weightedrarray,R_vals) # need weighted R vals

            # print(f'\tNew params are\n{self.params_current}',flush=True)
            # print(f'\t\tlambda_A updated: {self.lambda_A_current}',flush=True)
            print(f'\t\tlambda_A updated: [{",".join([str(i) for i in self.lambda_A_current])}]',flush=True)
            if self.T_S_index is not None:
                print(f'\t\tlambda_B updated: [{",".join([str(i) for i in self.lambda_B_current])}]',flush=True)
                print(f'\t\tgamma updated: [{self.gamma_current}]',flush=True)
            print(f'\t\trho = {self.rho/self.bin_size}',flush=True)
            print(f'\t\ttheta = {self.theta}',flush=True)
            # self.E = E_new / E_new.sum(axis=1)[:, np.newaxis]

            iteration += 1

            if self.save_iteration_files: 
                self.save_output_files(iteration)
        return None


def get_stationary_distribution_theory(matrix):
    # given theoretical matrix, calculate stationary distribution of markov chain
    eigvals, eigvecs = linalg.eig(matrix, left=True, right=False)
    try:
        unit_eigval_index = np.where( abs(1-eigvals)==np.min(abs(1-eigvals)))[0][0]
    except:
        print(f'failed to find unit eigenvalue...aborting.')
    theory_stat_dist = eigvecs[:,unit_eigval_index]/eigvecs[:,unit_eigval_index].sum()
    return theory_stat_dist

def write_lambda(segments,values,lambda_array):
    # used for initialisation
    # segments: time segment pattern, list of tuples
    # values: value for each segment
    # D: number of intervals 
    index = 0
    k = 0
    for i in segments:
        j = 0 
        while j<i[0]:
            if i[1]!=0:
                lambda_array[index:index+i[1]] = values[k]
                index += i[1]
            else:
                # lambda_array[index] = values[k]
                lambda_array[index:index+i[0]] = values[k]
                index += i[0]
                j = i[0]+1
            k += 1
            j += 1
    return lambda_array

def write_lambda_optimise(segments,values,lambda_array):
    # used for optimisation, possibly leave values of lambda_array untouched if fixed indices used
    # segments: time segment pattern, list of tuples
    # values: value for each segment
    # D: number of intervals 
    index = 0
    k = 0
    for i in segments:
        j = 0 
        while j<i[0]:
            if i[1]!=0:
                lambda_array[index:index+i[1]] = values[k]
                index += i[1]
                k += 1
            else:
                index += i[0]
                j = i[0]+1
            j += 1
    return lambda_array


def parse_lambda_input(lambda_values,D,segments):
    # take lambda_values and segments, then write lambda_array
    num_diff_values = sum([i[0] for i in segments if i[1]!=0]) + sum([1 for i in segments if i[1]==0]) 
    if len(lambda_values)!=num_diff_values:
        print(f'\n***problem in parse_lambda_input! len(lambdas)={len(lambda_values)} which is not equal to the number of specified segments = {num_diff_values}. Aborting***')
        sys.exit()
    # if len(lambdas)!=D:
    #     print(f'***problem in parse_lambda_input! len(lambdas)={len(lambdas)} which is not equal to D={D}. Aborting***')
    #     sys.exit()
    dummy_array = np.ones(D)
    lambdas = write_lambda(segments,lambda_values,dummy_array)
    return lambdas

def parse_lambda_fg(lambda_str,segments):
    num_diff_values = sum([i[0] for i in segments if i[1]>0]) + sum([1 for i in segments if i[1]==0]) 
    if lambda_str is not None:
        lambda_string = lambda_str
        lambda_list=lambda_string.split(',')
        lambda_list=[float(i) for i in lambda_list]
        lambda_values = np.array(lambda_list)
        if len(lambda_values)!=num_diff_values:
            print(f'\n***problem in parse_lambda_input! len(lambdas)={len(lambda_values)} which is not equal to the number of specified segments = {num_diff_values}. Aborting***')
            sys.exit()
    # if len(lambdas)!=D:
    #     print(f'***problem in parse_lambda_input! len(lambdas)={len(lambdas)} which is not equal to D={D}. Aborting***')
    #     sys.exit()
    else:
        lambda_values = np.ones(num_diff_values)
    return lambda_values

def initialisation_lambda(values,segments,D):
    # segments: list of tuples, tuple elements are ints. This is time segment pattern
    # values: array of values, for each segment. The length of the array must be equal to the number of different segments
    # D: int. Number of time intervals
    lambda_segs = segments
    lambda_current = parse_lambda_input(values,D,segments)
    index_fixed_params = []
    for i in range(0,len(segments)):
        if segments[i][1]==0:
            # index_fixed_params.append(sum([segments[j][0]*segments[j][1] for j in range(0,i)]))
            index_fixed_params.append(sum([segments[j][0]*segments[j][1] for j in range(0,i) if segments[j][1]!=0 ]) + sum([1 for j in range(0,i) if segments[j][1]==0 ]))           
    values = values.tolist()
    values = np.array([values[j] for j in range(0,len(values)) if j not in index_fixed_params])
    num_lambda_params = len(values)
    return segments, values, num_lambda_params

