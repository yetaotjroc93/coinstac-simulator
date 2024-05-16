import json
import os
import numpy as np
from math import exp, sqrt
from scipy.special import erf


def get_local_average_and_count(data_dir_path: str):
    data_file_filepath = os.path.join(data_dir_path, "data.json")

    print(f"\nloading data from: {data_file_filepath} \n")

    with open(data_file_filepath, "r") as file:
        data = json.load(file)
        
    fc_vecs = data["fc_vecs"]
    num_subs = len(fc_vecs)
    # differentially private parameters
    epsilon = data["epsilon"]
    delta = data["delta"]
    # pre-processing: clipping bound (default value is 1)
    if "pre_processing_bound" in data:
        bound = min(abs(data["pre_processing_bound"]), 1)
    else:
        bound = 1
    # post-processing: SVD components (default value is num_nodes)
    if "post_processing_components" in data:
        post_para_svd = data["post_processing_components"]
    else:
        post_para_svd = int(np.roots([1, -1, -2*len(fc_vecs[0])])[0])
        
    fc_vecs_clip = data_clip(fc_vecs, bound)
    fc_mean_mat_dp = dp_fc_mean(fc_vecs_clip, bound, epsilon, delta)
    fc_mean_mat_dp_svd = postprocess(fc_mean_mat_dp, bound, post_para_svd).tolist()

    return {"dp_fc_mean": fc_mean_mat_dp_svd, "count": num_subs, "epsilon": epsilon, "delta": delta, 
            "pre_processing_bound": bound, "post_processing_components": post_para_svd}


def data_clip(fc_vecs, bound):
    
    """Pre-processing"""

    # obtain clipped fc
    fc_vecs_clip = [np.clip(fc_vec, -bound, bound) for fc_vec in fc_vecs]

    return fc_vecs_clip    


def dp_fc_mean(fc_vecs, bound, epsilon, delta):

    num_subs, num_nodes, num_edges = para_from_vecs(fc_vecs)

    # sensitivity (add/remove) of whole matrix
    sensitivity = np.sqrt(bound**2*num_edges)/num_subs
    # use analytic Gaussian mechanism to compute noise variance
    sigma2 = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2

    # add noise to fc matrix
    fc_mean_vec = np.mean(np.array(fc_vecs), axis=0)
    fc_mean_mat_dp = noisy_mat(fc_mean_vec, bound, sigma2)

    return fc_mean_mat_dp


def para_from_vecs(fc_vecs):

    num_subs = len(fc_vecs)
    num_edges = len(fc_vecs[0])
    num_nodes = int(np.roots([1, -1, -2*num_edges])[0])

    return num_subs, num_nodes, num_edges


def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol=1.e-12):

    """
    Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Input:
      epsilon: target epsilon (epsilon > 0)
      delta: target delta (0 < delta < 1)
      GS: upper bound on L2 global sensitivity (GS >= 0)
      tol: error tolerance for binary search (tol > 0)

    Output:
      sigma: standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma


def noisy_mat(fc_mean_vec, bound, sigma2):

    fc_mean_vec_dp = np.clip(fc_mean_vec + np.random.normal(0, np.sqrt(sigma2), len(fc_mean_vec)), -bound, bound)
    fc_mean_mat_dp = vec_to_mat(fc_mean_vec_dp)

    return fc_mean_mat_dp


def vec_to_mat(fc_vec):

    num_nodes = int(np.roots([1, -1, -2*len(fc_vec)])[0])
    ind = np.triu_indices(num_nodes, 1)
    fc_mat = np.zeros((num_nodes, num_nodes))
    fc_mat[ind] = fc_vec
    fc_mat = fc_mat + fc_mat.T + np.diag([1]*num_nodes)

    return fc_mat


def postprocess(fc_mean_mat_dp, bound, post_para_svd):

    # SVD
    U, s, Vt = np.linalg.svd(fc_mean_mat_dp)
    s[post_para_svd:] = 0
    fc_mean_mat_dp_lowrank = U @ np.diag(s) @ Vt

    fc_mean_mat_dp_svd = np.clip(fc_mean_mat_dp_lowrank, -bound, bound)
    fc_mean_mat_dp_svd[range(len(fc_mean_mat_dp_svd)), range(len(fc_mean_mat_dp_svd))] = 1

    return fc_mean_mat_dp_svd
