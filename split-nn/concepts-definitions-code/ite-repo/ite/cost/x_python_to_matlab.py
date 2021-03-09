""" Python ITE <-> Matlab ITE transitions (where it exists).

Here we define dictionaries (Python -> Matlab), and their inversions. The
inversion means a 'key<->value' change, in other words given a cost type
(A/C/D/H/I/K/condH/condI) it provides the Matlab -> Python transitions.

"""


def inverted_dict(dict1):
    """" Performs key <-> value inversion in the dictionary

    Parameters
    ----------
    dict1 : dict

    Returns
    -------
    dict2 : dict
            Dictionary with inverted key-values.

    Examples
    --------
    dict1 = dict(a=1,b=2,c=3,d=4)
    inverted_dict(dict1) # result in {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
                           (up to possible permutation of the elements)

    """

    dict2 = {v: k for k, v in dict1.items()}

    return dict2


def merge_dicts(dict1, dict2):
    """ Merge two dictionaries.

    Parameters
    ----------
    dict1, dict2 : dict

    Returns
    -------
    dict_merged : dict
                  Merged dictionaries.

    """

    dict_merged = dict1.copy()
    dict_merged.update(dict2)

    return dict_merged

# #################
# Python -> Matlab:
# #################

# unconditional quantities:
dict_base_a_PythonToMatlab = dict(BASpearman1="Spearman1",
                                  BASpearman2="Spearman2",
                                  BASpearman3="Spearman3",
                                  BASpearman4="Spearman4",
                                  BASpearmanCondLT="Spearman_lt",
                                  BASpearmanCondUT="Spearman_ut",
                                  BABlomqvist="Blomqvist")

dict_meta_a_PythonToMatlab = dict(MASpearmanLT="Spearman_L",
                                  MASpearmanUT="Spearman_U")

dict_base_c_PythonToMatlab = dict(BCCE_KnnK="CE_kNN_k")
dict_meta_c_PythonToMatlab = dict()

dict_base_d_PythonToMatlab \
    = dict(BDKL_KnnK="KL_kNN_k",
           BDEnergyDist="EnergyDist",
           BDBhattacharyya_KnnK="Bhattacharyya_kNN_k",
           BDBregman_KnnK="Bregman_kNN_k",
           BDChi2_KnnK="ChiSquare_kNN_k",
           BDHellinger_KnnK="Hellinger_kNN_k",
           BDKL_KnnKiTi="KL_kNN_kiTi",
           BDL2_KnnK="L2_kNN_k",
           BDRenyi_KnnK="Renyi_kNN_k",
           BDTsallis_KnnK="Tsallis_kNN_k",
           BDSharmaMittal_KnnK="SharmaM_kNN_k",
           BDSymBregman_KnnK="symBregman_kNN_k",
           BDMMD_UStat="MMD_Ustat",
           BDMMD_VStat="MMD_Vstat",
           BDMMD_Online="MMD_online",
           BDMMD_UStat_IChol="MMD_Ustat_iChol",
           BDMMD_VStat_IChol="MMD_Vstat_iChol")

dict_meta_d_PythonToMatlab = dict(MDBlockMMD="BMMD_DMMD_Ustat",
                                  MDEnergyDist_DMMD="EnergyDist_DMMD",
                                  MDf_DChi2="f_DChiSquare",
                                  MDJDist_DKL="Jdistance",
                                  MDJR_HR="JensenRenyi_HRenyi",
                                  MDJT_HT="JensenTsallis_HTsallis",
                                  MDJS_HS="JensenShannon_HShannon",
                                  MDK_DKL="K_DKL",
                                  MDL_DKL="L_DKL",
                                  MDSymBregman_DB="symBregman_DBregman",
                                  MDKL_HSCE="KL_CCE_HShannon")

dict_base_h_PythonToMatlab = dict(BHShannon_KnnK="Shannon_kNN_k",
                                  BHShannon_SpacingV="Shannon_spacing_V",
                                  BHRenyi_KnnK="Renyi_kNN_k",
                                  BHTsallis_KnnK="Tsallis_kNN_k",
                                  BHSharmaMittal_KnnK="SharmaM_kNN_k",
                                  BHShannon_MaxEnt1="Shannon_MaxEnt1",
                                  BHShannon_MaxEnt2="Shannon_MaxEnt2",
                                  BHPhi_Spacing="Phi_spacing",
                                  BHRenyi_KnnS="Renyi_kNN_S")

dict_meta_h_PythonToMatlab = dict(MHShannon_DKLN="Shannon_DKL_N",
                                  MHShannon_DKLU="Shannon_DKL_U",
                                  MHTsallis_HR="Tsallis_HRenyi")

dict_base_i_PythonToMatlab = dict(BIDistCov="dCov",
                                  BIDistCorr="dCor",
                                  BI3WayJoint="3way_joint",
                                  BI3WayLancaster="3way_Lancaster",
                                  BIHSIC_IChol="HSIC",
                                  BIHoeffding="Hoeffding",
                                  BIKGV="KGV",
                                  BIKCCA="KCCA")

dict_meta_i_PythonToMatlab = dict(MIShannon_DKL="Shannon_DKL",
                                  MIChi2_DChi2="ChiSquare_DChiSquare",
                                  MIL2_DL2="L2_DL2",
                                  MIRenyi_DR="Renyi_DRenyi",
                                  MITsallis_DT="Tsallis_DTsallis",
                                  MIMMD_CopulaDMMD="MMD_DMMD",
                                  MIRenyi_HR="Renyi_HRenyi",
                                  MIShannon_HS="Shannon_HShannon",
                                  MIDistCov_HSIC="dCov_IHSIC")

dict_base_k_PythonToMatlab = dict(BKProbProd_KnnK="PP_kNN_k",
                                  BKExpected="expected")

dict_meta_k_PythonToMatlab = dict(MKExpJR1_HR="EJR1_HR",
                                  MKExpJR2_DJR="EJR2_DJR",
                                  MKExpJS_DJS="EJS_DJS",
                                  MKExpJT1_HT="EJT1_HT",
                                  MKExpJT2_DJT="EJT2_DJT",
                                  MKJS_DJS="JS_DJS",
                                  MKJT_HT="JT_HJT")

# conditional quantities:
dict_base_h_cond_PythonToMatlab = dict()
dict_meta_h_cond_PythonToMatlab = \
    dict(BcondHShannon_HShannon="Shannon_HShannon")

dict_base_i_cond_PythonToMatlab = dict()
dict_meta_i_cond_PythonToMatlab = \
    dict(BcondIShannon_HShannon="Shannon_HShannon")

# ##################################################
# merge the dictionaries of 'base' and 'meta' names:
# ##################################################

# unconditional quantities:
dict_A_PythonToMatlab = merge_dicts(dict_base_a_PythonToMatlab,
                                    dict_meta_a_PythonToMatlab)
dict_C_PythonToMatlab = merge_dicts(dict_base_c_PythonToMatlab,
                                    dict_meta_c_PythonToMatlab)
dict_D_PythonToMatlab = merge_dicts(dict_base_d_PythonToMatlab,
                                    dict_meta_d_PythonToMatlab)
dict_H_PythonToMatlab = merge_dicts(dict_base_h_PythonToMatlab,
                                    dict_meta_h_PythonToMatlab)
dict_I_PythonToMatlab = merge_dicts(dict_base_i_PythonToMatlab,
                                    dict_meta_i_PythonToMatlab)
dict_K_PythonToMatlab = merge_dicts(dict_base_k_PythonToMatlab,
                                    dict_meta_k_PythonToMatlab)

# conditional ones:
dict_H_Cond_PythonToMatlab = merge_dicts(dict_base_h_cond_PythonToMatlab,
                                         dict_meta_h_cond_PythonToMatlab)
dict_I_Cond_PythonToMatlab = merge_dicts(dict_base_i_cond_PythonToMatlab,
                                         dict_meta_i_cond_PythonToMatlab)

# ##############################################
# Matlab -> Python by inverted the dictionaries:
# ##############################################

# unconditional quantities:
dict_A_MatlabToPython = inverted_dict(dict_A_PythonToMatlab)
dict_C_MatlabToPython = inverted_dict(dict_C_PythonToMatlab)
dict_D_MatlabToPython = inverted_dict(dict_D_PythonToMatlab)
dict_H_MatlabToPython = inverted_dict(dict_H_PythonToMatlab)
dict_I_MatlabToPython = inverted_dict(dict_I_PythonToMatlab)
dict_K_MatlabToPython = inverted_dict(dict_K_PythonToMatlab)

# conditional quantities:
dict_H_Cond_MatlabToPython = inverted_dict(dict_H_Cond_PythonToMatlab)
dict_I_Cond_MatlabToPython = inverted_dict(dict_I_Cond_PythonToMatlab)


# Examples
# --------
# Python -> Matlab:
# >>> dict_A_PythonToMatlab['BASpearman1']
#    => 'Spearman1' is the Matlab name of 'BASpearman1'
#
# Matlab -> Python, given a cost type (A):
# >>>dict_A_MatlabToPython['Spearman1']
#    => 'BASpearman1' is the Python name of the 'Spearman1' association
