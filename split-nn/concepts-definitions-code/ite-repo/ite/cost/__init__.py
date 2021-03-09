""" Information theoretical estimators.

Estimators for entropy, mutual information, divergence, association
measures, cross quantities, kernels on distributions.

"""

# automatically load submodules:

# ###################
# unconditional ones:
# ###################

# base estimators:
from .base_a import BASpearman1, BASpearman2, BASpearman3, BASpearman4,\
                    BASpearmanCondLT, BASpearmanCondUT, BABlomqvist
from .base_c import BCCE_KnnK
from .base_d import BDKL_KnnK, BDEnergyDist, BDBhattacharyya_KnnK,\
                    BDBregman_KnnK, BDChi2_KnnK, BDHellinger_KnnK,\
                    BDKL_KnnKiTi, BDL2_KnnK, BDRenyi_KnnK, BDTsallis_KnnK,\
                    BDSharmaMittal_KnnK, BDSymBregman_KnnK, BDMMD_UStat,\
                    BDMMD_VStat, BDMMD_Online, BDMMD_UStat_IChol,\
                    BDMMD_VStat_IChol
from .base_h import BHShannon_KnnK, BHShannon_SpacingV, BHRenyi_KnnK,\
                    BHTsallis_KnnK, BHSharmaMittal_KnnK, BHShannon_MaxEnt1,\
                    BHShannon_MaxEnt2, BHPhi_Spacing, BHRenyi_KnnS
from .base_i import BIDistCov, BIDistCorr, BI3WayJoint, BI3WayLancaster,\
                    BIHSIC_IChol, BIHoeffding, BIKGV, BIKCCA
from .base_k import BKProbProd_KnnK, BKExpected

# meta estimators:
from .meta_a import MASpearmanLT, MASpearmanUT
from .meta_d import MDBlockMMD, MDEnergyDist_DMMD, MDf_DChi2,\
                    MDJDist_DKL, MDJR_HR, \
                    MDJT_HT, MDJS_HS, MDK_DKL,\
                    MDL_DKL, MDSymBregman_DB, MDKL_HSCE
from .meta_h import MHShannon_DKLN, MHShannon_DKLU, MHTsallis_HR
from .meta_i import MIShannon_DKL, MIChi2_DChi2, MIL2_DL2,\
                    MIRenyi_DR, MITsallis_DT, MIMMD_CopulaDMMD, \
                    MIRenyi_HR, MIShannon_HS, MIDistCov_HSIC
from .meta_k import MKExpJR1_HR, MKExpJR2_DJR, MKExpJS_DJS,\
                    MKExpJT1_HT, MKExpJT2_DJT, MKJS_DJS,\
                    MKJT_HT


# #################
# conditional ones:
# #################

from .meta_h_cond import BcondHShannon_HShannon
from .meta_i_cond import BcondIShannon_HShannon


from .x_factory import co_factory

# explicitly tilt "from X import *" type importing:
__all__ = []
