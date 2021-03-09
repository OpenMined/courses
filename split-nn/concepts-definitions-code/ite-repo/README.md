#Information Theoretical Estimators (ITE) in Python

It

* is the redesigned, Python implementation of the [Matlab/Octave ITE](https://bitbucket.org/szzoli/ite/) toolbox.
* can estimate numerous entropy, mutual information, divergence, association measures, cross quantities, and kernels on distributions.
* can be used to solve information theoretical optimization problems in a high-level way.
* comes with several demos.
* is free and open source: GNU GPLv3(>=).

Estimated quantities:

* `entropy (H)`: Shannon entropy, Rényi entropy, Tsallis entropy (Havrda and Charvát entropy), Sharma-Mittal entropy, Phi-entropy (f-entropy).
* `mutual information (I)`: Shannon mutual information (total correlation, multi-information), Rényi mutual information, Tsallis mutual information, chi-square mutual information (squared-loss mutual information, mean square contingency), L2 mutual information, copula-based kernel dependency, kernel canonical correlation analysis, kernel generalized variance, multivariate version of Hoeffding's Phi, Hilbert-Schmidt independence criterion, distance covariance, distance correlation, Lancaster three-variable interaction.
* `divergence (D)`: Kullback-Leibler divergence (relative entropy, I directed divergence), Rényi divergence, Tsallis divergence, Sharma-Mittal divergence, Pearson chi-square divergence (chi-square distance), Hellinger distance, L2 divergence, f-divergence (Csiszár-Morimoto divergence, Ali-Silvey distance), maximum mean discrepancy (kernel distance, current distance), energy distance (N-distance; specifically the Cramer-Von Mises distance), Bhattacharyya distance, non-symmetric Bregman distance (Bregman divergence), symmetric Bregman distance,  J-distance (symmetrised Kullback-Leibler divergence, J divergence), K divergence, L divergence, Jensen-Shannon divergence, Jensen-Rényi divergence, Jensen-Tsallis divergence.
* `association measures (A)`: multivariate extensions of Spearman's rho (Spearman's rank correlation coefficient, grade correlation coefficient), multivariate conditional version of Spearman's rho, lower and upper tail dependence via conditional Spearman's rho.
* `cross quantities (C)`: cross-entropy,
* `kernels on distributions (K)`: expected kernel (summation kernel, mean map kernel, set kernel, multi-instance kernel, ensemble kernel; specific convolution kernel), probability product kernel, Bhattacharyya kernel (Bhattacharyya coefficient, Hellinger affinity), Jensen-Shannon kernel, Jensen-Tsallis kernel, exponentiated Jensen-Shannon kernel, exponentiated Jensen-Rényi kernels, exponentiated Jensen-Tsallis kernels.
* `conditional entropy (condH)`: conditional Shannon entropy.
* `conditional mutual information (condI)`: conditional Shannon mutual information.

* * *

**Citing**: If you use the ITE toolbox in your research, please cite it \[[.bib](http://www.cmap.polytechnique.fr/~zoltan.szabo/ITE.bib)\].

**Download** the latest release:

- code: [zip](https://bitbucket.org/szzoli/ite-in-python/downloads/ITE-1.1_code.zip), [tar.bz2](https://bitbucket.org/szzoli/ite-in-python/downloads/ITE-1.1_code.tar.bz2),
- documentation: [pdf](https://bitbucket.org/szzoli/ite-in-python/downloads/ITE-1.1_documentation.pdf).

**Note**: the evolution of the code is briefly summarized in CHANGELOG.txt.

* * *

**ITE mailing list**: You can [sign up](https://groups.google.com/d/forum/itetoolbox) here.

**Follow ITE**: on [Bitbucket](https://bitbucket.org/szzoli/ite-in-python/follow), on [Twitter](https://twitter.com/ITEtoolbox).

**ITE applications**: [Wiki](https://bitbucket.org/szzoli/ite/wiki). Feel free to add yours.
