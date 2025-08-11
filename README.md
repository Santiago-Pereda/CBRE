This repository contains the following matlab m-files to implement the estimators in "Copula-Based Random Effects Models for Clustered Data" Pereda-Fernández (2021)

The article can be accessed at https://amstat.tandfonline.com/doi/full/10.1080/07350015.2019.1688665

The folder "Matlab files" contains all the functions that are needed to compute the Copula-Based Random Effects estimator. In particular, it includes the estimator with regular random effects (bc\_cbre.m) and with correlated random effects (bc\_cbcre.m) for the Gaussian, Clayton, Frank and Gumbel copulas, as well as their counterparts with the Bernstein copula (bc\_cbreb.m and bc\_cbcreb.m). In addition, it includes all the other necessary files that are used by these functions:

-bc\_cbre\_ll.m

-bc\_cbre\_lik.m

-bc\_cbcre\_ll.m

-bc\_cbcre\_lik.m

-bc\_cbreb\_ll.m

-bc\_cbreb\_lik.m

-bc\_cbcreb\_ll.m

-bc\_cbcreb\_lik.m

-gridcopula.m

In addition, it also contains some files that are used to obtain the pooled estimators without random effects (bc\_re.m and bc\_cre.m, respectively), as well as the necessary files that are used by these functions (bc\_re\_ll.m, bc\_re\_lik.m, bc\_re\_var.m, bc\_cre\_ll.m, bc\_cre\_lik.m, bc\_cre\_var.m, GaussHermite\_2.m); the file to compute the average partial effect (binaryape.m), the tests used in the paper (testswcbre.m, testswcbcre.m, testindcbre.m), and to sample the random effects using the copula distribution (sampleeta.m, mvcoprnd.m). A more detailed description of the input and output for each function can be found within each file.

