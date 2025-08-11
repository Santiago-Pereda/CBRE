function eta=sampleeta(family,rho,nc)

%Get random draws of a multivariate random variable of dimension nc,
%marginal N(0,1), and copula given by family with parameter rho
%
%Input
%
%family={Gaussian,Clayton,Frank,Gumbel}
%
%rho=copula correlation parameter
%
%nc=copula dimension
%
%Output
%
%eta=vector of standard normally distributed random variables with the
%
%correlation determined by the copula

eta=mvcoprnd(family,rho,1,nc);
eta=norminv(eta);