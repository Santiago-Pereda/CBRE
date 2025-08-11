function [test,crit,p]=testindcbre(family,beta,H,alpha)

%Test of independence of the copula parameter for the CBRE estimator
%
%Input:
%
%family={Gaussian,Clayton,Frank,Gumbel}
%
%beta=parameters
%
%H=covariance matrix of beta
%
%alpha=size of test
%
%Output:
%
%test=value of test
%
%H=crit=critical value for size alpha
%
%p=p-value

%make sure copula parameter belongs to parameter space
switch family
    case 'Clayton'
        beta0=0;
    case 'Gumbel'
        beta0=1;
    case 'Frank'
        beta0=0;
    case 'Gaussian'
        beta0=0;
end

se=sqrt(diag(H));
test=((beta(end)-beta0)/se(end))^2;
crit=chi2inv(1-2*alpha,1);
p=.5-.5*chi2cdf(test,1);