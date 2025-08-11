function test=testswcbcre(y,x,xm,nc,link1,link2,family1,family2,beta1,beta2,N1,N2,alpha)

%Schennach-Wilhelm (2017) test for the CBRE estimator with correlated
%effects
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%xm=covariates used for the correlated random effects
%
%nc=group sizes
%
%link={logit,probit}
%
%family={Gaussian,Clayton,Frank,Gumbel}
%
%beta=parameters
%
%N1,N2=number of points for copula grid
%
%alpha=size of test
%
%Output:
%
%test=p-value of test

C=length(nc);

%likelihood of each observation for each parametric estimator
[~,z01]=bc_cbcre_ll(y,x,xm,nc,beta1,link1,family1,N1,N2);
[~,z02]=bc_cbcre_ll(y,x,xm,nc,beta2,link2,family2,N1,N2);

sigmaa=mean((log(z01)-mean(log(z01))).^2);%\hat{\sigma}_A
sigmab=mean((log(z02)-mean(log(z02))).^2);%\hat{\sigma}_B
sigmaab=mean((log(z01)-mean(log(z01))).*(log(z02)-mean(log(z02))));%\hat{\sigma}_{AB}
sigma=sigmaa+sigmab-2*sigmaab;%\hat{\sigma}
z=norminv(alpha/2);
delta=sigma/(2*(z-sqrt(4+z^2)));

%tuning parameter estimated as recommended in SW
Cpl=normpdf(z-delta/sqrt(sigma))*(delta*(sigma-2*sigmaa-2*sigmab)/(4*sigma^3));
Csd=2*normpdf(z)*(length(beta1)/sqrt((sigmaa+sigmab)/2));
eps=(Csd/Cpl)^(1/3)*C^(-1/6)*log(log(C))^(1/3);

eps1=zeros(C,1);
eps2=zeros(C,1);
eps1(1:2:end,1)=eps;
eps2(2:2:end,1)=eps;
ll1=log(z01).*(ones(C,1)+eps1);
ll2=log(z02).*(ones(C,1)+eps2);

%\tilde{t}_n as in SW
tn=sqrt(C)*(mean(ll2-ll1))/sqrt(sigma*(1+eps)+eps^2/2*(sigmaa+sigmab));
test=normcdf(tn);