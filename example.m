clear
clc

%Parameters to simulate data
N=2000;%sample size
nc=10*ones(N/10,1);%clusters: groups of size 5
cumnc=[0;cumsum(nc)];
familysamp='clayton';%true copula
familyest='Clayton';%estimation copula 1
familyest2='Gaussian';%estimation copula 2
epsilon='logit';
rho=8;%copula parameter
T=4;%
beta=[-1.5;-1;-.5;0;1;3];%marginal parameters
alpha=.05;%Size of test
beta0=beta;%initial value for the RE estimator

K=1;%number of covariates (on top of time dummies)
% reps=1000;%repetitions
N1=50;% grid size of the copula
N2=50;
R=20;%grid size for the quadrature algorithm
maxit=15;%maximum number of iterations
tol=10^-3;%maximum tolerance

%Generate data
x=[kron(eye(T),ones(N,1)),rand(N*T,K)];%generate covariates vector

eta=zeros(N,1);%generate individual effects
for i=1:1:length(nc)
    eta(cumnc(i)+1:cumnc(i+1),1)=sampleeta(familysamp,rho,nc(i))';
end
ystar=beta(end)*kron(ones(T,1),eta)+x*beta(1:end-1)+random('logistic',0,1,N*T,1);%latent index
y=double(ystar>0);%dependent variable
y0=y;
x0=x;
y=reshape(y,N,T);
x=reshape(x(:,5:end),N,T*K);

%Random effects estimator
[betarel1,Htrel1,llrel1(i0)]=bc_re(y,x,beta0,R,maxit,tol,epsilon);

%Copula-Based Random Effects estimator
[betahatcbre1,Htcbre1,llcbre1(i0)]=bc_cbre(y,x,nc,[betarel1;rho],maxit,tol,epsilon,familyest1,N1,N2);
[betahatcbre2,Htcbre2,llcbre2(i0)]=bc_cbre(y,x,nc,[betarel1;rho],maxit,tol,epsilon,familyest2,N1,N2);

%Average Partial Effects
apere1=binaryreape(y,x,betarel1,R,epsilon);
apecbrec1=binaryreape(y,x,betahatcbre1,R,epsilon);
apecbrec2=binaryreape(y,x,betahatcbre2,R,epsilon);

%Test of independence for correct CBRE estimator
[~,~,test1]=testind(familyest1,betahatcbre1,Htcbre1,alpha);

%Schennach-Wilhelm tests for correct CBRE estimator vs incorrect CBRE
%estimators
test2=testsw(y,x,nc,epsilon,epsilon,familyest1,familyest2,betahatcbre1,betahatcbre2,N1,N2,alpha);
