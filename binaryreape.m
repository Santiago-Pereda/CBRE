function [ape,V]=binaryreape(y,x,b,R,link)

%Binary choice random effects APE estimator
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%b=initial value of parameters
%
%R=number of points for the Gauss-Hermite quadrature
%
%link={logit,probit}
%
%Output:
%
%ape=average partial effects
%
%V=variance matrix

[u,wu]=GaussHermite_2(R);%grid for Gauss-Hermite quadrature

[N,T]=size(y);%sample size, number of periods
[~,K]=size(x);
K=K/T;%number of covariates

%predefine some variables
apes=zeros(K,N);
for i1=1:1:N
    switch link
        case 'logit'
            xi=[eye(T),x(i1,:)'];
            z=kron(u*b(T+K+1)*sqrt(2),ones(1,T))+ones(R,1)*b(1:T+K)'*xi';
            h=exp(z)./(1+exp(z));%logit cdf
            apes(:,i1)=mean(b(T+1:T+K)*wu'*(h.*(1-h)),2);%individual partial effect
        case 'probit'
            xi=[eye(T),x(i1,:)'];
            z=kron(u*b(T+K+1)*sqrt(2),ones(1,T))+ones(R,1)*b(1:T+K)'*xi';
            h=normpdf(z);%probit cdf
            apes(:,i1)=mean(b(T+1:T+K)*wu'*(h),2);%individual partial effect
    end
end

ape=mean(apes,2);

%numerically approximate variance
eps=.0001;
s=zeros(K,N);
for i0=1:1:K
    b(T+i0)=b(T+i0)+eps;
    dapes=zeros(1,N);
    for i1=1:1:N
        switch link
            case 'logit'
                xi=[eye(T),x(i1,:)'];
                z=kron(u*b(T+K+1)*sqrt(2),ones(1,T))+ones(R,1)*b(1:T+K)'*xi';
                h=exp(z)./(1+exp(z));
                dapes(1,i1)=mean(b(T+i0)*wu'*(h.*(1-h)),2);
            case 'probit'
                xi=[eye(T),x(i1,:)'];
                z=kron(u*b(T+K+1)*sqrt(2),ones(1,T))+ones(R,1)*b(1:T+K)'*xi';
                h=normpdf(z);
                dapes(1,i1)=mean(b(T+i0)*wu'*(h),2);
        end
    end
    s(i0,:)=(dapes-apes(i0,:))/eps;
    b(T+i0)=b(T+i0)-eps;
end

V=zeros(K,K);
for i1=1:1:N
    V=V+s(i1,:)*s(i1,:)';
end