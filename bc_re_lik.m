function [ll,g,H]=bc_re_lik(y,x,b,u,wu,link)

%Binary choice random effects estimator jacobian and hessian
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%b=initial value of parameters
%
%u=quadrature points
%
%wu=quadrature weights
%
%link={logit,probit}
%
%Output:
%
%ll=estimated log-likelihood
%
%g=jacobian
%
%H=hessian

[N,T]=size(y);%sample size, number of periods
[~,K]=size(x);
K=K/T;%number of covariates
[R,~]=size(u);%quadrature grid size

%predefine some variables
p=zeros(N,1);
s=zeros(N,T+K+1);
for i1=1:1:N
    xi=[eye(T),x(i1,:)'];%individual covariates
    z=kron(u*b(T+K+1)*sqrt(2),ones(1,T))+ones(R,1)*b(1:T+K)'*xi';%latent variable for each value of individual effect
    switch link
        case 'logit'
            h=min(max(exp(z.*y(i1,:))./(1+exp(z)),10^-15),1-10^-15);%logit cdf
            dh=((y(i1,:)-1).*exp(z.*(1-y(i1,:)))+(y(i1,:).*exp(z.*y(i1,:))))./((1+exp(z)).^2);%derivative of cdf
            p(i1)=wu'*prod(h,2)/sqrt(pi);%individual probability
            s(i1,1:T+K)=(wu.*prod(h,2))'*(dh./h)*xi/sqrt(pi)/p(i1);%derivative of probability wrt covariates
            s(i1,T+K+1)=(wu.*prod(h,2))'*((sum(dh./h,2)).*u*sqrt(2))/sqrt(pi)/p(i1);%derivative of probability wrt standard deviation of individual effects
        case 'probit'
            h=ones(R,1)*(2*y(i1,:)-ones(1,T)).*min(max(normcdf(z),10^-15),1-10^-15)+ones(R,T)-ones(R,1)*y(i1,:);%normal cdf
            dh=ones(R,1)*(2*y(i1,:)-ones(1,T)).*normpdf(z);%derivative of cdf
            p(i1)=wu'*prod(h,2)/sqrt(pi);%individual probability
            s(i1,1:T+K)=(wu.*prod(h,2))'*(dh./h)*xi/sqrt(pi)/p(i1);%derivative of probability wrt covariates
            s(i1,T+K+1)=(wu.*prod(h,2))'*((sum(dh./h,2)).*u*sqrt(2))/sqrt(pi)/p(i1);%derivative of probability wrt standard deviation of individual effects
    end
end

ll=sum(log(sum(p,2)));

g=sum(s,1)';

H=zeros(T+K+1,T+K+1);
for i1=1:1:N
    H=H+s(i1,:)'*s(i1,:);
end