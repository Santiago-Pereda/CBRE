function ll=bc_re_ll(y,x,b,u,wu,link)

%Binary choice random effects estimator log-likelihood
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

[N,T]=size(y);%sample size, number of periods
[~,K]=size(x);
K=K/T;%number of covariates
[R,~]=size(u);%quadrature grid size

%predefine some variables
p=zeros(N,1);
for i1=1:1:N
    xi=[eye(T),reshape(x(i1,:),K,T)'];%individual covariates
    z=kron(u*b(T+K+1)*sqrt(2),ones(1,T))+ones(R,1)*b(1:T+K)'*xi';%latent variable for each value of individual effect
    switch link
        case 'logit'
            h=min(max(exp(z.*y(i1,:))./(1+exp(z)),10^-15),1-10^-15);%logit cdf
            p(i1)=wu'*prod(h,2)/sqrt(pi);%individual probability
        case 'probit'
            h=ones(R,1)*(2*y(i1,:)-ones(1,T)).*min(max(normcdf(z),10^-15),1-10^-15)+ones(R,T)-ones(R,1)*y(i1,:);%normal cdf
            p(i1)=wu'*prod(h,2)/sqrt(pi);%individual probability
    end
end

ll=sum(log(sum(p,2)));