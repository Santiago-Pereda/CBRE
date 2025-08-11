function ll=bc_cbreb_ll(y,x,nc,b,a,link,N1,d)

%Binary choice copula-based random effects estimator with Bernstein copula
%log-likelihood
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%nc=cluster sizes
%
%b=initial value of marginal parameters
%
%a=initial value of copula parameters
%
%link={logit,probit}
%
%N1=number of points for copula grid
%
%d=order of Bernstein copula
%
%Output:
%
%ll=estimated log-likelihood

[~,T]=size(y);%number of periods
[~,K]=size(x);
K=K/T;%number of covariates
cumnc=[0;cumsum(nc)];
C=length(nc);%number of clusters

grid=linspace(1/(N1+1),N1/(N1+1),N1);%grid of individual effects
eta=norminv(grid);

A=zeros(d+1,d+1);%alpha parameters as in Sancetta, Satchell
A(end,:)=linspace(0,1,d+1);
A(:,end)=A(end,:)';
A(2:end-1,2:end-1)=reshape(a,d-1,d-1);
B=(A(2:end,2:end)+A(1:end-1,1:end-1)-A(2:end,1:end-1)-A(1:end-1,2:end))*d^2;%beta parameters as in Sancetta, Satchell
binoms=zeros(1,d);
for i1=0:1:d-1
    binoms(i1+1)=nchoosek(d-1,i1);
end
polord=0:1:d-1;
P=(ones(N1,1)*binoms).*(grid'.^(polord)).*(1-grid').^(d-1-polord);
c=P*B*P';%copula density matrix

%predefine some variables
z0=zeros(C,1);
for i1=1:1:C
    yi=reshape(y(cumnc(i1)+1:cumnc(i1+1),:)',nc(i1)*T,1);%dependent variable
    xi=[kron(ones(nc(i1),1),eye(T)),reshape(x(cumnc(i1)+1:cumnc(i1+1),:)',K,nc(i1)*T)'];%covariates
    z=xi*b(1:end-1)+b(end)*eta;
    switch link
        case 'logit'
            z2=exp(yi*ones(1,N1).*z)./(1+exp(z));%logit cdf
        case 'probit'
            z2=normcdf(z).*(2*yi*ones(1,N1)-ones(nc(i1)*T,N1))+ones(1,N1)-yi*ones(1,N1);%probit cdf
    end
    z0(i1)=prod(z2(1:T,:),1)*c*prod(z2(T+1:2*T,:),1)'/sum(sum(c));%individual likelihood
end

p1=max(z0,1e-300);%make sure it is positive

ll=sum(log(p1));