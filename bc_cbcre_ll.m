function ll=bc_cbcre_ll(y,x,xm,nc,b,link,family,N1,N2)

%Binary choice copula-based random effects estimator log-likelihood
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%xm=covariates used for the correlated random effects
%
%nc=cluster sizes
%
%b=initial value of parameters
%
%link={logit,probit}
%
%family={Gaussian,Clayton,Frank,Gumbel,Ali,Joe}
%
%N1,N2=number of points for copula grid
%
%Output:
%
%ll=estimated log-likelihood

%make sure copula parameter belongs to parameter space
switch family
    case 'Clayton'
        b(end)=max(b(end),0);
    case 'Gumbel'
        b(end)=max(b(end),1);
    case 'Frank'
        b(end)=max(b(end),0);
    case 'Ali'
        b(end)=max(min(b(end),1),0);
    case 'Joe'
        b(end)=max(b(end),1);
    case 'Gaussian'
        b(end)=max(min(b(end),1),0);
end

[~,T]=size(y);%number of periods
[~,K]=size(x);
K=K/T;%number of covariates
cumnc=[0;cumsum(nc)];
C=length(nc);%number of clusters

grid=gridcopula(family,b(end),N1,N2);%grid of individual effects
eta=norminv(grid);

%predefine some variables
z0=zeros(C,1);
for i1=1:1:C
    yi=reshape(y(cumnc(i1)+1:cumnc(i1+1),:)',nc(i1)*T,1);%dependent variable
    xi=[kron(ones(nc(i1),1),eye(T)),reshape(x(cumnc(i1)+1:cumnc(i1+1),:)',K,nc(i1)*T)',kron(xm(cumnc(i1)+1:cumnc(i1+1),:),ones(T,1))];%covariates
    for i2=1:1:N1%double integral: first wrt "N2", then wrt "N1"
        etai=eta(i2,:);
        switch link
            case 'logit'
                g=zeros(nc(i1),1);
                z=(exp(yi*ones(1,N2).*(ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2))))./(1+exp(ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2)));%logit cdf
                for i3=1:1:nc(i1)%for each individual, compute product across time
                    g(i3)=sum(prod(z((i3-1)*T+1:i3*T,:),1),2);
                end
                z0(i1)=z0(i1)+prod(g)/N2^nc(i1)/N1;%for each "N1", compute product across individuals in the cluster and sum it up
            case 'probit'
                g=zeros(nc(i1),N2);
                z=ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2);
                h=min(max(normcdf(z),10^-15),1-10^-15).*(2*yi*ones(1,N2)-ones(nc(i1)*T,N2))+ones(nc(i1)*T,N2)-yi*ones(1,N2);%normal cdf
                for i3=1:1:nc(i1)%for each individual, compute product across time
                    g(i3,:)=prod(h((i3-1)*T+1:i3*T,:),1);
                end
                z0(i1)=z0(i1)+prod(sum(g,2))/N2^nc(i1)/N1;%for each "N1", compute product across individuals in the cluster and sum it up
        end
    end
end

p1=max(z0,1e-300);%make sure it is positive

ll=sum(log(p1));