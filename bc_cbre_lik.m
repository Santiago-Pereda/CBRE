function [ll,g,H]=bc_cbre_lik(y,x,nc,b,link,family,N1,N2)

%Binary choice copula-based random effects estimator jacobian and hessian
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%nc=cluster sizes
%
%b=initial value of parameters
%
%link={logit,probit}
%
%family={Gaussian,Clayton,Frank,Gumbel}
%
%N1,N2=number of points for copula grid
%
%Output:
%
%ll=estimated log-likelihood
%
%g=jacobian
%
%H=hessian

[~,T]=size(y);%number of periods
[~,K]=size(x);
K=K/T;%number of covariates
cumnc=[0;cumsum(nc)];
C=length(nc);%number of clusters

grid=gridcopula(family,b(end),N1,N2);%grid of individual effects
eta=norminv(grid);

%predefine some variables
z0=zeros(C,1);
z00=zeros(C,length(b)-1);
for i1=1:1:C
    yi=reshape(y(cumnc(i1)+1:cumnc(i1+1),:)',nc(i1)*T,1);%dependent variable
    xi=[kron(ones(nc(i1),1),eye(T)),reshape(x(cumnc(i1)+1:cumnc(i1+1),:)',K,nc(i1)*T)'];%covariates
    for i2=1:1:N1%double integral: first wrt "N2", then wrt "N1"
        z3=zeros(1,size(xi,2));
        z4=0;
        etai=eta(i2,:);
        switch link
            case 'logit'
                g=zeros(nc(i1),1);
                z=(exp(yi*ones(1,N2).*(ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2))))./(1+exp(ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2)));%logit cdf
                for i3=1:1:nc(i1)
                    g(i3)=sum(prod(z((i3-1)*T+1:i3*T,:),1),2);%for each individual, compute product across time
                    z2(i3,1:N2)=prod(z((i3-1)*T+1:i3*T,:))./sum(prod(z((i3-1)*T+1:i3*T,:)));
                    z3=z3+z2(i3,:)*(yi((i3-1)*T+1:i3*T)*ones(1,N2)-exp(ones(T,1)*etai*b(end-1)+xi((i3-1)*T+1:i3*T,:)*b(1:end-2)*ones(1,N2))./(1+exp(ones(T,1)*etai*b(end-1)+xi((i3-1)*T+1:i3*T,:)*b(1:end-2)*ones(1,N2))))'*xi((i3-1)*T+1:i3*T,:);%derivative wrt covariates
                    z4=z4+z2(i3,:).*etai*sum(yi((i3-1)*T+1:i3*T)*ones(1,N2)-exp(ones(T,1)*etai*b(end-1)+xi((i3-1)*T+1:i3*T,:)*b(1:end-2)*ones(1,N2))./(1+exp(ones(T,1)*etai*b(end-1)+xi((i3-1)*T+1:i3*T,:)*b(1:end-2)*ones(1,N2))),1)';%derivative wrt individual effect standard deviation
                end
                z0(i1)=z0(i1)+prod(g)/N2^nc(i1)/N1;%for each "N1", compute product across individuals in the cluster and sum it up
                z00(i1,:)=z00(i1,:)+[prod(g)*z3,prod(g)*z4]/N2^nc(i1)/N1;
            case 'probit'
                g=zeros(nc(i1),N2);
                z=ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2);%probit cdf
                h=min(max(normcdf(z),10^-15),1-10^-15).*(2*yi*ones(1,N2)-ones(nc(i1)*T,N2))+ones(nc(i1)*T,N2)-yi*ones(1,N2);
                dh=normpdf(z).*(2*yi-ones(nc(i1)*T,1));
                for i3=1:1:nc(i1)
                    g(i3,:)=prod(h((i3-1)*T+1:i3*T,:),1);%for each individual, compute product across time
                    z3=z3+g(i3,:)*(dh((i3-1)*T+1:i3*T,:)'./h((i3-1)*T+1:i3*T,:)'*xi((i3-1)*T+1:i3*T,:))/sum(g(i3,:));%derivative wrt covariates
                    z4=z4+g(i3,:)*(sum(dh((i3-1)*T+1:i3*T,:)'./h((i3-1)*T+1:i3*T,:)',2).*etai')/sum(g(i3,:));%derivative wrt individual effect standard deviation
                end
                z0(i1)=z0(i1)+prod(sum(g,2))/N2^nc(i1)/N1;%for each "N1", compute product across individuals in the cluster and sum it up
                z00(i1,:)=z00(i1,:)+[prod(sum(g,2))*z3,prod(sum(g,2))*z4]/N2^nc(i1)/N1;
        end
    end
end

z00=z00./(z0*ones(1,size(xi,2)+1));
z00(isnan(z00))=0;
z00(isinf(z00))=0;

p1=max(z0,1e-300);%make sure it is positive

ll=sum(log(p1));

%numerically approximate derivative wrt copula parameter
switch family
    case 'Clayton'
        eps=0.001;
    case 'Gumbel'
        eps=0.001;
    case 'Frank'
        eps=0.1;
    case 'Gaussian'
        eps=0.001;
end
s=[z00,zeros(C,1)];

b(end)=b(end)+eps;

grid=gridcopula(family,b(end),N1,N2);
eta=norminv(grid);
z0=zeros(C,1);
for i1=1:1:C
    yi=reshape(y(cumnc(i1)+1:cumnc(i1+1),:)',nc(i1)*T,1);
    xi=[kron(ones(nc(i1),1),eye(T)),reshape(x(cumnc(i1)+1:cumnc(i1+1),:)',K,nc(i1)*T)'];
    for i2=1:1:N1
        etai=eta(i2,:);
        switch link
            case 'logit'
                g=zeros(nc(i1),1);
                z=(exp(yi*ones(1,N2).*(ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2))))./(1+exp(ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2)));
                for i3=1:1:nc(i1)
                    g(i3)=sum(prod(z((i3-1)*T+1:i3*T,:),1),2);
                end
                z0(i1)=z0(i1)+prod(g)/N2^nc(i1)/N1;
            case 'probit'
                g=zeros(nc(i1),N2);
                z=ones(nc(i1)*T,1)*etai*b(end-1)+xi*b(1:end-2)*ones(1,N2);
                temp1=min(max(normcdf(z),10^-15),1-10^-15);
                h=temp1.*(2*yi*ones(1,N2)-ones(nc(i1)*T,N2))+ones(nc(i1)*T,N2)-yi*ones(1,N2);
                for i3=1:1:nc(i1)
                    g(i3,:)=prod(h((i3-1)*T+1:i3*T,:),1);
                end
                z0(i1)=z0(i1)+prod(sum(g,2))/N2^nc(i1)/N1;
        end
    end
end
z0=max(z0,1e-300);
s(:,end)=(log(z0)-log(p1))/eps;
b(end)=b(end)-eps;

g=sum(s,1)';

H=zeros(length(b),length(b));
for i1=1:1:C
    H=H+s(i1,:)'*s(i1,:);
end