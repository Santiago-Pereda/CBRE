function [b,H,ll]=bc_cre(y,x,xm,beta0,R,maxit,tol,link)

%Binary choice random effects estimator
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%xm=covariates used for the correlated random effects
%
%beta0=initial value of parameters
%
%R=number of points for the Gauss-Hermite quadrature
%
%maxit=maximum number of iterations for convergence
%
%tol=maximum tolerance for convergence
%
%link={logit,probit}
%
%Output:
%
%b=estimated parameters
%
%H=estimated hessian
%
%ll=estimated log-likelihood

[~,T]=size(y);
[~,K]=size(x);
K=K/T;
[~,Kv]=size(xm);

[u,wu]=GaussHermite_2(R);%grid for Gauss-Hermite quadrature

b=beta0;
i1=0;
H=eye(T+K+Kv+1);
g=ones(T+K+Kv+1,1);
llold=-10^10;
ll=bc_cre_ll(y,x,xm,b,u,wu,link);%ll with initial parameters
while (max(abs(H\g))>tol) && (i1<maxit) && (ll-llold>tol)
    i1=i1+1;
    bold=b;
    llold=ll;
    [~,g,H]=bc_cre_lik(y,x,xm,bold,u,wu,link);%get gradient and hessian
    step=H\g;
    ll=bc_cre_ll(y,x,xm,bold+step,u,wu,link);%ll for initial parameters+step
    if ll-llold>0%improved ll
        b=bold+step;%take full step
        b(end)=abs(b(end));
%         disp(['iteration: ' num2str(i1)]);
%         disp(['log-likelihood: ' num2str(ll)]);
    else%decreased likelihood
        i0=1;
        while ll-llold<0 && i0<5
            step=H\g*(.25^i0);
            ll=bc_cre_ll(y,x,xm,bold+step,u,wu,link);%ll for initial parameter+step
            b=bold+step;%take smaller step
            i0=i0+1;
        end
        if i0<5
            [~,g,H]=bc_cre_lik(y,x,xm,b,u,wu,link);%get gradient and hessian
        end
%         disp(['iteration: ' num2str(i1)]);
%         disp(['log-likelihood: ' num2str(ll)]);
    end
end