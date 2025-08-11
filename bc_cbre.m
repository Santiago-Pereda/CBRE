function [b,H,ll]=bc_cbre(y,x,nc,beta0,maxit,tol,epsilon,family,N1,N2)

%Binary choice copula-based random effects estimator
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%nc=cluster sizes
%
%beta0=initial value of parameters
%
%maxit=maximum number of iterations for convergence
%
%tol=maximum tolerance for convergence
%
%epsilon={logit,probit}
%
%family={Gaussian,Clayton,Frank,Gumbel}
%
%N1,N2=number of points for copula grid
%
%Output:
%
%b=estimated parameters
%
%H=estimated hessian
%
%ll=estimated log-likelihood

%make sure copula parameter belongs to parameter space
b=beta0;
switch family
    case 'Clayton'
        b(end)=max(b(end),0);
    case 'Gumbel'
        b(end)=max(b(end),1);
    case 'Frank'
        b(end)=max(b(end),0);
    case 'Gaussian'
        b(end)=max(min(b(end),.999),0.001);
end

i1=0;
H=eye(length(b));
g=ones(length(b),1);
llold=-10^10;
ll=bc_cbre_ll(y,x,nc,b,epsilon,family,N1,N2);%ll for initial parameter
while (max(abs(H\g))>tol) && (i1<maxit) && (ll-llold>tol)
    i1=i1+1;
    bold=b;
    llold=ll;
    [~,g,H]=bc_cbre_lik(y,x,nc,bold,epsilon,family,N1,N2);%get gradient and hessian
    step=H\g;
    ll=bc_cbre_ll(y,x,nc,bold+step,epsilon,family,N1,N2);%ll for initial parameters+step
    if ll-llold>0%improved ll
        b=bold+step;%take full step
        switch family
            case 'Clayton'
                b(end)=max(b(end),0);
            case 'Gumbel'
                b(end)=max(b(end),1);
            case 'Frank'
                b(end)=max(b(end),0);
            case 'Gaussian'
                b(end)=max(min(b(end),.999),0.001);
        end
%         disp(['iteration: ' num2str(i1)]);
%         disp(['log-likelihood: ' num2str(ll)]);
    else%decreased likelihood
        i0=1;
        while ll-llold<0 && i0<8
            step=H\g*(.25^i0);
            ll=bc_cbre_ll(y,x,nc,bold+step,epsilon,family,N1,N2);%ll for initial parameter+step
            b=bold+step;%take smaller step
            i0=i0+1;
        end
        switch family
            case 'Clayton'
                b(end)=max(b(end),0);
            case 'Gumbel'
                b(end)=max(b(end),1);
            case 'Frank'
                b(end)=max(b(end),0);
            case 'Gaussian'
                b(end)=max(min(b(end),.999),0.001);
        end
%         disp(['iteration: ' num2str(i1)]);
%         disp(['log-likelihood: ' num2str(ll)]);
    end
end