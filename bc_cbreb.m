function [b,a,H]=bc_cbreb(y,x,nc,beta0,alpha0,maxit,tol,link,N1,d)

%Binary choice copula-based random effects estimator with Bernstein copula
%
%Input:
%
%y=dependent variable
%
%x=covariates (excluding time dummies)
%
%nc=cluster sizes
%
%beta0=initial value of marginal parameters
%
%alpha0=initial value of copula parameters
%
%maxit=maximum number of iterations for convergence
%
%tol=maximum tolerance for convergence
%
%link={logit,probit}
%
%N1=number of points for copula grid
%
%d=order of Bernstein copula
%
%Output:
%
%b=estimated marginal parameters
%
%a=estimated copula parameters
%
%H=estimated hessian

b=beta0;
a=alpha0;
a=bernsteintrue(a,d);%make sure it it well-defined

%the algorithm alternatively takes steps for the marginal and the copula
%parameters for numerical reasons
i1=0;
llold=-10^10;
ll=bc_cbreb_ll(y,x,nc,b,a,link,N1,d);%ll for initial parameter
stepb=tol+1;
while (max(abs(stepb))>tol) && (i1<maxit) && (ll-llold>tol)
    i1=i1+1;
    bold=b;
    llold=ll;
    [~,gb,Hb,~,~]=bc_cbreb_lik(y,x,nc,bold,a,link,N1,d);%get gradient and hessian
    stepb=Hb\gb;
    ll=bc_cbreb_ll(y,x,nc,bold+stepb,a,link,N1,d);%ll for initial parameters+step
    if (ll-llold>0)%improved ll
        b=bold+stepb;%take full step
    else%decreased ll
        i0=1;
        while ll-llold<0 && i0<8
            stepb=stepb*.25;%take smaller step
            b=bold+stepb;
            ll=bc_cbreb_ll(y,x,nc,bold+stepb,a,link,N1,d);%ll for initial parameters+step
            i0=i0+1;
        end
        if ll-llold<0 && i0==8
            b=bold;
        end
    end
    i2=0;
    llold2=llold;
    stepa=tol+1;
    while (max(abs(stepa))>tol) && (i2<maxit) && (ll-llold2>tol)
        i2=i2+1;
        aold=a;
        llold2=ll;
        [~,~,~,ga,Ha]=bc_cbreb_lik(y,x,nc,b,aold,link,N1,d);%get gradient and hessian
        stepa=ga./diag(Ha);
        [a,~]=bernsteintrue(aold+stepa,d);%take full step and make sure copula is well-defined
        ll=bc_cbreb_ll(y,x,nc,b,a,link,N1,d);%ll for initial parameters+step
        if (ll-llold2>0)%improved ll
        else%decreased ll
            i0=1;
            while (ll-llold2<0) && (i0<8)
                stepa=stepa*.1;
                [a,~]=bernsteintrue(aold+stepa,d);%take smaller step
                ll=bc_cbreb_ll(y,x,nc,b,a,link,N1,d);%ll for initial parameters+step
                i0=i0+1;
            end
            if ll-llold2<0 && i0==8
                a=aold;
                break;
            end
        end
    end
    disp(['iteration: ' num2str(i1)]);
    disp(['log-likelihood: ' num2str(ll)]);
end

[~,~,~,~,~,H]=bc_cbreb_lik(y,x,nc,b,a,link,N1,d);%hessian