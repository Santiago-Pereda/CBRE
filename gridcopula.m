function grid=gridcopula(dist,rho,N1,N2)

%Generate the grid use to approximate integrals with a copula
%
%Input
%
%family={Gaussian,Clayton,Frank,Gumbel}
%
%rho=copula correlation parameter
%
%N1,N2=number of points for copula grid
%
%Output
%
%grid=approximation grid

gridtheta=linspace(1/(N1+1),N1/(N1+1),N1);
quant=linspace(1/(N2+1),N2/(N2+1),N2);
grid=zeros(N1,N2);
switch dist
    case 'Clayton'
        if rho==0
            grid=ones(N1,1)*linspace(1/(N2+1),N2/(N2+1),N2);
        else
            gridtheta=gaminv(gridtheta,1/rho,1);
            for i=1:1:N1
                grid(i,:)=(1-log(quant)/gridtheta(i)).^(-1/rho);
            end
        end
    case 'Gumbel'
        if rho==1
            grid=ones(N1,1)*linspace(1/(N2+1),N2/(N2+1),N2);
        else
            gridtheta=stblinv(gridtheta,1/rho,1,cos(pi/2/rho)^rho,0);
            for i=1:1:N1
                grid(i,:)=exp(-(-log(quant)/gridtheta(i)).^(1/rho));
            end
        end
    case 'Frank'
        if rho==0
            grid=ones(N1,1)*linspace(1/(N2+1),N2/(N2+1),N2);
        else
            Fx=0;
            x=1;
            i=1;
            while Fx<N1/(N1+1)
                fx=(1-exp(-rho)).^x./x/rho;
                Fx=Fx+fx;
                while gridtheta(i)<Fx
                    grid(i,:)=-log(1+exp(log(quant)./x)*(exp(-rho)-1))/rho;
                    if i<N1
                        i=i+1;
                    else
                        break;
                    end
                end
                x=x+1;
            end
        end
    case 'Gaussian'
        if rho==0
            grid=ones(N1,1)*linspace(1/(N2+1),N2/(N2+1),N2);
        else
            if rho==0
                grid=ones(N1,1)*quant;
            else
                grid=normcdf(sqrt(rho^2)*norminv(quant)'*ones(1,N1)+sqrt(1-rho^2)*ones(N1,1)*norminv(quant));
            end
        end
end