function Ids = FYP_IV_03_func(beta0,Vdata)
%Psi = beta0(1);
Vds = Vdata(:,1);
Vgs = Vdata(:,2);
% Vgs = 10;

%%%%%%%%%%%%%%%%%%%%
polyfit_alpha = [1.223915494981986e-04,-0.004301149853144,0.022028603005521,0.541506201518383];
polyfit_beta = [-0.037436671972722,1.887523041469408,-29.061496665745054,1.556436553230904e+02];
polyfit_lambda = [4.863148330020342e-07,-4.725010620926286e-05,0.001902005775586,-0.040919366350027,0.505596931001760,-3.574488135690544,13.390874599767667,-20.553060805654944];

alpha = polyfit_alpha(1)*Vgs.^3+polyfit_alpha(2)*Vgs.^2+polyfit_alpha(3)*Vgs.^1+polyfit_alpha(4)*Vgs.^0;
beta = polyfit_beta(1)*Vgs.^3+polyfit_beta(2)*Vgs.^2+polyfit_beta(3)*Vgs.^1+polyfit_beta(4)*Vgs.^0;
lambda = polyfit_lambda(1)*Vgs.^7+polyfit_lambda(2)*Vgs.^6+polyfit_lambda(3)*Vgs.^5+polyfit_lambda(4)*Vgs.^4+polyfit_lambda(5)*Vgs.^3+polyfit_lambda(6)*Vgs.^2+polyfit_lambda(7)*Vgs.^1+polyfit_lambda(8)*Vgs.^0;

P1 = beta0(1);
P2 = beta0(2);
P3 = beta0(3);
as = beta0(4);
Vpk0 = beta0(5);
Vpks = beta0(6);
P4 = beta0(7);
P5 = beta0(8);
%kappa = beta0(9);
%k = beta0(10);
%x = beta0(11);
Vpk = Vpk0 + (Vpks-Vpk0)*tanh(as*Vds);
Vgs0 = Vgs - Vpk;
%Ids_max = 700;


% alpha = polyval(polyfit_alpha,Vgs);
% beta = polyval(polyfit_beta,Vgs);
% lambda = polyval(polyfit_lambda,Vgs);

%Psi = P1*Vgs0 + P2*Vgs0.^2 + P3*Vgs0.^3;


Psi = P1.*Vgs0 + P2.*Vgs0.^2 + P3.*Vgs0.^3 + P4.*Vgs0.^4 + P5.*Vgs0.^5;
%%%%%%%%%%%%%%%%%%%%
%beta=beta*13/5.2;
%beta=beta*8.6/5.2;
%lambda=lambda*9/5.1;
%Ids = log10(beta.*(1+tanh(Psi)).*(1+lambda.*Vds).*tanh(alpha.*Vds));
%Ids = log10(beta.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds).*tanh(alpha.*Vds));
Ids = log10(beta.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds).*erf(tanh(alpha.*Vds)));
%Ids = log10(beta.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds+x.*Vgs).*erf(tanh(alpha.*Vds)));
%Ids = log10(beta.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds+x.*Vgs).^(kappa).*erf(tanh(alpha.*Vds).*(1-k.*Vds)));
%Ids = log10(beta.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds+x.*Vgs).*erf(tanh(alpha.*Vds)));


%Ids = log10(beta.*(1+tanh(Psi)).*(1+erf(lambda).*Vds).*tanh(erf(alpha.^1.3).*Vds));
%Ids = log10(beta.^0.5.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds).*k.*erf(tanh(alpha.*Vds)));
%Ids = log10(beta.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds).*erf(1./(1+exp(alpha.^4.9.*Vds))));
%Ids = log10(beta.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds).*(-exp(-(alpha.*Vds).^2)./sqrt(pi).*(alpha.*Vds)));
%Ids_max = log10(beta.*(1+tanh(Psi)).*(1+lambda.*Vds_large).*tanh(alpha.*Vds_large));
%Ids = log10(beta.*3./(1+exp(-Psi.*Vgs)).*(1+lambda.*Vds).*Ids_max.*Vds./(1/2*Ids_max+Vds));
%Ids = log10(beta.^(3.3).*(1+tanh(Psi)).*(1+lambda.*Vds).*tanh(alpha.*Vds));
%Ids = log10(beta.*(1+0.1*sinh(0.4*Psi+1.5)).*(1+lambda.*Vds).*tanh(alpha.*Vds));
%Ids = log10((beta).^(3.3).*(1+sinh(Psi)).*(1+4*lambda.*Vds.^(1/2)).*tanh(1.3*alpha.*Vds.^(1/2)));
%Ids = (beta).^(3.3).*(1+tanh(Psi)).*(1+4*lambda.*Vds.^(1/2)).*tanh(1.3*alpha.*Vds.^(1/2));
%Ids = beta.*(1+tanh(Psi)).*(1+lambda.*Vds).*tanh(alpha.*Vds);
%Ids = log10((beta).*(1+tanh(Psi)).*(1+lambda.*Vds.^(1/2)).*tanh(alpha.*Vds.^(1/2)));
Ids = real(Ids);
% Ids = log10(beta.*(1+1-2./(exp(2*Psi)+1)).*(1+lambda.*Vds).*(1-2./(exp(2*alpha.*Vds)+1)));

end