clear all 
clc

load  TN.mat
X = beihuTN1(1:1600); 
alpha = 201;       
tau = 0;          
K = 4;              
DC = 0;             
init = 1;           
tol = 1e-7;     

[u, u_hat, omega] = VMD(X, alpha, tau, K, DC, init, tol);

%Sequence decomposition using sliding windows.
imf=u;
for i = 1:1:400
    
    X = beihuTN1(i:1599+i); 
    alpha = 201;       
    tau = 0;          
    K = 4;              
    DC = 0;             
    init = 1;           
    tol = 1e-7;     
    [u, u_hat, omega] = VMD(X, alpha, tau, K, DC, init, tol);
    a=u(1:K,1600);
    
    imf=[imf,a];
end