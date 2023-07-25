function ff = Cost(c)
load TN.mat
X = beihuTN1(1:1600); 
alpha = round(c(1));       
tau = 0;          
K = round(c(2));             
DC = 0;             
init = 1;           
tol = 1e-7;     

[u, u_hat, omega] = VMD(X, alpha, tau, K, DC, init, tol);
for i = 1:K
	xx= abs(hilbert(u(i,:))); 
	xxx = xx/sum(xx); 
    ssum=0;
    for ii = 1:size(xxx,2)
		bb = xxx(1,ii)*log(xxx(1,ii)); 
        ssum=ssum+bb;  
    end
    fitness(i,:) = -ssum;   
end
ff = mean(fitness);
end
