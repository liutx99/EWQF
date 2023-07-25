clear all 
clc
tic;%tic1


search_agents_no = 4;
max_iteration = 10;
lb=[200,1];
ub=[1000,12];
dim=2;
fobj=@Cost;
N = 10;

[Fbest,Lbest,Convergence_curve]=IGWO(dim,N,max_iteration,lb,ub,fobj);

display(['The best solution obtained by I-GWO is : ', num2str(Lbest)]);
display(['The best optimal value of the objective funciton found by I-GWO is : ', num2str(Fbest)]);
