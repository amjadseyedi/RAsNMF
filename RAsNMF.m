clear
close all
clc
[parentdir,~,~] = fileparts(pwd);
%% choose the data and corresponding parameters
% cornell texas wisconsin washington
load texas
n = size(A,1);
%% Parameters
lambda=400;
p=0.49;
S=simi(A,5,p);
D = diag(sum(S,2));
L = D - S;
%% Initialization
[W, H] = initializeNMFwithSVD(A,r);
minH=realmin*ones(r,r);
minW=realmin*ones(n,r);
%% updating the objective function
objective = zeros(1,num_iter);
for iter = 1:num_iter/2
    % updating H
    H=pinv(W)*A*pinv(W');
    
    % updating W
    Hp=(abs(H)+H)/2;
    Hn=(abs(H)-H)/2;
    
    numer_W = (A)*(W)*(Hp') + (A')*(W)*(Hp) +(W)*(Hn)*(W')*(W)*(Hn') + (W)*(Hn')*(W')*(W)*(Hn)...
               + lambda*(S')*W+ lambda*(S')*W;
    denom_W = (A)*(W)*(Hn') + (A')*(W)*(Hn) +(W)*(Hp)*(W')*(W)*(Hp') + (W)*(Hp')*(W')*(W)*(Hp)...
               + 4*lambda*(D)*W;

    W=W.*((numer_W./max(denom_W,minW)).^(1/8));

    objective(iter)= norm(A-(W)*(H)*(W)')^2 + lambda*trace((W')*(L)*(W));
end
%% Evaluation
E = diag(sum(W,1));
W = W/E;
[~,predict_label] = max(W,[],2);

jaccard = PSJaccard(predict_label, true_label); % jaccard
nmi = PSNMI(predict_label, true_label); % nmi
predicted = bestMap(true_label, predict_label);
accuracy = sum(predicted == true_label)/length(predicted); % accuracy

fprintf('Jaccard \t %.3f\nNMI \t\t %.3f\nAccuracy \t %.3f\n', jaccard, nmi, accuracy);
