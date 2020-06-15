%% Factorization-Based Data Modeling --  TP 3: couple matrix factorization 

clear 
close all
clc

%% load the data

tmp = load('uclaf_data.mat'); 
X1 = tmp.UserLocAct;
X2 = tmp.UserLoc;
X3 = tmp.LocFea;
X4 = tmp.UserUser;
X5 = tmp.ActAct;

X1 = X1 + eps;
X2 = X2 + eps;
X3 = X3 + eps;
X4 = X4 + eps;
X5 = X5 + eps;


I = 164;
J = 168;
K = 5;
F = 14; %loc feature


R = 3;

%% Algorithm: Multiplicative Update Rules

%make sure they are initialized non-negative
Amur = 1  * rand(I,R);
Bmur = 1 * rand(J,R);
Cmur = 1 * rand(K,R);

Bmur1 = 1 * rand(R,J);
Dmur = 1 * rand(R,F);
Emur = 1 * rand(R,I);
Fmur = 1 * rand(R,K);

%X1 = A * B * C
for i = 1:I
    for j = 1:J
        for k = 1:K
            X1hat(i,j,k) = 0;
            for r = 1:R
                X1hat(i,j,k) = X1hat(i,j,k) + Amur(i,r) * Bmur(j,r) * Cmur(k,r);
            end
        end
    end
end
            

X2hat = Amur * Bmur1;    %X2 = A * B1
X3hat = Bmur * Dmur;     %X3 = B * D 
X4hat = Amur * Emur;     %X4 = A * E
X5hat = Cmur * Fmur;     %X5 = C * F

O2 = ones(I,J);
O3 = ones(J,F);
O4 = ones(I,I);
O5 = ones(K,K);

MaxIterMur = 100;
obj_mur = zeros(MaxIterMur,1);


%% 
for z = 1:MaxIterMur
    
    Bmur1 = Bmur1 .* (Amur'*(X2./X2hat))./(Amur'*O2);
    Dmur = Dmur .* (Bmur'*(X3./X3hat))./(Bmur'*O3);
    Emur = Emur .* (Amur'*(X4./X4hat))./(Amur'*O4);
    Fmur = Fmur .* (Cmur'*(X5./X5hat))./(Cmur'*O5);
    
    Atmp = ones(I,R);
    for i=1:I
        for r = 1:R
            tmp1 = 0;
            tmp2 = 0;
            for j = 1:J
                for k = 1:K
                    tmp1 = tmp1 + X1(i,j,k) / Amur(i,r);
                    tmp2 = tmp2 + Bmur(j,r) * Cmur(k,r);
                end
            end
            Atmp(i,r) = Atmp(i,r) * tmp1/tmp2;
        end
    end
    
    Amur = Amur .* ((X4./X4hat)*Emur')./(O4*Emur') ;
    % +Amur .*((X2./X2hat)*Bmur1')./(O2*Bmur1')+Atmp
    Btmp = ones(J,R);
    for j=i:J
        for r = 1:R
            tmp1 = 0;
            tmp2 = 0;
            for i = 1:I
                for k = 1:K    
                    tmp1 = tmp1 + X1(i,j,k) / Bmur(j,r);
                    tmp2 = tmp2 + Amur(i,r) * Cmur(k,r);
                end
            end
            Btmp(j,r) = Btmp(j,r) * tmp1/tmp2;
        end
    end
    Bmur = Bmur .* ((X3./X3hat)*Dmur')./(O3*Dmur')+ Btmp;
    %; 
    
    Ctmp = ones(K,R);
    for k =i:K
        for r = 1:R
            tmp1 = 0;           
            tmp2 = 0;
            for i = 1:I
                for j = 1:J
                    tmp1 = tmp1 + X1(i,j,k) / Cmur(k,r);
                    tmp2 = tmp2 + Amur(i,r) * Bmur(j,r);
                end
            end
            Ctmp(k,r) = Ctmp(k,r) * tmp1/tmp2;
        end
    end
    Cmur = Cmur .* ((X5./X5hat)*Fmur')./(O5*Fmur');
    %+ Ctmp;
    
    
    for i = 1:I
      for j = 1:J
        for k = 1:K
            X1hat(i,j,k) = 0;
            for r = 1:R
                X1hat(i,j,k) = X1hat(i,j,k) + Amur(i,r) * Bmur(j,r) * Cmur(k,r);
            end
        end
      end
    end
   
    X2hat = Amur * Bmur1;    %X2 = A * B1
    X3hat = Bmur * Dmur;     %X3 = B * D
    X4hat = Amur * Emur;     %X4 = A * E
    X5hat = Cmur * Fmur;     %X5 = C * F
    
    X1hat = X1hat + eps;
    X2hat = X2hat + eps;
    X3hat = X3hat + eps;
    X4hat = X4hat + eps;
    X5hat = X5hat + eps;
    
    
%     X1tmp =0
%      for i = 1:I
%       for j = 1:J
%         for k = 1:K
%             X1tmp = X1tmp + X1(i,j,k) - log( X1(i,j,k)/ X1hat(i,j,k)) - X1(i,j,k) + X1hat(i,j,k);           
%         end
%       end
%     end
   
    obj_mur(z) = sum(sum((X5 .* log(X5./X5hat))-X5+X5hat))+sum(sum((X3 .* log(X3./X3hat))-X3+X3hat))+sum(sum((X4 .* log(X4./X4hat))-X4+X4hat));
    % 
    %sum(sum(sum((X1 .* log(X1./X1hat))-X1+X1hat))) 
    %
    %sum(sum((X2 .* log(X2./X2hat))-X2+X2hat))

     
    
    
end


figure, 
plot(obj_mur);
xlabel('Iterations');
ylabel('Objective Value');
title('MUR');




figure, 
subplot(2,4,1);
imagesc(X2); axis xy;
colorbar;
title('X2');
subplot(2,4,5);
imagesc(X2hat); axis xy;
colorbar
title('X2hat');

subplot(2,4,2);
imagesc(X3); axis xy;
colorbar;
title('X3');
subplot(2,4,6);
imagesc(X3hat); axis xy;
colorbar
title('X3hat');

subplot(2,4,3);
imagesc(X4); axis xy;
colorbar;
title('X4');
subplot(2,4,7);
imagesc(X4hat); axis xy;
colorbar
title('X4hat');

subplot(2,4,4);
imagesc(X5); axis xy;
colorbar;
title('X5');
subplot(2,4,8);
imagesc(X5hat); axis xy;
colorbar
title('X5hat');



figure, 
subplot(2,1,1);
imagesc(X1(:,:,1)); axis xy;
colorbar;
title('X1');
subplot(2,1,2);
imagesc(X1hat(:,:,1)); axis xy;
colorbar
title('X1hat');
