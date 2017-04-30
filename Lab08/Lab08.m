%% Data Preparation

clear all
close all
clc

A=imread('melanoma_4.jpg');

figure
imshow(A);


% A is a 3D-matrix made of 3 N1 ? N2 matrices, the first one stores 
% unsigned integers from 0 to 255 that encode red, the second green, 
% the third blue.

[N1,N2,N3]=size(A); 
N=N1*N2;% N is the total number of pixels 
B=double(reshape(A,N,N3));

% B is a matrix with N rows and 3 columns: the three columns can be thought
% of as features, and we can apply the clustering algorithms like the
% hard and soft K-means.


F=N3;
y=B;

%% ---------- Hard K-Means Algorithm --------- %%

% starting conditions
K=4; % number of clusters

% we can decide to start with x_k random, or taken as the mean of the two
% known classes. In a clustering case, the random choice is more realistic
rng('shuffle');
x_k = rand(K,F)*255;
init = x_k;

for iteration=1:10
    % evaluating the distance
    for n=1:N
        for k=1:K
            dist(n,k)=norm(y(n,:)-x_k(k,:)).^2;
        end
    end

    [M,decision]=min(dist,[],2); % taking the decision
    %'decision' is an array of length N with the corresponding closest region
    % for each element

    for k=1:K
        w_k=y(find(decision==k),:);
        N_k(k) = size(w_k,1);
        x_k(k,:) = mean(w_k,1);
    end

end

for k=1:K
    i_k=find(decision==k);
    n_k(k)=length(i_k);
    for ii=1:n_k(k)
        y(i_k(ii),:)=x_k(k,:);
    end
end

Bnew=floor(y);
Anew=reshape(uint8(Bnew),N1,N2,N3);
imwrite(Anew,'new_image_2.jpg');

figure
imshow(Anew);


