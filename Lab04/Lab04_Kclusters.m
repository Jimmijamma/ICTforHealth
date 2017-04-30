clear all
close all
clc

load('arrhythmia.mat')

%% ----------- Data Preparation ----------- %%

A=arrhythmia;

A(:, find(sum(abs(A)) == 0)) = []; % we erase the zero columns
class_id=A(:,end); % last vector of the matrix
y=A;
y(:,end)=[]; % we put in y all the features but the last one
[N,F]=size(y);

%normalizing y
mean_y=mean(y,1);
stdv_y=std(y,1);

o=ones(N,1);% o is a column vector 
y=(y-o*mean_y)./(o*stdv_y);% y is normalized

mean_y=mean(y,1); % checking that y matrix is properly normalized
var_y=var(y,1);


% we make a list of classes
classes = sort(unique(class_id)); 
C=length(classes);

%% ---------- Hard K-Means Algorithm --------- %%

% starting conditions
K=4; % number of classes
var_k = ones(1,K); % we start from a set of variance values = 1
pi_k = (1/K)*ones(1,K); % we set all the prior probabilities pi_k to 1/K

% we can decide to start with x_k random, or taken as the mean of the two
% known classes. In a clustering case, the random choice is more realistic
rng('shuffle')
x_k = rand(K,F);

for iteration=1:12

    % evaluating the distance
    for n=1:N
        for k=1:K
            dist(n,k)=norm(y(n,:)-x_k(k,:)).^2;
        end
    end

    for k=1:K
        R_k(:,k) = (pi_k(k)/((2*pi*var_k(k))^(N/2)))*exp(-dist(:,k)/(2*var_k(k)));
    end

    [M,decision]=max(R_k,[],2); % taking the decision
    %'decision' is an array of length N with the corresponding closest region
    % for each element (patient)

    for k=1:K
        w_k=y(find(decision==k),:);
        N_k(k) = size(w_k,1);
        pi_k(k) = N_k(k) / N; 
        x_k(k,:) = mean(w_k,1);
        var_k(k)=0;
        for i = 1:N_k(k)
            var_k(k) = var_k(k) + norm(w_k(i,:)-x_k(k,:)).^2;
        end
        var_k(k) = var_k(k)/((N_k(k) - 1)*F);
         
        m_class(k)=mean(class_id(find(decision==k))); % trying to extract information from the elements of the cluster
        std_class(k)=std(class_id(find(decision==k)));
    end

end




