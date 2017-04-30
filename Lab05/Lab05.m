clear all
close all
clc

load('ckd3.mat');

%% ----------- Data Preparation ------------ %%

keylist={'normal','abnormal','present','notpresent','yes','no','good','poor','ckd','notckd','?',''};
keymap=[0,1,0,1,0,1,0,1,2,1,NaN,NaN];

ckd=chronickidneydisease;


% adjusting the format of the input data
[kR,kC] = size(ckd);
b=[];
for kr=1:kR
    for kc=1:kC
        c=strtrim(ckd(kr,kc)); % remove blanks
        check=strcmp(c,keylist);
        if sum(check)==0
            b(kr,kc)=str2num(ckd{kr,kc}); % from text to numeric
        else
            ii=find(check==1);
            b(kr,kc)=keymap(ii); % use the lists
        end
    end
end

b=b(:,1:end-1); % I mistakenly considered an empty column in the import phase

[N,F]=size(b);

X=b(:,1:end-1); % we don't consider the last column, that stores classification evaluated by the doctor
classes=b(:,end);

n_healthy=sum(classes==1);
n_ill=sum(classes==2);


%% ---------- Hierachical Clustering ---------- %%

D=pdist(X); % algorithm that evaluates the distance between each 
            % measurement stored in the matrix
D_matrix=squareform(D); % the output of pdist is a row vector. 'squareform' 
                 % transforms it into a matrix of distances d(i,j)
                 
Z=linkage(D); % 

K=2; % selecting the number of clusters we want to consider

    % we create the hierachical tree
T=cluster(Z,'maxclust',K);
    % The output matrix Z contains cluster information. Z has size m-1 by 3
    % where m is the number of observations in the original data. Each newly-formed
    % cluster, corresponding to Z(i,:), is assigned the index m+i, where m is
    % the total number of initial leaves. Z(i,1:2) contains the indices of
    % the two component clusters which form cluster m+i. There are m-1 higher
    % clusters which correspond to the interior nodes of the output
    % clustering tree. Z(i,3) contains the corresponding linkage distances
    % between the two clusters which are merged in Z(i,:).

p=0;
figure
dendrogram(Z,p) 

% we compare the clustering with the classification given by doctors
perc_true=100*length(find(T==classes))/N; % 0.8025
perc_false=length(find(T~=classes))/N; % 0.1975 (1-perc_true)

n_false_negative=length(find(classes(T==1)==2));
n_false_positive=length(find(classes(T==2)==1));
n_true_negative=length(find(classes(T==1)==1));
n_true_positive=length(find(classes(T==2)==2));

p_true_positive=100*n_true_positive/n_ill; % 87.92
p_true_negative=100*n_true_negative/n_healthy; % 93.87
p_false_positive=100*n_false_positive/n_healthy; % 6.12
p_false_negative=100*n_false_negative/n_ill; % 12.07

figure
hold on
b=bar(1,perc_true);
b2=bar(2,p_true_positive,'r');
b3=bar(3,p_true_negative,'g');
b4=bar(4,p_false_positive,'y');
b5=bar(5,p_false_negative,'m');

title('Clustering Results')
legend('pStrike','pTruePositive','pTrueNegative','pFalsePositive','pFalseNegative')


% trying to evaluate the sum of the squared error, that measures the
% performance of the clustering algorithm
w1=X(find(T==1),:);
w2=X(find(T==2),:);

m_k(1,:)=mean(w1,1);
m_k(2,:)=nanmean(w2,1);

SSE_1=0;
SSE_2=0;
for i=1:size(w1,1)
    error_1(i)=norm(w1(i,:)-m_k(1,:)).^2;
    SSE_1=SSE_1+error_1(i);
end

for i=1:size(w2,1)
    error_2(i)=norm(w2(i,:)-m_k(2,:)).^2;
    SSE_2=SSE_2+error_2(i);
end

SSE=SSE_1+SSE_2;


%% --------- Classification Tree ---------- %%

% R_x=X'*X/N;
% [U, E] = eig(R_x);
% 
% P = sum(diag(E));
% percentage = 0.999; % we set the percentage of information that we want to keep
% new_P = percentage * P; 
% 
% cumulative_P = cumsum(diag(E)); % function that evaluates the cumulative 
%                                 % sum of each element of the diagonal of A 
% L = length(find(cumulative_P<new_P)); % determines the first L features 
%                                 % that contribut to obtain new_P amount 
%                                 % of "information"
%                                 
% U_L = U(:,1:L); % we only consider the first L features
% 
% Z = X * U_L;

tc=fitctree(X,classes); % function that generates the classification tree

view(tc); % decision tree explained in command window
view(tc,'Mode','graph'); % graphical representation of the decision tree

% implementation of the decision tree
for i=1:N
    if X(i,15)<13.05
        if X(i,16)<44.5
            ct_classes(i)=2;
        else
            ct_classes(i)=1;
        end
    else
        if X(i,3)<1.0175
            ct_classes(i)=2;
        else
            if X(i,4)<0.5
                ct_classes(i)=1;
            else
                ct_classes(i)=2;
            end
        end
    end
end

ct_classes=ct_classes';

perc_true_ct=100*length(find(ct_classes==classes))/N; % 0.9275
perc_false_ct=length(find(ct_classes~=classes))/N; % 0.0725 (1-perc_true)

n_false_negative_ct=length(find(classes(ct_classes==1)==2));
n_false_positive_ct=length(find(classes(ct_classes==2)==1));
n_true_negative_ct=length(find(classes(ct_classes==1)==1));
n_true_positive_ct=length(find(classes(ct_classes==2)==2));

p_true_positive_ct=100*n_true_positive_ct/n_ill; % 90.40
p_true_negative_ct=100*n_true_negative_ct/n_healthy; % 96.66
p_false_positive_ct=100*n_false_positive_ct/n_healthy; % 3.33
p_false_negative_ct=100*n_false_negative_ct/n_ill; % 9.60

figure
hold on
b=bar(1,perc_true_ct);
b2=bar(2,p_true_positive_ct,'r');
b3=bar(3,p_true_negative_ct,'g');
b4=bar(4,p_false_positive_ct,'y');
b5=bar(5,p_false_negative_ct,'m');

title('Classification Results')
legend('pStrike','pTruePositive','pTrueNegative','pFalsePositive','pFalseNegative')