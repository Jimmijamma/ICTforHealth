%% Data preparation

clear all
close all
clc

load('arrhythmia.mat');

A=arrhythmia;

A(:, find(sum(abs(A)) == 0)) = []; % we erase the zero columns

class_id=A(:,end); % last vector of the matrix
class_id(find(class_id>1))=2; % all the values higher than 1 are put equal to 2 
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

n_healthy=sum(class_id==1);
n_ill=sum(class_id==2);


%% SVM implementation

C = [2:0.5:5]

for ii=1:length(C)

    Mdl=fitcsvm(y,class_id,'BoxConstraint',C(ii),'KernelFunction','linear'); 
    classhat=sign(y*Mdl.Beta+Mdl.Bias);

    n_false_negative=length(find(class_id(classhat==-1)==2));
    n_false_positive=length(find(class_id(classhat==1)==1));
    n_true_negative=length(find(class_id(classhat==-1)==1));
    n_true_positive=length(find(class_id(classhat==1)==2));

    p_true_positive=100*n_true_positive/n_ill; % 87.92
    p_true_negative=100*n_true_negative/n_healthy; % 93.87
    p_false_positive=100*n_false_positive/n_healthy; % 6.12
    p_false_negative=100*n_false_negative/n_ill; % 12.07

    p_strike(ii)=(n_true_positive+n_true_negative)/N % 91,15
    
    
    CVMdl = crossval(Mdl);
    classLoss(ii) = kfoldLoss(CVMdl);

    
end

figure
plot(C,p_strike)
xlabel('Box constraint');
ylabel('Strike probability');

figure
plot(C,classLoss)
xlabel('Box constraint');
ylabel('classLoss');

% figure
% hold on
% b=bar(1,p_strike);
% b2=bar(2,p_true_positive,'r');
% b3=bar(3,p_true_negative,'g');
% b4=bar(4,p_false_positive,'y');
% b5=bar(5,p_false_negative,'m');
% 
% title('Results')
% legend('pStrike','pTruePositive','pTrueNegative','pFalseePositive','pFalseNegative')




