clear all
close all
clc

load('za.mat');
load('ze.mat');
load('zi.mat');
load('zo.mat');
load('zu.mat');

K=16; % number of quantization levels
M=8; % number of states
TRANS_HAT=rand(M,M)
for ii=1:M
    TRANS_HAT(ii,:)=(TRANS_HAT(ii,:))/sum((TRANS_HAT(ii,:)));
end
EMIT_HAT=rand(M,K);
for ii=1:M
    EMIT_HAT(ii,:)=(EMIT_HAT(ii,:))/sum((EMIT_HAT(ii,:)));
end

[ESTTRa,ESTEMITa]=hmmtrain(za,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200);
save('ESTTRa.mat','ESTTRa');
save('ESTEMITa.mat','ESTEMITa');
[ESTTRe,ESTEMITe]=hmmtrain(ze,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',100);
save('ESTTRe.mat','ESTTRe');
save('ESTEMITe.mat','ESTEMITe');
[ESTTRi,ESTEMITi]=hmmtrain(zi,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',100);
save('ESTTRi.mat','ESTTRi');
save('ESTEMITi.mat','ESTEMITi');
[ESTTRo,ESTEMITo]=hmmtrain(zo,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',100);
save('ESTTRo.mat','ESTTRo');
save('ESTEMITo.mat','ESTEMITo');
[ESTTRu,ESTEMITu]=hmmtrain(zu,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',100);
save('ESTTRu.mat','ESTTRu');
save('ESTEMITu.mat','ESTEMITu');

