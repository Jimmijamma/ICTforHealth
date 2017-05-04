clear all
close all
clc

load('a1.mat')
load('a2.mat')
load('a3.mat')
load('a4.mat')
load('a5.mat')
load('e1.mat')
load('e2.mat')
load('e3.mat')
load('e4.mat')
load('e5.mat')
load('i1.mat')
load('i2.mat')
load('i3.mat')
load('i4.mat')
load('i5.mat')
load('o1.mat')
load('o2.mat')
load('o3.mat')
load('o4.mat')
load('o5.mat')
load('u1.mat')
load('u2.mat')
load('u3.mat')
load('u4.mat')
load('u5.mat')

Kquant=16;

za=[a1,a2,a3,a4,a5];
ze=[e1,e2,e3,e4,e5];
zi=[i1,i2,i3,i4,i5];
zo=[o1,o2,o3,o4,o5];
zu=[u1,u2,u3,u4,u5];

for ii=1:5
    amax=max(za(:,ii));
    amin=min(za(:,ii));
    delta=(amax-amin)/(Kquant-1);%quantization interval
    za(:,ii)=round((za(:,ii)-amin)/delta)+1;%quantized signal
end

for ii=1:5
    amax=max(ze(:,ii));
    amin=min(ze(:,ii));
    delta=(amax-amin)/(Kquant-1);%quantization interval
    ze(:,ii)=round((ze(:,ii)-amin)/delta)+1;%quantized signal
end

for ii=1:5
    amax=max(zi(:,ii));
    amin=min(zi(:,ii));
    delta=(amax-amin)/(Kquant-1);%quantization interval
    zi(:,ii)=round((zi(:,ii)-amin)/delta)+1;%quantized signal
end

for ii=1:5
    amax=max(zo(:,ii));
    amin=min(zo(:,ii));
    delta=(amax-amin)/(Kquant-1);%quantization interval
    zo(:,ii)=round((zo(:,ii)-amin)/delta)+1;%quantized signal
end

for ii=1:5
    amax=max(zu(:,ii));
    amin=min(zu(:,ii));
    delta=(amax-amin)/(Kquant-1);%quantization interval
    zu(:,ii)=round((zu(:,ii)-amin)/delta)+1;%quantized signal
end

save('za.mat','za');
save('ze.mat','ze');
save('zo.mat','zo');
save('zi.mat','zi');
save('zu.mat','zu');