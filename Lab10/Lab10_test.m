clear all
close all
clc

load('ESTTRa.mat');
load('ESTEMITa.mat');
load('ESTTRe.mat');
load('ESTEMITe.mat');
load('ESTTRi.mat');
load('ESTEMITi.mat');
load('ESTTRo.mat');
load('ESTEMITo.mat');
load('ESTTRu.mat');
load('ESTEMITu.mat');

load('za.mat');
load('ze.mat');
load('zi.mat');
load('zo.mat');
load('zu.mat');
% 
% Kquant=16
% 
% Fsamp=8000;
% Nbits=8;
% Nchann=1;
% interval=1;
% recObj = audiorecorder(Fsamp, Nbits, Nchann);
% 
% disp('Start speaking "a" after hitting the key')
% w = input('Hit any key to continue ');
% recordblocking(recObj, interval); 
% ta = getaudiodata(recObj);
% amax=max(ta);
% amin=min(ta);
% delta=(amax-amin)/(Kquant-1);%quantization interval
% ta=round((ta-amin)/delta)+1;%quantized signal
% save('ta.mat','ta');
% 
% disp('Start speaking "e" after hitting the key')
% w = input('Hit any key to continue ');
% recordblocking(recObj, interval); 
% te = getaudiodata(recObj);
% amax=max(te);
% amin=min(te);
% delta=(amax-amin)/(Kquant-1);%quantization interval
% te=round((te-amin)/delta)+1;%quantized signal
% save('te.mat','te');
% 
% disp('Start speaking "i" after hitting the key')
% w = input('Hit any key to continue ');
% recordblocking(recObj, interval); 
% ti = getaudiodata(recObj);
% amax=max(ti);
% amin=min(ti);
% delta=(amax-amin)/(Kquant-1);%quantization interval
% ti=round((ti-amin)/delta)+1;%quantized signal
% save('ti.mat','ti');
% 
% disp('Start speaking "o" after hitting the key')
% w = input('Hit any key to continue ');
% recordblocking(recObj, interval); 
% to = getaudiodata(recObj);
% amax=max(to);
% amin=min(to);
% delta=(amax-amin)/(Kquant-1);%quantization interval
% to=round((to-amin)/delta)+1;%quantized signal
% save('to.mat','to');
% 
% disp('Start speaking "u" after hitting the key')
% w = input('Hit any key to continue ');
% recordblocking(recObj, interval); 
% tu = getaudiodata(recObj);
% amax=max(tu);
% amin=min(tu);
% delta=(amax-amin)/(Kquant-1);%quantization interval
% tu=round((tu-amin)/delta)+1;%quantized signal
% save('tu.mat','tu');


ta=za(:,5);
te=ze(:,5);
ti=zi(:,5);
to=zo(:,5);
tu=zu(:,5);


[PSTATESaa,logpseqaa] = hmmdecode(ta',ESTTRa,ESTEMITa);
paa=logpseqaa;
[PSTATESaa,logpseqae] = hmmdecode(ta',ESTTRe,ESTEMITe);
pae=logpseqae;
[PSTATESaa,logpseqai] = hmmdecode(ta',ESTTRi,ESTEMITi);
pai=logpseqai;
[PSTATESaa,logpseqao] = hmmdecode(ta',ESTTRo,ESTEMITo);
pao=logpseqao;
[PTATESaa,logpseqau] = hmmdecode(ta',ESTTRu,ESTEMITu);
pau=logpseqau;
row1=[paa,pae,pai,pao,pau];


[PSTATESaa,logpseqea] = hmmdecode(te',ESTTRa,ESTEMITa);
pea=logpseqea;
[PSTATESaa,logpseqee] = hmmdecode(te',ESTTRe,ESTEMITe);
pee=logpseqee;
[PSTATESaa,logpseqei] = hmmdecode(te',ESTTRi,ESTEMITi);
pei=logpseqei;
[PSTATESaa,logpseqeo] = hmmdecode(te',ESTTRo,ESTEMITo);
peo=logpseqeo;
[PTATESaa,logpseqeu] = hmmdecode(te',ESTTRu,ESTEMITu);
peu=logpseqeu;
row2=[pea,pee,pei,peo,peu];


[PSTATESaa,logpseqia] = hmmdecode(ti',ESTTRa,ESTEMITa);
pia=logpseqia;
[PSTATESaa,logpseqie] = hmmdecode(ti',ESTTRe,ESTEMITe);
pie=logpseqie;
[PSTATESaa,logpseqii] = hmmdecode(ti',ESTTRi,ESTEMITi);
pii=logpseqii;
[PSTATESaa,logpseqio] = hmmdecode(ti',ESTTRo,ESTEMITo);
pio=logpseqio;
[PTATESaa,logpseqiu] = hmmdecode(ti',ESTTRu,ESTEMITu);
piu=logpseqiu;
row3=[pia,pie,pii,pio,piu];


[PSTATESaa,logpseqoa] = hmmdecode(to',ESTTRa,ESTEMITa);
poa=logpseqoa;
[PSTATESaa,logpseqoe] = hmmdecode(to',ESTTRe,ESTEMITe);
poe=logpseqoe;
[PSTATESaa,logpseqoi] = hmmdecode(to',ESTTRi,ESTEMITi);
poi=logpseqoi;
[PSTATESaa,logpseqoo] = hmmdecode(to',ESTTRo,ESTEMITo);
poo=logpseqoo;
[PTATESaa,logpseqou] = hmmdecode(to',ESTTRu,ESTEMITu);
pou=logpseqou;
row4=[poa,poe,poi,poo,pou];


[PSTATESaa,logpsequa] = hmmdecode(tu',ESTTRa,ESTEMITa);
pua=logpsequa;
[PSTATESaa,logpseque] = hmmdecode(tu',ESTTRe,ESTEMITe);
pue=logpseque;
[PSTATESaa,logpsequi] = hmmdecode(tu',ESTTRi,ESTEMITi);
pui=logpsequi;
[PSTATESaa,logpsequo] = hmmdecode(tu',ESTTRo,ESTEMITo);
puo=logpsequo;
[PTATESaa,logpsequu] = hmmdecode(tu',ESTTRu,ESTEMITu);
puu=logpsequu;
row5=[pua,pue,pui,puo,puu];

probs=[row1;row2;row3;row4;row5];

n_strikes=0;
for ii=1:5
    [M,I] = max(probs(ii,:))
    if ii == I
        n_strikes=n_strikes+1;
    end
end

p_strikes=n_strikes/5;
        