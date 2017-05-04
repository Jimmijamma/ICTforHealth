clear all
close all
clc

Fsamp=8000;
Nbits=8;
Nchann=1;
interval=1;
recObj = audiorecorder(Fsamp, Nbits, Nchann);

disp('Start speaking "a" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
a1 = getaudiodata(recObj);
save('a1.mat','a1');
disp('Start speaking "a" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
a2 = getaudiodata(recObj);
save('a2.mat','a2');
disp('Start speaking "a" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
a3 = getaudiodata(recObj);
save('a3.mat','a3');
disp('Start speaking "a" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
a4 = getaudiodata(recObj);
save('a4.mat','a4');
disp('Start speaking "a" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
a5 = getaudiodata(recObj);
save('a5.mat','a5');

disp('Start speaking "e" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
e1 = getaudiodata(recObj);
save('e1.mat','e1');
disp('Start speaking "e" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
e2 = getaudiodata(recObj);
save('e2.mat','e2');
disp('Start speaking "e" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
e3 = getaudiodata(recObj);
save('e3.mat','e3');
disp('Start speaking "e" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
e4 = getaudiodata(recObj);
save('e4.mat','e4');
disp('Start speaking "e" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
e5 = getaudiodata(recObj);
save('e5.mat','e5');


disp('Start speaking "i" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
i1 = getaudiodata(recObj);
save('i1.mat','i1');
disp('Start speaking "i" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
i2 = getaudiodata(recObj);
save('i2.mat','i2');
disp('Start speaking "i" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
i3 = getaudiodata(recObj);
save('i3.mat','i3');
disp('Start speaking "i" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
i4 = getaudiodata(recObj);
save('i4.mat','i4');
disp('Start speaking "i" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
i5 = getaudiodata(recObj);
save('i5.mat','i5');

disp('Start speaking "o" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
o1 = getaudiodata(recObj);
save('o1.mat','o1');
disp('Start speaking "o" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
o2 = getaudiodata(recObj);
save('o2.mat','o2');
disp('Start speaking "o" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
o3 = getaudiodata(recObj);
save('o3.mat','o3');
disp('Start speaking "o" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
o4 = getaudiodata(recObj);
save('o4.mat','o4');
disp('Start speaking "o" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
o5 = getaudiodata(recObj);
save('o5.mat','o5');

disp('Start speaking "u" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
u1 = getaudiodata(recObj);
save('u1.mat','u1');
disp('Start speaking "u" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
u2 = getaudiodata(recObj);
save('u2.mat','u2');
disp('Start speaking "u" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
u3 = getaudiodata(recObj);
save('u3.mat','u3');
disp('Start speaking "u" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
u4 = getaudiodata(recObj);
save('u4.mat','u4');
disp('Start speaking "u" after hitting the key')
w = input('Hit any key to continue ');
recordblocking(recObj, interval); 
u5 = getaudiodata(recObj);
save('u5.mat','u5');





y=myRecording
%[y,Fs] = audioread('AleAndroid01.wav');
%y=mean(y,2); % converting from stereo to mono
N_file=length(y);

%normalizing the signal
% y=(y-min(y))/(max(y)-min(y));
% m_y=mean(abs(y));
% y=y-m_y;

