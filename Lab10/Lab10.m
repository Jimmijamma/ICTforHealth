clear all
close all
clc

[y,Fs] = audioread('AleAndroid01.wav');
y=mean(y,2); % converting from stereo to mono
N_file=length(y);

%normalizing the signal
y=(y-min(y))/(max(y)-min(y));
m_y=mean(abs(y));
y=y-m_y;

figure
plot(y)
title('original signal');

window_size=floor(Fs/7);

for a=1:20
    y_trunc=y(a:a*window_size);
    m=mean(abs(y_trunc));
    if m<=m_y*0.07
        start_y=a*window_size;
    else
        break
    end
end

y=y(start_y:end);

figure
plot(y)
title('truncated signal')

m_y=mean(abs(y));

for a=1:10
    y_trunc=y(a:a*window_size);
    m=mean(abs(y_trunc));
    if m>=m_y*1.7
        start_y=a*window_size;
    else
        break
    end
end

y=y(start_y:end);

figure
plot(y)
title('truncated signal')

% applying quantization
Kquant=16;
ymax=max(y);
ymin=min(y);
delta=(ymax-ymin)/(Kquant-1); % quantization interval
yr=round((y-ymin)/delta)+1; % quantized signal

figure
plot(yr)
title('truncated signal')

Max_samples=8000;
for a=1:4
    za(:,a)=yr(a:a*Max_samples);
end

% [ESTTR,ESTEMIT]=hmmtrain(za,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiteration',100)
