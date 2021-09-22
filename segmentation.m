close all
clear all
clc

%% This script generate audio files based on time stamps provided in a text files.

%% provide the time stamps of the words or sentence
fileList = dir('C:\Users\fh_ai\Documents\frontier_ADReSS\word_timestamps\data\word_timestamps\test\*.txt');



for i=1:1:length(fileList)
    
     file=fileList(i).name;
     
     % reading the time stamps 
     T = readtable(strcat('./word_timestamps/test/',file));
     
     string=strsplit(file,'-');
     
     % loading audio file
     [data, fs]=audioread(strcat('./ADReSS-IS2020-data/test/Full_wave_enhanced_audio/',string{1},'.wav'));
data=[data ; zeros(10.*44100,1)];
%       [data, fs]=audioread(strcat('C:\Users\fh_ai\Documents\frontier_ADReSS\word_timestamps\data\wav\train\transcript_aligned\cc\',file(:,1:end-4),'.wav'))

     for j=1:1: height(T)
     w(j)=T{j,1};
     st(j)= 1000.*T{j,2}+str2num(string{2}); % adding time offset
     et(j)=1000.*T{j,3}+str2num(string{2})+200; % adding time offset
     
     sts(j)=floor(44.1.*(st(j))); % coverting to samples
     ets(j)=floor(44.1.*(et(j)));  % coverting to samples
     y=data(sts(j):ets(j));  % extracting word segment
     y=y./(max(abs(y))); % normalising
     
     % saving audio clips
     filename= strcat('./test/',file(:,1:end-4),'-',num2str(j),'-',w{j},'-',num2str(floor(1000.*T{j,2})),'-' ,num2str(floor(1000.*T{j,3})),'.wav');
     audiowrite( filename , y , fs )
     
     clear y 
     end
     
clear file data    y T st et sts ets
end

