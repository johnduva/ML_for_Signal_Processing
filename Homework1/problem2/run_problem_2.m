%% Load Notes and Music

[s,fs] = audioread('/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_2/data/polyushka.wav'); 
% resample the signal to a standard sampling rate for convenience
s = resample(s,16000,fs);

% Compute the complex short-time Fourier transform (stft) of the signal, 
% and the magnitude spectrogram from it,
spectrum = stft(s',2048,256,0,hann(2048));
music = abs(spectrum);
sphase = spectrum ./(abs(spectrum)+eps);

notesfolder = '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_2/data/notes_15/'; 
listname = dir([notesfolder '*.wav']); notes = [];
for k=1:length(listname)
    [s,fs] = audioread([notesfolder listname(k).name]);
    s = s(:,1)';
    s = resample(s,16000,fs);
    spectrum = stft(s,2048,256,0,hann(2048));
    
    % Find the central frame
    middle = ceil(size(spectrum,2)/2);
    j = abs(spectrum(:,middle));
    % Clean up everything more than 40db below the peak
    j(find(j < max(j(:))/100)) = 0;
    j = j/norm(j); %normalize the note to unit length
    notes = [notes,j];
end


%% Solution for Problem 2.1 here
W = []; % weight matrix
for j = 1 : size(notes,2)
    N = notes(:,j);
    w = pinv(N) * music ; % w = current weight
    W = [W; w];
end
dlmwrite("problem2a.dat",W);

% Recompose the music 
Mnew = 0;
for i = 1 : size(notes,2)
    N = notes(:,i);
    w = W(i,:);
    M = N*w;
    Mnew = Mnew + M;  
end

% original music audio
reconstructedsignal = stft(music.*sphase,2048,256,0,hann(2048));
audiowrite('music.wav',reconstructedsignal,fs);
% reconstruction audio
reconstructedsignal = stft(Mnew.*sphase,2048,256,0,hann(2048));
audiowrite('reconstructedsignal.wav',reconstructedsignal,fs);

player = audioplayer(reconstructedsignal, fs);
play(player);
stop(player);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The recomposed music doesn't sound as good as the original signal (sounds 
% a bit raspier). The result could be improved by using recomposing jointly
% rather than individually.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Solution to Problem 2.2 here:  Synthesize Music
% Use 'wavwrite' function to write the synthesized music as 'problem2b_synthesis.wav'
% to the 'results' folder.

% Recompose the music 
N = notes;
W2 = pinv(N) * music ; % w = current weight
Mnew2 = N*W2;

% Mnew2 = 0;
% for i = 1 : size(W2,1)
%     N = notes;
%     w = W2(i,:);
%     M = N*w;
%     Mnew2 = Mnew2 + M;
% end

reconstructedsignal = stft(Mnew2.*sphase,2048,256,0,hann(2048));
audiowrite('problem2b_synthesis.wav',reconstructedsignal,fs);
player = audioplayer(reconstructedsignal2, fs);
play(player);
stop(player);

% plot the final output
subplot(1,2,1)
imagesc(Mnew)
subplot(1,2,2)
imagesc(Mnew2)

% Is the recomposed music identical to the music you constructed in Part 1.2?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The second recomposition sounds only slightly better than the first. This
% is confirmed by plotting the output above^.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

