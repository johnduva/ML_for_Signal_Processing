%% Load Notes and Music
% Use the 'load_data' function here

%% Calculate the weight matrix (eqn. 7)
[s,fs] = audioread('/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_2/data/polyushka.wav'); 
% resample the signal to a standard sampling rate for convenience
s = resample(s,16000,fs);

% Compute the complex short-time Fourier transform (stft) of the signal, and the magnitude spectrogram from it,
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

% Recompose the music 
N = notes;
W = pinv(N) * music ; % w = current weight
M = N*W;

%% Step 1: Error calculation
% F = dimensionality of M
F = size(M,2);
% T = number of frames/time-steps
T = 16000;

% calulcate E
E = 1/(F*T) * norm(music-N*W, 'fro')^2;
% derivative of E
Edif = 2 * norm(music-N*W, 'fro');

%% Step 2: Gradient Descent
%Call the gradDescent function - ensure max_iter = 500

w0 = ones(size(W));
[final_weight01, final_weight0001] = gradDescent(w0,Edif,500,T, F);
plot(1:n,E)


%% Step 3: Reconstruction (optional)

