% Compare spikesorted spiketimes to our spiketimes
myKsDir = "/Users/johnduva/Desktop/2020/Git/MLSP/Project/Kilosort_Output/";
[~, ~, ~, spikeSites] = ksDriftmap(myKsDir); % this function lives in 'spikes' from 
spikeTimes = readNPY('spike_times.npy');
all = [spikeTimes, spikeSites];

% our time interval is 113,000 to 116,500
all2 = [];
for spiketime = 1 : length(all)
    if all(spiketime,1)>=113000 && all(spiketime,1)<=116500
        all2 = [all2; all(spiketime, :)];
    end
end

% channels is 70 to 110
all3 = [];
for i = 1 : size(all2, 1)
    if all2(i,2)>=70 && all2(i,2)<=110
        all3 = [ all3; all2(i,:) ];
    end
end

x = linspace(113000,116500, 3501)';

y = zeros(length(x), 1);
for frame = 1 : length(x) %3500
    for spike = 1 : size(all3, 1) %744
        if x(frame) == all3(spike,1)
            y(frame,1) = 1;
        else
            continue
        end
    end
end

%% Plot figure
load('vOutput.mat')

subplot(2,1,1)
plot(1:3501, y)
title("KiloSort Spikes")

subplot(2,1,2)
plot(1:3501, Voutput)
title("Vedant's Spikes")




