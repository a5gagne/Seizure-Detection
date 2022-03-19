function [test_window] = classification(test,Mdl)

    sampleRate = 256;
    window = 512;
    slide = 128;
    
    % filter
    wn=[0.5 33]/(sampleRate/3);
    [b,a] = butter(2,wn,'bandpass');
    test = filter(b,a,test);

    % pads the start and end of the array with copy of signal
    test = padarray(test.',ceil(window/2),'replicate');

    % number of windows used to span signal
    steps = (length(test)-window)/slide; 

    % allocate feature arrays
    peak_f = zeros(1,ceil(steps));
    avg_f = zeros(1,ceil(steps));
    entropy = zeros(1,ceil(steps));
    j = 1;

        for i = 0 : slide : (steps*slide)

            % PEAK FREQUENCY
            [psd,f] = pwelch(test(i+1:i+window),[],[],[],sampleRate); % psd of window
            [~,loc] = max(psd); % finds largest peak in the psd
            peak_f(j) = f(loc); % peak frequency

            % MEAN FREQUENCY
            avg_f(j) = meanfreq(psd,sampleRate); % mean frequency of psd

            % ENTROPY
            p = hist(test(i+1:i+window)); % probability distribution
            p = p/sum(p); % calculate probabilities
            entropy(j) = -sum(p.*log2(p)); % entropy of window

            j = j + 1; % window number
        end

    %% CLASSIFICATION

    % predicted class array
    class = predict(Mdl,horzcat(peak_f.',avg_f.',entropy.')); 
    
     % fit to window size
     a = zeros(1,round((length(class))/(window/slide))*(window/slide));
     a(1:length(class)) = class.';

    % Divide GT and predicted labels into windows (each column = 1 window)
    test_window = reshape(a,(window/slide),[]);

    % Determine label for entire window
    test_window = ceil(mean(test_window));
    
end
