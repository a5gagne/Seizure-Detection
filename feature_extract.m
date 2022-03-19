function Mdl = feature_extract

    sampleRate = 256;  % sampling rate (needed for time conversion)
    window = 512;
    slide = 128;
 

    filtered{20,24} = [];
    eeg{20,2} = [];
    features{4,21} = [];

    for j = 0 : 20 
        load(['EEG_subject0',num2str(j,'%02i'),'.mat']);
        load(['seizureGT_subject0',num2str(j,'%02i'),'.mat']);
            for i = 0 : 22
                buffer = EEG(i+1).ch;
                wn=[0.5 33]/(sampleRate/3);
                [b,a] = butter(2,wn,'bandpass');
                filtered{j+1,i+1} = filter(b,a,buffer);
            end
        filtered{j+1,24} = seizureGT;
    end

    for j = 0 : 20
       eeg{j+1,1} = mean([filtered{j+1,1:23}].'); 
       eeg{j+1,2} = filtered{j+1,24};
    end

    %% moving window and pad signal

    for k = 0 : 20

        buffer = [eeg{k+1,1}].'; % load signal and GT into vectors
        GT = [eeg{k+1,2}];

        % pads the start and end of the array with copy of signal
        padded = padarray(buffer,ceil(window/2),'replicate');
        GT = padarray(GT,ceil(window/2),'replicate');

        % number of windows used to span signal
        steps = (length(padded)-window)/slide; 

        % allocate feature arrays
        peak_f = zeros(1,ceil(steps));
        avg_f = zeros(1,ceil(steps));
        entropy = zeros(1,ceil(steps));
        indicator = zeros(1,ceil(steps));
        j=1; % window number

            %% FEATURE EXTRACTION
            for i = 0 : slide : (steps*slide)

            % PEAK FREQUENCY
            [psd,f] = pwelch(padded(i+1:i+window),[],[],[],sampleRate); % psd of window
            [~,loc] = max(psd); % finds largest peak in the psd
            peak_f(j) = f(loc); % peak frequency

            % MEAN FREQUENCY
            avg_f(j) = meanfreq(psd,sampleRate); % mean frequency of psd

            % ENTROPY
            p = hist(padded(i+1:i+window)); % probability distribution
            p = p/sum(p); % calculate probabilities
            entropy(j) = -sum(p.*log2(p)); % entropy of window

            % INDICATOR
            indicator(j) = ceil(mean(GT(i+1:i+window)));

            j = j + 1; % window number

            end

        % put features in cell array    
        features{1,k+1} = peak_f.'; 
        features{2,k+1} = avg_f.';
        features{3,k+1} = entropy.';
        features{4,k+1} = indicator.';

    end
    
    % Plot frequency response of filter
    t = (0:1:(length(EEG(1).ch)-1))/sampleRate; % create time variable
    figure;
    freqz(b, a, 512, sampleRate);
    title('Frequency Response of Butterworth filter');
    
    % Plot filteref/unfiltered channel
    figure;
    subplot(2,1,1);
    plot(t,EEG(10).ch); % plots unfiltered channel
    legend('Unfiltered channel');
    axis tight;
    subplot(2,1,2);
    plot(t,filtered{21,11}); % plots filtered channel
    legend('Filtered channel');
    xlabel('Time (s)');
    axis tight;
    
    % Plot averaged signal
    figure;
    plot(t,eeg{10,1}); % plot synchronized average
    axis tight;
    xlabel('Time (s)');
    
    %% TRAIN KNN MODEL

% combine features of each subject in columns
peak_f = vertcat(features{1,:}); % combines features in single column
avg_f = vertcat(features{2,:});
entropy = vertcat(features{3,:});
indicator = vertcat(features{4,:});

% create feature array by concatenating features horizontally
data = horzcat(peak_f,avg_f,entropy);

% Create data partitions (80% training, 20% testing)
cvp = cvpartition(length(data),'Holdout',0.2);

Xtrain = data(training(cvp),:); % Training set indices
Ytrain = indicator(training(cvp),:);

Xtest = data(test(cvp),:); % Test set indices
Ytest = indicator(test(cvp),:);

% train KNN model using training set
Mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',5);

% Test model 
labels = predict(Mdl,Xtest); % model generated labels on test set

C = confusionmat(Ytest,labels); % confusion matrix of test set
accuracy = (C(1,1)+C(2,2))/sum(sum(C))*100
F1score = 200*C(2,2)/(2*C(2,2)+C(2,1)+C(1,2))
specificity = 100*C(1,1)/(C(1,1)+C(1,2))
sensitivity = 100*C(2,2)/(C(2,2)+C(2,1))
precision = 100*C(2,2)/(C(2,2)+C(1,2))

end