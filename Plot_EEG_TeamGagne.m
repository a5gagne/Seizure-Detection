window = 512;
sampleRate = 256;
Mdl = feature_extract();
%load('Mdl.mat');

name = 'subject019.mat'; % Subject
load(['EEG_',name]); % Load EEG data
load(['seizureGT_',name]); % Load Seizure Ground truth data
test = mean([EEG.ch].'); % averages channels of test data

[seizureMarker_auto] = classification(test,Mdl); % classifies using trained model

performanceMetrics = validation(seizureMarker_auto, seizureGT); % results


%% plot auto marker and ground truth

seizure_window = zeros(1,length(seizureGT));
seizure = find(seizureMarker_auto); % finds seizure locations

% expand marker to entire window 
for j = 1 : length(seizure)
    % makes entire window 1
    seizure_window(seizure(j)*window:((seizure(j)+1)*window)-1) = 1; 
end
t = (0:1:(length(EEG(1).ch)-1))/sampleRate; % create time variable
figure;
subplot(2,1,1);
plot(t,EEG(10).ch);
hold on;
plot(t,seizureGT*1000);
axis tight;
xlabel('Time (s)');
legend('EEG Channel 10','Ground Truth');

subplot(2,1,2);
plot(t,EEG(10).ch);
hold on;
plot(t,seizure_window*1000);
axis tight;
xlabel('Time (s)');
legend('EEG Channel 10','Auto Marker');


%% plot 3 channels of 1 patient to show similarity
figure;
subplot(3,1,1);
plot(t,EEG(10).ch);
axis tight;
legend('Channel 10');
subplot(3,1,2);
plot(t,EEG(14).ch);
axis tight;
legend('Channel 14');
subplot(3,1,3);
plot(t,EEG(6).ch);
axis tight;
legend('Channel 6');
xlabel('Time (s)');