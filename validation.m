function performanceMetrics = validation(seizureMarker_auto, seizureGT)
    window = 512;
    
    % fit to window size
    GT = zeros(1,ceil(length(seizureGT)/window)*window);
    GT(1:length(seizureGT)) = seizureGT.';

    % Divide GT and predicted labels into windows (each column = 1 window)
    GT_window = reshape(GT,window,[]);

    % Determine label for entire window
    GT_window = ceil(mean(GT_window));

    C = confusionmat(GT_window,seizureMarker_auto(1:length(GT_window)));
    performanceMetrics = 2*C(1,1)/(2*C(1,1)+C(1,2)+C(2,1)); % F1 score
    
    accuracy = (C(1,1)+C(2,2))/sum(sum(C))*100;
    specificity = 100*C(1,1)/(C(1,1)+C(1,2));
    sensitivity = 100*C(2,2)/(C(2,2)+C(2,1));
    precision = 100*C(2,2)/(C(2,2)+C(1,2));
end