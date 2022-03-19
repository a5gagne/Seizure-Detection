function Mdl = train_model[features]

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
    accuracy = (C(1,1)+C(2,2))/sum(sum(C)); % accuracy of model on test set
    disp(accuracy);
    
end