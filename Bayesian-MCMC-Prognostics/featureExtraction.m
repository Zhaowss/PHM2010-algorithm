function [features] = featureExtraction(datastore, SelectedVariables, fs)
    % 选择要读取的信号通道的名称
    datastore.SelectedVariables=SelectedVariables;
    reset(datastore);
    % hasdata determine whether data is avaiable to read
    features=table;
    while hasdata(datastore)
        data = read(datastore);
        channel_temp=eval(['data.' char(SelectedVariables(2))]);
        channel_selected = channel_temp{1};
        table_temp = table;

        % Time Domain Features
        table_temp.Mean = mean(channel_selected);
        table_temp.Std = std(channel_selected);
        table_temp.Skewness = skewness(channel_selected);
        table_temp.Kurtosis = kurtosis(channel_selected);
        table_temp.Peak2Peak = peak2peak(channel_selected);
        table_temp.RMS = rms(channel_selected);
        table_temp.CrestFactor = max(channel_selected)/table_temp.RMS;
        table_temp.ShapeFactor = table_temp.RMS/mean(abs(channel_selected));
        table_temp.ImpulseFactor = max(channel_selected)/mean(abs(channel_selected));
        table_temp.MarginFactor = max(channel_selected)/mean(abs(channel_selected))^2;
        table_temp.Energy = sum(channel_selected.^2);

         % Compute spectral kurtosis with window size = 128
        wc = 128;
        [SK, F] = pkurtosis(channel_selected, fs, wc);

        % 4 Spectral Kurtosis related features
        table_temp.SKMean = mean(SK);
        table_temp.SKStd = std(SK);
        table_temp.SKSkewness = skewness(SK);
        table_temp.SKKurtosis = kurtosis(SK);

        % store the extracted featurs in each loop
        features=[features; table_temp];
        clear channel_selected SK F table_temp;

        % write the derived features to the corresponding file
    %     writeToLastMemberRead(hsbearing, features); 
    end
    X=sprintf(['finished ' char(SelectedVariables(2)) ' feature extraction']);
    disp(X);
end

