function data = helperReadData(filename, variables)
% Read data variables for the fileEnsemble
%
% Inputs:
% filename  - a string of the file name to read from.
% variables - a string array containing variable names to read.
%             It must be a subset of DataVariables specified
%             in fileEnsembleDatastore.
% Output:
% data      - return a table with a single row

% Copyright 2017-2018 The MathWorks, Inc.

data = table;
mfile = matfile(filename); % Allows partial loading
for ct = 1:numel(variables)
    if strcmp(variables{ct}, 'Date')
        % Extract the datetime information from the file names
        % as the independent variable of the ensemble datastore
        data = table;
        [~, fname] = fileparts(filename);
        %token = regexp(fname, 'data-(\w+)', 'tokens');
        token = regexp(fname, 'c_4_(\w+)', 'tokens');
        %data.Date = datetime(token{1}{1}, 'InputFormat', 'yyyyMMdd''T''HHmmss''Z''');
        nameTmp = char(token{1}{1});%分割之后是cell类型，需要转换为char类型
        nameTmp=str2num(nameTmp);
        data.Date = nameTmp(1:length(nameTmp)); % fileName 结果为   001-315
    else
        val = mfile.(variables{ct});
        % Convert non-scalar values into a single cell
        if numel(val) > 1
            val = {val};
        end
        data.(variables{ct}) = val;
    end
end
end