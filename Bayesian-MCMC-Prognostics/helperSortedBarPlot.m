function sortedIdx = helperSortedBarPlot(tbl, ylbl)
% HELPERSORTEDBARPLOT helper function to create sorted bar plot

%  Copyright 2018 The MathWorks, Inc.
[~, sortedIdx] = sort(tbl{1,:}, 'descend');
tblSorted = tbl(:, sortedIdx);
figure
bar(tblSorted{1,:})
xticks(1:size(tblSorted,2))
xticklabels(tbl.Properties.VariableNames(sortedIdx))
xtickangle(45)
ylabel(ylbl)
end