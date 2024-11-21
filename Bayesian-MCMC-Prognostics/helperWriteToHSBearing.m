function helperWriteToHSBearing(filename, data)
% Write data to the fileEnsemble
% Inputs:
% filename - a string of the file name to write to.
% data     - a structure
save(filename, '-append', '-struct', 'data');
end