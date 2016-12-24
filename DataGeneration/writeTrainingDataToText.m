function writeTrainingDataToText(trainingData,name)

% trainingData is the cell array containing the training data, name is the
% name of the txt file you want to save to.

% Create new file
fid = fopen(name,'w');

% Print header line
fprintf(fid,repmat('%s,',1,size(trainingData,2)),trainingData{1,1:end-1});
fprintf(fid,'%s\n',trainingData{1,end});

for row = 2:size(trainingData,1)
    row
    fprintf(fid,'%d,',trainingData{row,1});
    fprintf(fid,repmat('%.4f,',1,12), trainingData{row,2:end-1});
    fprintf(fid,repmat('%d ',1,length(trainingData{end})-1),trainingData{row,end}(1:end-1));
    fprintf(fid,'%d\n',trainingData{row,end}(end));
end


fclose(fid);