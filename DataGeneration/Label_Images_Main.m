% Click left top, middle top, right top, middle bottom, left bottom, right
% bottom

while direc_not_empty('PreProcessedImages/') == 1; %!!!
    load('trainingData.mat'); %!!!
    files = dir('PreProcessedImages/*.jpg'); %!!!
    
    if ~isnumeric(trainingData{end,1}) % Sample #1 for the first entry
        sample_number = 1;
    else
        sample_number = trainingData{end,1}+1; % Otherwise make it the next available integer
    end
    
    f = ['PreProcessedImages/' files(1).name]; % get filename !!!
    
    pts = readPoints(f,6); % mark the points
    
    sav = input('Save? [y/n]: ','s'); % Save?
    
    while sav == 'n'
        d_or_r = input('Delete or Retry? [d/r]: ','s');
        
        if d_or_r == 'd'
            delete(f);
            sav = 'cont';
        end
        
        if d_or_r == 'r'
            pts = readPoints(f,6); % mark the points
            sav = input('Save? [y/n]: ','s'); % Save?
        end
    end
        
    
    if sav == 'y' %!!!
        row = size(trainingData,1)+1;
        trainingData{row,1} = sample_number;
        trainingData{row,2} = pts(1,1);
        trainingData{row,3} = pts(2,1);
        trainingData{row,4} = pts(1,2);
        trainingData{row,5} = pts(2,2);
        trainingData{row,6} = pts(1,3);
        trainingData{row,7} = pts(2,3);
        trainingData{row,8} = pts(1,4);
        trainingData{row,9} = pts(2,4);
        trainingData{row,10} = pts(1,5);
        trainingData{row,11} = pts(2,5);
        trainingData{row,12} = pts(1,6);
        trainingData{row,13} = pts(2,6);
        im = imread(f);
        im = im';
        im = im(:)';
        trainingData{row,14} = im;
        movefile(f, ['LabeledImages/' num2str(sample_number) '.jpg']);
        save('trainingData.mat', 'trainingData'); %!!!
        clear trainingData;
    end
    if sav == 'cont'
        continue
    end
end   
        
        

