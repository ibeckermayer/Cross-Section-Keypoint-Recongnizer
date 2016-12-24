function [pts] = readPoints(image, n)
%readPoints   Read manually-defined points from image
%   POINTS = READPOINTS(IMAGE) displays the image in the current figure,
%   then records the position of each click of button 1 of the mouse in the
%   figure, and stops when another button is clicked. The track of points
%   is drawn as it goes along. The result is a 2 x NPOINTS matrix; each
%   column is [X; Y] for one point. cell is the variable where points will
%   be saved and sav returns whether or not you chose to save the points
% 
%   POINTS = READPOINTS(IMAGE, N) reads up to N points only.
% http://de.mathworks.com/matlabcentral/answers/118724-how-to-get-onclick-coordinate-pixel-value-and-location-from-an-image
if nargin < 2
    n = Inf;
    pts = zeros(2, 0);
else
    pts = zeros(2, n);
end
imshow(image);     % display image
MaximizeFigureWindow()
xold = 0;
yold = 0;
k = 0;
hold on;           % and keep it there while we plot
while 1
    [xi, yi, but] = ginput(1);      % get a point
    if ~isequal(but, 1)             % stop if not button 1
        break
    end
    k = k + 1;
    pts(1,k) = xi;
    pts(2,k) = yi;
      if xold
          plot([xold xi], [yold yi], 'go');  % draw as we go
      else
          plot(xi, yi, 'go');         % first point on its own
      end
      if isequal(k, n)
          break
      end
      xold = xi;
      yold = yi;
      
  end
hold off;
pause(1)
close all
if k < size(pts,2)
    pts(:,k+1:size(pts,2)) = NaN;
end

end
% NOTE= a = imread(filename), a = a', a = a(:)' for row sorted vector of
% image

% NOTE: arrange data in cell, then: 
% for row = 1:size(a,1)
% fprintf(fid, '%d,%d,%d,%d,', a{row,1:end-1})
% fprintf(fid,repmat('%d ',1,length(a{end})-1),a{row,end}(1:end-1))
% fprintf(fid,'%d\n',a{row,end}(end))
% end
% 

%