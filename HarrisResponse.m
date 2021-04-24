file = '/Users/madsrossen/Documents/4. Semester/Projekt/code/CornersFound.txt'


Corner = importdata(file, ',', 1) % The "1" means that there is one header 

% Plot the image gradient of the image
scatter(Corner.data(:, 3),Corner.data(:, 4),'filled')
ax = gca;

ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

ax.YLabel.String = 'Iy';
ax.YLabel.FontSize = 20;
ax.XLabel.String = 'Ix';
ax.XLabel.FontSize = 20;

ax.XLim = [-20 100];
ax.YLim = [-20 100];

grid(ax, 'on')
hold off

% New plot
% Following is used to determine the noise in the image

xList = 1:1:998;
xListFrom1 = 1:1:998;

plot(xListFrom1, Corner.data(xList, 5))
hold on
for y = 0:1:996
    xList = xList+998;
    s = plot(xListFrom1, Corner.data(xList, 5));
end
hold on
xlabel('x (image coordinate)')
ylabel('R')