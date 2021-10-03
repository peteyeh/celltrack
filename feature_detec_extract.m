tstack = Tiff('RFP-1.tif');
% [I,J] = size(tstack.read()); 
% K = length(imfinfo('BF-1.tif'));
I = tstack.read(); %gets the data of the current IFD (the first image)

tstack2 = Tiff('BF-1.tif');
original_tif = tstack2.read();

% points = detectKAZEFeatures(I);
% imshow(I);
% hold on;
% plot(selectStrongest(points,20));
% hold off;


% [N, edges] = histcounts(I);


I1 = double(I);

I1 = I1 - min(I1(:));
I1 = I1 / max(I1(:));
figure(); imshow(I1);

[counts, x] = imhist(I1);
thresh = graythresh(I1);

indicesI1 = find(I1 < 2.5*thresh);
indicesI1_2 = find(I1 > 2.5*thresh);
I1(indicesI1) =  1;
I1(indicesI1_2) = 0;

figure();
imshow(I1);

[row,col] = find(I1 == 0);

y = row;
x = col;

data = [x,y];

%density-based
[idx, corepts] = dbscan(data, 6, 1);
figure(); hold on;
gscatter(data(:,1),-data(:,2),idx);

mean_x_coordinates = []; %store for each observation/cluster
mean_y_coordinates = []; %same as line above

for i = 1:max(idx)
    index = find(idx == i); %returns a vector of the values of the observations that fall under this cluster no.
    x_values = []; %to store x coordinates for each observation
    y_values = []; %to store y coordinates for each observation
    for j= 1:length(index) %iterates through each observation
        observation_no = index(j); %extracts the observation value we are on
        x_values = [x_values; data(observation_no,1)];
        y_values = [y_values; data(observation_no,2)];
    end
    
    centroid_x = mean(x_values);
    centroid_y = mean(y_values);
    
    centroid_x = round(centroid_x);
    centroid_y = round(centroid_y);
    
    if ((centroid_x + 51) > 1128)
        continue; 
    end
    
    if ((centroid_x - 51) <= 0)
        continue;
    end
    
    if ((centroid_y + 51) > 832)
        continue;
    end
    
    if ((centroid_y - 51) <= 0)
        continue;
    end
    
    mean_x_coordinates = [mean_x_coordinates; centroid_x]; %stores x coordinate of centroid for each cluster
    mean_y_coordinates = [mean_y_coordinates; centroid_y]; %same for y
    
%     max_x = max(x_values); %for bounding box column borders
%     min_x = min(x_values);
%     
%     max_y = max(y_values); %for bounding box
%     min_y = min(y_values);
    
    %%Crop for bounding box
    
    ColStart = centroid_x - 50;
    ColEnd = centroid_x + 50;
    
    RowStart = centroid_y - 50;
    RowEnd = centroid_y + 50;
    
    newRow = RowEnd-RowStart+1;
    newCol = ColEnd-ColStart+1;
    
    ITemp = zeros(100,100)
    ITemp(1:newRow, 1:newCol) = original_tif((RowStart:RowEnd),(ColStart:ColEnd));
    
    figure(); hold on;
    ITemp = double(ITemp);
    ITemp = ITemp - min(ITemp(:));
    ITemp = ITemp / max(ITemp(:));
    figure(); 
    imshow(ITemp);
    
    H = fspecial('log');
%     log_ITemp = imfilter(ITemp, H, 'replicate');
    log_ITemp = imfilter(ITemp,H,'symmetric', 'conv');
    figure(); 
    imshow(log_ITemp);
    
    %points = detectSURFFeatures(log_ITemp);
    
    %% Visualizing HOG Features
    
    %Extract the HOG features and the HOG visualization properties
    [hog_2x2, vis2x2] = extractHOGFeatures(log_ITemp, 'CellSize', [2 2]);
    [hog_4x4, vis4x4] = extractHOGFeatures(log_ITemp, 'CellSize', [4 4]);
    [hog_8x8, vis8x8] = extractHOGFeatures(log_ITemp, 'CellSize', [8 8]);
    
    %Plots
    figure;
    subplot(2,3,1:3); imshow(log_ITemp);
    
    subplot(2,3,4);
    plot(vis2x2);
    title('CellSize = [2 2]');
    
    subplot(2,3,5);
    plot(vis4x4);
    title('CellSize = [4 4]');
    
    subplot(2,3,6);
    plot(vis8x8); 
    title('CellSize = [8 8]');
    
    %%
     
%     ColStart = min_x;
%     ColEnd = max_x;
%     
%     RowStart = min_y;
%     RowEnd = max_y;
%     
%     newRow = RowEnd - RowStart +1;
%     newCol = ColEnd - ColStart +1;
%     
%     ITemp = zeros(newRow, newCol);
%     ITemp(1:newRow, 1:newCol) = I1((RowStart:RowEnd), (ColStart:ColEnd));
%     
%     figure(); hold on;
%     imshow(ITemp);
    
end



%kmeans
% 
% [idx,C] = kmeans(data,54);
% figure(); hold on;
% gscatter(data(:,1),-data(:,2),idx);
% 
% mean_x_coordinates = []; %store for each observation/cluster
% mean_y_coordinates = []; %same as line above
% 
% for i = 1:max(idx)
%     index = find(idx == i);
%     x_values = [];
%     y_values = [];
%     for j= 1:length(index)
%         observation_no = index(j);
%         x_values = [x_values; data(observation_no,1)];
%         y_values = [y_values; data(observation_no,2)];
%     end
%     
%     centroid_x = mean(x_values);
%     centroid_y = mean(y_values);
%     
%     mean_x_coordinates = [mean_x_coordinates; mean(x_values)];
%     mean_y_coordinates = [mean_y_coordinates; mean(y_values)];
%     
% end





%[idx,C] = kmeans(data,54)
% figure; hold on;
% 
% imshow(I1);

% plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3);

% imhist(I1)
% 
% imtool(I1)



