% First, run post_process_npz2mat.py (to convert npz to mat files)
clc, clear, close all

%%
current_DIR = pwd;
matFile = [current_DIR, '/input_output/outputs/wood/N882A6_ch2_main_20221012110243_20221012110912.mp4_tracks_and_visibility.mat'];
jsonFilePath = [current_DIR, '/input_output/outputs/wood/marker_detection_Lorex_N882A6_ch2_main_20221012110243_20221012110912.json'];
excelFilePath = [current_DIR, '/input_output/outputs/wood/displacement response.xlsx'];



% filter parameter
windowSize = 5;  %5  % Adjust as needed


%% %%


% matFile = 'N882A6_ch2_main_20221012110243_20221012110912.mp4_tracks_and_visibility.mat';
load(matFile)



tracks = double(tracking_data.tracks(:,:,1));


tracks(:, :) = tracks(:, :) - tracks(1, :); % Subtract the first row from all other rows

visibility = tracking_data.visibility;
clear tracking_data


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% remove lost dots (columns0
columns_to_remove = [];

for dot = 1:size(tracks,2)

    if all(visibility(end-100:end,dot)==0)
        columns_to_remove(end+1) = dot;
    end
end

visibility(:, columns_to_remove) = [];
tracks(:, columns_to_remove) = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% remove other outliers
outlier_sensors = [];

columns_to_remove = [];
for from_end = [0, 100, 200, 500, 750, 1000]

    row = size(tracks, 1)-from_end;

    % Loop over columns
    row_median = median(tracks(row, :));

    threshold = 0.05 * row_median;

    % Find indices of columns above and below the threshold
    above_threshold_indices = find(tracks(row, :) > (row_median + threshold));
    below_threshold_indices = find(tracks(row, :) < (row_median - threshold));

    % Combine indices of columns to remove
    columns_to_remove = [columns_to_remove, above_threshold_indices, below_threshold_indices];
end
columns_to_remove = unique(columns_to_remove);

% Remove columns from the tracks matrix
tracks(:, columns_to_remove) = [];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filter the signals

for dot = 1:size(tracks,2)
    polynomialOrder = 3;
    tracks(:, dot) = sgolayfilt(tracks(:, dot), polynomialOrder, windowSize);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tracks in time
fps = 30;
tracks_t0 = datetime('11:05:29', 'Format', 'HH:mm:ss'); %start time
tracks_t = tracks_t0 + seconds(0:length(tracks)-1)/fps;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the path to the JSON file
% jsonFilePath = 'marker_detection_Lorex_N882A6_ch2_main_20221012110243_20221012110912.json';

% Read JSON data
jsonData = jsondecode(fileread(jsonFilePath));

% Initialize variables
allCornersForID959 = [];    % Initialize an empty matrix for averaged corner data
markerId = '959';           % Marker ID to search for
allTimesForID959 = {};      % Initialize an empty cell array to store corresponding times as strings

% Search through the JSON data for the specific marker ID
for i = 1:length(jsonData)
    currentData = jsonData{i};  % Access the struct in a cell
    ids = currentData.ids;      % Access ids directly as an array

    % Loop through each ID to check if it matches the markerId
    for j = 1:numel(ids)        % Use numel in case ids is a matrix
        idStr = num2str(ids(j));  % Convert each id to string
        if strcmp(idStr, markerId)  % Compare the current ID to markerId
            % If a match is found, process and store the corresponding corners
            if ~isempty(currentData.corners) && size(currentData.corners, 1) >= j
                currentCorners = squeeze(currentData.corners(j, :, :));
                meanCorners = mean(currentCorners, 1);
                allCornersForID959 = [allCornersForID959; meanCorners];  % Append as a new row
                allTimesForID959{end+1} = currentData.time;  % Store time in cell array
            end
        end
    end
end

% Assume allTimesForID959 is already a cell array of strings in proper format, e.g., 'HH:mm:ss'
allTimesForID959 = datetime(allTimesForID959, 'InputFormat', 'HH:mm:ss');
allCorners=allCornersForID959(:,1)-allCornersForID959(1,1);


% % filter the signals
%
% for dot = 1:size(allCorners,2)
%     windowSize = 51;  %5  % Adjust as needed
%     polynomialOrder = 3;
%     allCorners(:, dot) = sgolayfilt(allCorners(:, dot), polynomialOrder, windowSize);
%
% end



%%%%%%%%%%%%%%%%%%%%%

% Define the path to the Excel file
% excelFilePath = 'displacement response.xlsx';

% Specify import options for reading the Excel file
opts = detectImportOptions(excelFilePath, 'NumHeaderLines', 6); % Skips the first 7 header rows

% % Display all variable names detected in the file
% disp('Detected variable names:');
% disp(opts.VariableNames);

% Uncomment the following lines after confirming the correct variable name
% Set the variables to read
opts.SelectedVariableNames = {'SP19_mm_'};

% Read the specified data from the Excel file
tableData = readtable(excelFilePath, opts);

% Extract the data for plotting
dispData = tableData.('SP19_mm_');

% % Optional: Customize axes and grid
% grid on;
% ax = gca;
% ax.GridLineStyle = '--';
% ax.GridColor = [0.5, 0.5, 0.5]; % Set grid color to grey
% ax.GridAlpha = 0.6; % Set transparency of grid lines

% Define start time
startTime = datetime('11:04:10', 'Format', 'HH:mm:ss');

% Define the time step in seconds
timeStep = seconds(0.005);

% Number of time steps
numSteps = length(dispData);

% Generate time vector
timeVector = startTime + (0:numSteps-1) * timeStep;

%%%%%%%%%%
scale= 4.067;

% figure(1)
% plot(scale*tracks,'DisplayName','tracks')
% hold on
%
% plot( scale*median(tracks,2), 'LineWidth',1.5,'Color','r')
% xlabel('Time');
% ylabel('Displacement(mm)');
% legend('Segmentation');
%
% figure(2)
% plot(allTimesForID959,scale*allCorners,'LineWidth',1.5,'Color','b')
% xlabel('Time');
% ylabel('Displacement(mm)');
% legend('Lorex Camera');
%
% figure(3)
% plot(timeVector,-dispData, 'LineWidth', 1.5, 'Color', 'g')
% xlabel('Time');
% ylabel('Displacement(mm)');
% legend('SP19[mm]');





figure(4)
set(figure(4), "Position",[100 100 1500 700])
plot(allTimesForID959,scale*allCorners,'LineWidth',1.5,'Color','b', 'DisplayName','Lorex')
hold on
plot(timeVector,-dispData, 'LineWidth', 1.5, 'Color', 'g', 'DisplayName','SP19')

plot(tracks_t,3.3*1.28*median(tracks,2), 'LineWidth', 1.5, 'Color', 'r', 'DisplayName','Segmnentation')


xlabel('Time');
ylabel('Displacement(mm)');
legend()

xlim([datetime('11:05:00', 'Format', 'HH:mm:ss'), datetime('11:08:00', 'Format', 'HH:mm:ss')]);
