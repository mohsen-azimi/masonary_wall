% First, run post_process_npz2mat.py (to convert npz to mat files)


clc, clear, close all


matFile = "input_output/outputs/wood/N882A6_ch2_main_20221012110243_20221012110912.mp4_tracks_and_visibility.mat";
load(matFile)
clear matFile



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
    windowSize = 51;  %5  % Adjust as needed
    polynomialOrder = 3;
    tracks(:, dot) = sgolayfilt(tracks(:, dot), polynomialOrder, windowSize);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


plot(tracks,'DisplayName','tracks')
hold on

plot( median(tracks,2), 'LineWidth',1.5,'Color','r')
