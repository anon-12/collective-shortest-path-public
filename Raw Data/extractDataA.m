function extractData(city)

startTime = 7;
endTime = 19;

% Get bounding box
if strcmp(city, 'porto') == 1
    % Distance = 1.5km
    bbox = [41.15906768713956, 41.132044280933165, -8.592458475590403, -8.628207578835637];
    % Distance = 3km
    %bbox = [41.17257934234985, 41.11853252994112, -8.574584008103125, -8.646082209600989];
    north = bbox(1);
    south = bbox(2);
    west = bbox(4);
    east = bbox(3);
    load('/Datasets/Porto Taxis/PortoTaxiLatLongs.mat')
    fname = '/Collective Shortest Paths/porto-data.csv';
    T = PortoTaxiLatLongs;
elseif strcmp(city, 'nyc') == 1
    % Distance = 1.5km
    bbox = [40.76414884063925, 40.737127126787776, -73.9761330283043, -74.01166510987719];
    % Distance = 3km
    %bbox = [40.777659648702645, 40.72361622100542, -73.9583671972282, -74.02943135549766];
    north = bbox(1);
    south = bbox(2);
    west = bbox(4);
    east = bbox(3);
    load('/Datasets/NYCtaxisOct14.mat')
    fname = '/Collective Shortest Paths/nyc-data.csv';
    T = NYCtaxisOct14;
end

% Filter by position
startNS = T.start_lat < north & T.start_lat > south;
startEW = T.start_long < east & T.start_long > west;
endNS = T.end_lat < north & T.end_lat > south;
endEW = T.end_long < east & T.end_long > west;
keepPosition = startNS & startEW & endNS & endEW;

% Filter by time
startTimeDecimal = startTime/24;
endTimeDecimal = endTime/24;
if strcmp(city, 'porto') == 1
    T.date_datenum_start = T.date_datenum;
    T.date_time_start = T.date_time;
    T.date_datenum_end = T.date_datenum_start + T.duration/(60*60*24);
    T.date_time_end = T.date_datenum_end - floor(T.date_datenum_end);
end
keepSameDay = floor(T.date_datenum_start) == floor(T.date_datenum_end);
keepTime = T.date_time_start > startTimeDecimal & T.date_time_end < endTimeDecimal ...
            & T.duration < 60*60*(endTime-startTime) & keepSameDay;

% Filter by day
if iscategorical(T.weekend) == 1
    T.weekend = double(T.weekend) - 1;
end
keepWeekend = T.weekend == 0;

% Filter
keep = keepPosition & keepTime & keepWeekend;
T = T(keep, :);

% Calculate distance between points
T.dist_between = vect_haversine([T.start_long, T.start_lat], [T.end_long, T.end_lat]);

% Determine time after designated start
T.timeafterstart = round((T.date_time_start - startTime/24)*24*60*60);

% Keep necessary columns
Texp = T(:, {'start_long', 'start_lat', 'end_long', 'end_lat', 'dist_between', 'duration', 'timeafterstart'});

writetable(Texp, fname);

end

function [km] = vect_haversine(loc1, loc2)

% Sub-function to calculate the Haversine distance between two longitude
% and latitude points
%
%  INPUT - loc1, loc2 - two locations in the format [longitude, latitude]
%
% OUTPUT - km - Haversine distance between the two points in kilometres
%
% Adapted from: https://www.mathworks.com/matlabcentral/fileexchange/27785-distance-calculation-using-haversine-formula

% Convert from decimals to radians
loc1 = loc1 .* (pi/180);
loc2 = loc2 .* (pi/180);

% Begin calculation
R = 6371;                             % Earth's radius in km
delta_lat = loc2(:,2) - loc1(:,2);        % difference in latitude
delta_lon = loc2(:,1) - loc1(:,1);        % difference in longitude
a = sin(delta_lat/2).^2 + cos(loc1(:,2)) .* cos(loc2(:,2)) .* ...
    sin(delta_lon/2).^2;
c = 2 * atan2(sqrt(a), sqrt(1-a));
km = R * c;                           % distance in km

end