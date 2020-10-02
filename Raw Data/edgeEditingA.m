function edgeEditing(city)

path = strcat('/data/', city);

Eimport = shaperead(strcat(path, '/edges/Medium/edges.shp'));
from = convertData({Eimport.from}.');
to = convertData({Eimport.to}.');
length = convertData({Eimport.length}.');
lanes = convertData({Eimport.lanes}.');

T = table(from, to, length);

Htype = {Eimport.highway}.';

% Speed Limits in km/h for Porto
% Speed Limits in mph for NYC
if strcmp(city, 'porto') == 1
    motorway = 120;
    primary = 50;
    secondary = 50;
    tertiary = 50;
    residential = 50;
    trunk = 90;
    unclassified = 90;
    units = 'km/h';
elseif strcmp(city, 'nyc') == 1
    motorway = 45;
    primary = 25;
    secondary = 25;
    tertiary = 25;
    residential = 25;
    trunk = 25;
    unclassified = 25;
    units = 'mph';
end

speed_limit = zeros(height(T), 1);

% Assign speed limits
Fmotorway = strcmp(Htype, 'motorway') == 1 | strcmp(Htype, 'motorway_link') == 1;
Fprimary = strcmp(Htype, 'primary') == 1 | strcmp(Htype, 'primary_link') == 1;
Fsecondary = strcmp(Htype, 'secondary') == 1 | strcmp(Htype, 'secondary_link') == 1;
Ftertiary = strcmp(Htype, 'tertiary') == 1 | strcmp(Htype, 'tertiary_link') == 1;
Ftrunk = strcmp(Htype, 'trunk') == 1 | strcmp(Htype, 'trunk_link') == 1;
Fresidential = strcmp(Htype, 'residential') == 1 | strcmp(Htype, 'living_street') == 1;
Funclassified = strcmp(Htype, 'unclassified') == 1;

speed_limit(Fmotorway) = motorway;
speed_limit(Fprimary) = primary;
speed_limit(Fsecondary) = secondary;
speed_limit(Ftertiary) = tertiary;
speed_limit(Ftrunk) = trunk;
speed_limit(Fresidential) = residential;
speed_limit(Funclassified) = unclassified;

% Manually assign speed limits to multi-classified edges
keep = Fmotorway | Fprimary | Fsecondary | Ftertiary | Ftrunk | Fresidential | Funclassified;
remainingHtypes = unique(Htype(~keep));

for ii = 1:size(remainingHtypes, 1)
    Y = string(remainingHtypes(ii));
    S = sprintf('Input the speed limit (in %s) for %s: \n', units, Y);
    Q = input(S);
    F = strcmp(Htype, Y) == 1;
    speed_limit(F) = Q;
end
    
% Convert to m/s
if strcmp(city, 'porto') == 1
    speed_limit = speed_limit*1000/(60*60);
elseif strcmp(city, 'nyc') == 1
    speed_limit = speed_limit*1609.34/(60*60);
end

lanes(isnan(lanes)) = 1;

T = addvars(T, speed_limit, lanes);

fname = strcat(path, '-edge-infoMedium.csv');
writetable(T, fname);

end

function [out] = convertData(in)

N = size(in);
out = zeros(N);

for ii = 1:N(1)
    for jj = 1:N(2)
        out(ii, jj) = str2double(in(ii,jj));
    end
end

end
