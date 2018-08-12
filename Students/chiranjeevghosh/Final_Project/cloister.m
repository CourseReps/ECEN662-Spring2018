function f = cloister(xmin, xmax, ymin, ymax, n)
% Generates a 2D cloister shape for landmarks

if nargin < 5
    n = 9;
end

% Center of the Cloister
x0 = (xmin + xmax)/2;
y0 = (ymin + ymax)/2;

% Size of the Cloister
hsize = xmax - xmin;
vsize = ymax - ymin;

tsize = diag([hsize vsize]);

% Integer Cordinates of points
outer = (-(n - 3)/2 : (n - 3)/2 );
inner = (-(n - 3)/2 : (n - 3)/2 );

% Outer north coordinates
No = [ outer; (n - 1)/2 * ones(1, numel(outer))];

% Inner North
Ni = [ inner; (n - 3)/2 * ones(1, numel(inner))];

% East (rotate 90 degrees wrt north points)
E = [0 -1;1 0] * [No Ni];

% South and West are negatives of N and E respectively
points = [No Ni E -No -Ni -E];

% Rescale
f = tsize*points/(n - 1);

% Move
f(1, :) = f(1, :) + x0;
f(2, :) = f(2, :) + y0;
end
