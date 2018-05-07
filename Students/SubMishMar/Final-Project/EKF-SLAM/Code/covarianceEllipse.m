function [X, Y] = covarianceEllipse(x, P, n, NP)

% input : x - center of the ellipse (mean)
%         P - covariance
%         n - n sigma ellipse
%         NP - number of points to used to plot the ellipse - 1

% output : X - x coordinates of the ellipse
%          Y - y coordinates of the ellipse

alpha = 2*pi*(0:NP)./NP; % NP angle interval from [0, 2pi]
circle = [cos(alpha); sin(alpha)]; % The unit circle

[R, D] = svd(P);

d = sqrt(D);

% n sigma ellipse


ellip = n * R * d * circle;

X = x(1) + ellip(1, :);
Y = x(2) + ellip(2, :);

end