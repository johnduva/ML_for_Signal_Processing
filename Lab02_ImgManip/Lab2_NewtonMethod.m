% SCRIPT TO CREATE A DEMO FOR NEWTON'S METHOD
% Use this as JUST a guide and feel free to write as you wish

% create a function poly.m and write desired equation and return
% independent variable
f = @poly;

% create a function poly_derivative.m and write desired equation and return
% independent variable
fder = @poly_derivative;

maxIters =  200;
tol = 1e-06;
% experiment with different values of xi
xi = -100.0;

% Initialization of relative errors, rel_errs
rel_errs = zeros(maxIters,1);
xr = xi;

% caluculate function values for each value of xlim_values using for loop
f_values=[];
xlim_values=[-abs(xr):0.1:abs(xr)];
% write from here
for xVal = 1: length(xlim_values)
    f_values = [f_values, f(xlim_values(xVal))];
end

% plot the xlim_values vs function values and draw x-axis and y-axis centered at origin
% write your code here
plot(xlim_values, f_values)

% write xr as 'x0' to denote initial point. Use text function to write text on figures
% write from here
figure(1)
plot(xlim_values, f_values)
text(xr,f(xr),'x0')

% plot tangent at xr
% write from here
figure(2)
plot(xlim_values, f_values)
hold on;
dy = fder(xlim_values);
tang = (xlim_values-xr) * fder(xr) + f(xr);
plot(xlim_values,tang)
scatter(xr,f(xr))
text(xr,f(xr),'f(xr)')
ylim([0 inf])
hold off

% find Newtons update and write on the same plot
% write from here
[xr] = newtons_update(f,fder, xi);
figure(3)
hold on;
    plot(xlim_values, f_values)
    plot(xlim_values,tang)
    scatter(xr, f(xr))
    text(xr, f(xr), 'f(xr)')
    text(-100,f(-100),'x0')
    ylim([0 inf])
    
% draw line from xr to f(xr). Use functions text and line
% write from here
    plot(xlim_values, f_values)
    line([xr,xr],[0,f(xr)])
    text(xr,f(xr),'f(xr)')
    text(xr,0,'xr')
hold off

% M is the variable to hold frames of video. Use getframe function
clear M;
count=1;
% write command here and store in M[count]
M(count) = getframe;
rel_errs(count) = xr - xi;
count = count+1;
% pause % why is this here?

for iter = 1:maxIters
    xrold = xr;
    % find Newtons update
    [xr] = newtons_update(f,fder, xrold);
    
    % Relative error from xr and xrold and stopping criteria and break if
    % rel_err<tol. 
    % write from here
    err = abs(xrold - xr);
    if err > tol
        rel_errs(count) = err;
    else 
        return %break
    end
    
    % plot the xlim_values vs function values and draw x-axis and y-axis
    % centered at origin
    % write from here
    hold on;
    ylim([0 inf])
    plot(xlim_values, f_values)

    % plot tangent at xr
    % write from here
    tang = (xlim_values-xr) * fder(xr) + f(xr);
    plot(xlim_values,tang)

    % write xr as xiter_no. ex: x1, x2 for first and second iteration
    % write from here
    xrCurrent = "x" + count;
    text(xr,0,xrCurrent)

    % draw line from xr to f(xr)
    % write from here
    line([xr,xr],[0,f(xr)])
    text(xr,f(xr),'f('+xrCurrent+ ')')

    % find Newtons update and write on the same plot
    % write from here
    [xr] = newtons_update(f,fder, xr);
    xrNext= "x" + (count+1);
    text(xr, f(xr), 'f(' +xrNext+ ')')
    hold off
    
    % save the current frame for the video. Store in M(count)
    % write from here
    M(count) = getframe;
    count=count+1;
    pause

end
root = xr; % root found by your algorithm

%  play movie using movie commnad. 
% write from here
movie(M)


