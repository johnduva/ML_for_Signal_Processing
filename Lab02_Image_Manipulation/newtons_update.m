function [xr] = newtons_update(f,fder, xi)
    xr = xi - (f(xi)/fder(xi));
end