
function cosplot()
% MATLAB file example to approximate a sawtooth
% with a truncated Fourier expansion.
nterms=5;
fourbypi=4.0/pi;
np=100;
y(1:np)=pi/2.0;
x(1:np)=linspace(-2.0*pi,2*pi,np);

for k=1:nterms
 twokm=2*k-1;
 y=y-fourbypi*cos(twokm*x)/twokm^2;
end

plot(x,y)
print -dpsc matlab_test_plot.ps
quit
end


