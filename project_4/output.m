% Nonlinear Output Equations
function obs = output(x, u, w)
  y1 = x(1);
  y2 = x(2);
  y3 = x(3) - w(4) * sin(w(2)) + w(3) * cos(w(2)) * cos(w(1));
  y4 = x(4) - w(5) * sin(w(1)) + w(4) * cos(x(1)) * cos(w(2)) + ...
    w(3) * cos(x(1)) * sin(w(2));
  obs = [y1; y2; y3; y4];
end
