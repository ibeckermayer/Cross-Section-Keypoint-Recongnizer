function bol = direc_not_empty(f)
a = size(ls(f));
bol = ~(a(1) == 0);
end