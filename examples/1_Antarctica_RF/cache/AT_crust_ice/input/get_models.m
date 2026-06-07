infile = "vp_model_smooth_2x2.su"
domain = 1
[seis, nt, nx, dt, errflag] = readsu(infile, domain)

imagesc(seis);
