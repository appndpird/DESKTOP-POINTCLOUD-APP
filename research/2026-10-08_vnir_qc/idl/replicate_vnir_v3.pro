; Independent re-computation of the PhenoApp v3 "cleaned VNIR" per-plot features for Muresk 2025-09-30,
; straight from the ENVI cube and the refit grid shapefile (polygon inset 10 cm, pixel-centre rasterisation,
; 0 = missing, vegetation = NDVI(800/670) >= 0.50, excluded 0-415 / 755-770 / 928-962 nm).

function rv3_inset, v, d
  compile_opt idl2
  n = n_elements(v[0, *]) - 1
  x = reform(double(v[0, 0:n-1])) & y = reform(double(v[1, 0:n-1]))
  area = total(x * shift(y, -1) - shift(x, -1) * y) / 2d
  if area lt 0 then begin
    x = reverse(x) & y = reverse(y)
  endif
  dx = shift(x, -1) - x & dy = shift(y, -1) - y & len = sqrt(dx^2 + dy^2)
  nx = -dy / len & ny = dx / len
  px = x + d * nx & py = y + d * ny
  rx = dblarr(n) & ry = dblarr(n)
  for i = 0, n-1 do begin
    j = (i - 1 + n) mod n
    det = dx[j] * dy[i] - dy[j] * dx[i]
    t = ((px[i] - px[j]) * dy[i] - (py[i] - py[j]) * dx[i]) / det
    rx[i] = px[j] + t * dx[j] & ry[i] = py[j] + t * dy[j]
  endfor
  return, transpose([[rx], [ry]])
end

function rv3_inside, poly, xc, yc
  compile_opt idl2
  n = n_elements(poly[0, *])
  ok = bytarr(n_elements(xc)) + 1b
  for i = 0, n-1 do begin
    j = (i + 1) mod n
    ex = poly[0, j] - poly[0, i] & ey = poly[1, j] - poly[1, i]
    cr = ex * (yc - poly[1, i]) - ey * (xc - poly[0, i])
    ok = ok and (cr ge 0)
  endfor
  return, ok
end

function rv3_nearest, wl, excl, nm
  compile_opt idl2
  cand = where(~excl)
  i = cand[(sort(abs(wl[cand] - nm)))[0]]
  return, i
end

function rv3_mean, arr, mask, sd=sd, n=n
  compile_opt idl2
  w = where(mask, n)
  if n eq 0 then begin
    sd = !values.d_nan & return, !values.d_nan
  endif
  m = mean(double(arr[w]))
  sd = sqrt((mean(double(arr[w])^2) - m^2) > 0)
  return, m
end

pro replicate_vnir_v3
  compile_opt idl2
  shp_file  = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\aligned_grid_refit_20250930.shp'
  cube_file = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin'
  out_dir   = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\vnir_qc_2026-10-08'
  inset = 0.10d & veg_t = 0.50d & scale = 10000d
  t0 = systime(1)
  e = ENVI(/HEADLESS)
  cube = e.OpenRaster(cube_file)
  csr = cube.SPATIALREF & ctp = double(csr.TIE_POINT_MAP) & ps = double((csr.PIXEL_SIZE)[0])
  nb = cube.NBANDS & wl = double(cube.METADATA['wavelength'])
  excl = (wl le 415d) or ((wl ge 755d) and (wl le 770d)) or ((wl ge 928d) and (wl le 962d))
  b670 = rv3_nearest(wl, excl, 670) & b800 = rv3_nearest(wl, excl, 800) & b740 = rv3_nearest(wl, excl, 740)
  b470 = rv3_nearest(wl, excl, 470) & b550 = rv3_nearest(wl, excl, 550) & b531 = rv3_nearest(wl, excl, 531)
  b570 = rv3_nearest(wl, excl, 570) & b900 = rv3_nearest(wl, excl, 900) & b970 = rv3_nearest(wl, excl, 970)
  b700 = rv3_nearest(wl, excl, 700) & b780 = rv3_nearest(wl, excl, 780)
  print, 'bands (1-based): 670->', b670+1, ' 800->', b800+1, ' 740->', b740+1, ' 470->', b470+1, ' 550->', b550+1, $
         ' 531->', b531+1, ' 570->', b570+1, ' 900->', b900+1, ' 970->', b970+1, ' 700->', b700+1, ' 780->', b780+1
  s = obj_new('IDLffShape', shp_file)
  s.GetProperty, N_ENTITIES=np
  openw, lun, out_dir + '\replicate_vnir_v3_muresk_2025-09-30.csv', /GET_LUN
  printf, lun, 'Plot_ID,name,region_px,fcover_vnir,valid_670_frac,valid_470_frac,valid_550_frac,NDVI_nb,NDVI_nb_sd,NDVI_nb_veg,NDVI_nb_valid,' + $
               'NDRE740,NDRE740_veg,PRI_veg,WBI_veg,refl_G,refl_R,refl_RE1,refl_N,REP_guyot,red_edge_pos_deriv_nm,red_edge_slope_max'
  hist_all = lonarr(200)
  for i = 0, np-1 do begin
    ent = s.GetEntity(i, /ATTRIBUTES)
    v = *ent.vertices & att = *ent.attributes
    pid = att.ATTRIBUTE_8 & name = att.ATTRIBUTE_10
    s.DestroyEntity, ent
    poly = rv3_inset(v, inset)
    c0 = floor((min(poly[0, *]) - ctp[0]) / ps) & c1 = ceil((max(poly[0, *]) - ctp[0]) / ps)
    r0 = floor((ctp[1] - max(poly[1, *])) / ps) & r1 = ceil((ctp[1] - min(poly[1, *])) / ps)
    nc = c1 - c0 + 1 & nr = r1 - r0 + 1
    cols = rebin(dindgen(nc), nc, nr) + c0 & rows = rebin(transpose(dindgen(nr)), nc, nr) + r0
    xc = ctp[0] + (cols + 0.5d) * ps & yc = ctp[1] - (rows + 0.5d) * ps
    m = rv3_inside(poly, xc[*], yc[*])
    idx = where(m, n_all)
    sub = cube.GetData(INTERLEAVE='bsq', SUB_RECT=[c0, r0, c1, r1])
    spx = (reform(sub, long(nc) * nr, nb))[idx, *]          ; [n_all, nb] uint
    r670 = double(spx[*, b670]) & r800 = double(spx[*, b800]) & r740 = double(spx[*, b740])
    okn = (r670 gt 0) and (r800 gt 0)
    ndvi = dblarr(n_all) + !values.d_nan
    w = where(okn, nw)
    if nw gt 0 then ndvi[w] = (r800[w] - r670[w]) / (r800[w] + r670[w])
    if nw gt 0 then hist_all += histogram(ndvi[w], min=-0.2d, max=1.0d, nbins=200)
    veg = okn and (ndvi ge veg_t)
    n_veg = long(total(veg)) & n_fin = long(total(okn))
    ndvi_all = rv3_mean(ndvi, okn, sd=ndvi_sd) & ndvi_veg = rv3_mean(ndvi, veg)
    okr = (r740 gt 0) and (r800 gt 0)
    ndre = dblarr(n_all) + !values.d_nan
    w = where(okr, nw)
    if nw gt 0 then ndre[w] = (r800[w] - r740[w]) / (r800[w] + r740[w])
    ndre_all = rv3_mean(ndre, okr) & ndre_veg = rv3_mean(ndre, okr and veg)
    r531 = double(spx[*, b531]) & r570 = double(spx[*, b570])
    okp = (r531 gt 0) and (r570 gt 0)
    pri = dblarr(n_all) + !values.d_nan
    w = where(okp, nw)
    if nw gt 0 then pri[w] = (r531[w] - r570[w]) / (r531[w] + r570[w])
    pri_veg = rv3_mean(pri, okp and veg)
    r900 = double(spx[*, b900]) & r970 = double(spx[*, b970])
    okw = (r900 gt 0) and (r970 gt 0)
    wbi = dblarr(n_all) + !values.d_nan
    w = where(okw, nw)
    if nw gt 0 then wbi[w] = r900[w] / r970[w]
    wbi_veg = rv3_mean(wbi, okw and veg)
    ; vegetation mean spectrum (band > 0 and veg)
    spec = dblarr(nb) + !values.d_nan
    for b = 0, nb-1 do begin
      col = double(spx[*, b]) & w = where((col gt 0) and veg, nw)
      if nw gt 0 then spec[b] = mean(col[w]) / scale
    endfor
    specc = spec & specc[where(excl)] = !values.d_nan
    refl_G = mean(specc[where((wl ge 510) and (wl le 600))], /NAN)
    refl_R = mean(specc[where((wl ge 620) and (wl le 690))], /NAN)
    refl_RE1 = mean(specc[where((wl ge 695) and (wl le 715))], /NAN)
    refl_N = mean(specc[where((wl ge 780) and (wl le 900))], /NAN)
    rep = 700 + 40 * ((spec[b670] + spec[b780]) / 2 - spec[b700]) / (spec[b740] - spec[b700])
    seg = where((wl ge 680) and (wl le 760)) & wseg = wl[seg] & sseg = spec[seg]
    ; numpy.gradient equivalent: 2nd-order interior on non-uniform spacing, 1st-order one-sided ends
    ns = n_elements(wseg) & d = dblarr(ns)
    for k = 1, ns-2 do begin
      hs = wseg[k] - wseg[k-1] & hd = wseg[k+1] - wseg[k]
      d[k] = (hs^2 * sseg[k+1] + (hd^2 - hs^2) * sseg[k] - hd^2 * sseg[k-1]) / (hs * hd * (hd + hs))
    endfor
    d[0] = (sseg[1] - sseg[0]) / (wseg[1] - wseg[0]) & d[ns-1] = (sseg[ns-1] - sseg[ns-2]) / (wseg[ns-1] - wseg[ns-2])
    imax = (where(d eq max(d, /NAN)))[0]
    f = '(F0.6)'
    printf, lun, strjoin([strtrim(pid, 2), name, strtrim(n_all, 2), string(double(n_veg) / (n_fin > 1), format=f), $
      string(total(r670 gt 0) / n_all, format=f), string(total(spx[*, b470] gt 0) / n_all, format=f), string(total(spx[*, b550] gt 0) / n_all, format=f), $
      string(ndvi_all, format=f), string(ndvi_sd, format=f), string(ndvi_veg, format=f), string(double(n_fin) / n_all, format=f), $
      string(ndre_all, format=f), string(ndre_veg, format=f), string(pri_veg, format=f), string(wbi_veg, format=f), $
      string(refl_G, format=f), string(refl_R, format=f), string(refl_RE1, format=f), string(refl_N, format=f), $
      string(rep, format='(F0.4)'), string(wseg[imax], format='(F0.3)'), string(d[imax], format='(E0.6)')], ',')
    if (i mod 32) eq 0 then print, 'plot ', pid, ' region_px ', n_all, ' ndvi ', ndvi_all, ' t=', long(systime(1) - t0), 's'
  endfor
  free_lun, lun
  obj_destroy, s
  ; Otsu on the flight-wide NDVI histogram (same bins as the tool)
  c = -0.2d + (dindgen(200) + 0.5d) * (1.2d / 200)
  h = double(hist_all)
  w1 = total(h, /CUMULATIVE) & w2 = w1[-1] - w1
  m1 = total(h * c, /CUMULATIVE) / (w1 > 1)
  m2 = reverse(total(reverse(h * c), /CUMULATIVE)) / (w2 > 1)
  var = w1[0:198] * w2[0:198] * (m1[0:198] - m2[1:199])^2
  print, 'Otsu NDVI threshold (raw): ', c[(where(var eq max(var)))[0]], '  -> bounded to [0.15, 0.50]'
  cube.Close & e.Close
  print, 'replicate done in ', long(systime(1) - t0), ' s'
end
