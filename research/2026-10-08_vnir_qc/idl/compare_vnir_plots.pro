; Compare every per-plot VNIR GeoTIFF with the same window of the original ENVI orthomosaic.
; Runs headless in ENVI 6.2 / IDL 9.2.

function cvp_rgb_tile, d, bands, maxes, W, H
  compile_opt idl2
  dims = size(d, /DIMENSIONS)
  nc = dims[0] & nr = dims[1]
  img = bytarr(3, W, H)
  for k = 0, 2 do begin
    b = float(d[*, *, bands[k]])
    img[k, 0:nc-1, 0:nr-1] = byte(((b < maxes[k]) / maxes[k]) * 255)
  endfor
  return, img
end

pro compare_vnir_plots
  compile_opt idl2
  plot_dir  = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass Dataset\NUE_I_DPIRD_Muresk_F_Gobi\2025-09-30\vnir'
  cube_file = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin'
  out_dir   = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\vnir_qc_2026-10-08'
  file_mkdir, out_dir
  t0 = systime(1)

  e = ENVI(/HEADLESS)
  cube = e.OpenRaster(cube_file)
  csr = cube.SPATIALREF
  nb = cube.NBANDS
  wl = cube.METADATA['wavelength']
  print, 'cube: ', cube.NCOLUMNS, cube.NROWS, nb, ' ', cube.DATA_TYPE, ' crs ', csr.COORD_SYS_CODE
  print, 'cube tie map ', csr.TIE_POINT_MAP, ' tie pixel ', csr.TIE_POINT_PIXEL, ' ps ', csr.PIXEL_SIZE, format='(A,2F14.4,A,2F6.2,A,2F7.4)'
  ctp = csr.TIE_POINT_MAP & cps = csr.PIXEL_SIZE

  ; index csv written by the extraction tool
  idx = read_csv(plot_dir + '\plots_vnir_index.csv', HEADER=hdr)
  idx_id = long(idx.FIELD1) & idx_w = long(idx.FIELD4) & idx_h = long(idx.FIELD5) & idx_px = long(idx.FIELD6)

  files = file_search(plot_dir + '\plot_*.tif', COUNT=nf)
  ids = lonarr(nf) & names = strarr(nf)
  for i = 0, nf-1 do begin
    s = stregex(file_basename(files[i]), 'plot_([0-9]+)_([A-Za-z0-9]+)\.tif', /SUBEXPR, /EXTRACT)
    ids[i] = long(s[1]) & names[i] = s[2]
  endfor
  o = sort(ids) & files = files[o] & ids = ids[o] & names = names[o]
  print, 'plots found: ', nf

  ; outputs
  openw, lun, out_dir + '\plot_vs_cube_comparison.csv', /GET_LUN
  printf, lun, 'Plot_ID,name,tif_cols,tif_rows,tif_bands,tif_dtype,tif_crs,tif_pixel_m,cube_col0,cube_row0,' + $
               'align_resid_x_m,align_resid_y_m,n_inside_tif,n_inside_index,n_cube_allzero_in_box,n_outside_poly_with_cube_data,' + $
               'n_inside_px_any_band_diff,max_abs_diff_inside,frac_inside_bitexact,ndvi_tif,ndvi_cube,ndvi_cube_fullbox,' + $
               'mean_nir800_tif,mean_nir800_cube,mean_red670_tif,mean_red670_cube'
  openw, lun2, out_dir + '\plot_mean_spectra_tif_vs_cube.csv', /GET_LUN
  printf, lun2, 'Plot_ID,name,source,' + strjoin('b' + strtrim(indgen(nb)+1, 2) + '_' + strtrim(string(wl, format='(F0.1)'), 2), ',')
  band_maxdiff = lonarr(nb)

  sel = [1, 33, 64, 97, 128]
  W = max(idx_w) + 4 & H = max(idx_h) + 4 & gap = 6
  print, 'tile size ', W, H
  nsel = n_elements(sel)
  fig = bytarr(3, 3*W + 2*gap, nsel*H + (nsel-1)*gap) + 40b
  rgb_b = [114, 77, 43] & rgb_max = [6000., 1500., 1300.]

  for i = 0, nf-1 do begin
    r = e.OpenRaster(files[i])
    if i eq 0 then begin
      bn = r.METADATA['band names']
      print, 'tif band names (first 3, last): ', bn[0], ' | ', bn[1], ' | ', bn[2], ' | ', bn[-1]
      print, 'cube wavelengths (first 3, last): ', wl[0], wl[1], wl[2], wl[-1]
    endif
    sr = r.SPATIALREF
    nc = r.NCOLUMNS & nr = r.NROWS & nbt = r.NBANDS & dtype = r.DATA_TYPE
    tp = sr.TIE_POINT_MAP & ps = sr.PIXEL_SIZE & crs = sr.COORD_SYS_CODE
    fx = (tp[0] - ctp[0]) / cps[0]
    fy = (ctp[1] - tp[1]) / cps[1]
    col0 = round(fx) & row0 = round(fy)
    resx = (fx - col0) * cps[0] & resy = (fy - row0) * cps[1]
    if (col0 lt 0) or (row0 lt 0) or (col0+nc gt cube.NCOLUMNS) or (row0+nr gt cube.NROWS) then $
      message, 'plot ' + strtrim(ids[i],2) + ' window falls outside the cube'
    tif = r.GetData(INTERLEAVE='bsq')
    sub = cube.GetData(INTERLEAVE='bsq', SUB_RECT=[col0, row0, col0+nc-1, row0+nr-1])
    r.Close

    m = total(tif ne 0, 3) gt 0                 ; pixels the extraction kept (polygon, no hole)
    n_in = long(total(m))
    cube_nz = total(sub ne 0, 3) gt 0
    n_cube_zero = long(total(cube_nz eq 0))
    n_out_data = long(total((m eq 0) and cube_nz))
    diff = long(tif) - long(sub)
    dmask = total(diff ne 0, 3) gt 0
    n_diff_in = long(total(dmask and m))
    m3 = rebin(byte(m), nc, nr, nb)
    ad = abs(diff) * m3
    maxd = max(ad)
    for b = 0, nb-1 do band_maxdiff[b] = band_maxdiff[b] > max(ad[*, *, b])
    mt = total(total(double(tif) * m3, 1), 1) / n_in
    mc = total(total(double(sub) * m3, 1), 1) / n_in
    win = where(m)
    nir_t = float(tif[*, *, 114]) & red_t = float(tif[*, *, 77])
    nir_c = float(sub[*, *, 114]) & red_c = float(sub[*, *, 77])
    ndvi_t = mean(((nir_t - red_t) / ((nir_t + red_t) > 1))[win])
    ndvi_c = mean(((nir_c - red_c) / ((nir_c + red_c) > 1))[win])
    ndvi_box = mean(((nir_c - red_c) / ((nir_c + red_c) > 1))[where(cube_nz)])
    j = (where(idx_id eq ids[i], cnt))[0]
    n_idx = cnt gt 0 ? idx_px[j] : -1L

    f4 = '(F0.4)'
    printf, lun, strjoin([strtrim(ids[i],2), names[i], strtrim(nc,2), strtrim(nr,2), strtrim(nbt,2), dtype, $
      strtrim(crs,2), string(ps[0], format='(F0.3)'), strtrim(col0,2), strtrim(row0,2), $
      string(resx, format='(F0.5)'), string(resy, format='(F0.5)'), strtrim(n_in,2), strtrim(n_idx,2), strtrim(n_cube_zero,2), $
      strtrim(n_out_data,2), strtrim(n_diff_in,2), strtrim(maxd,2), string(1d - double(n_diff_in)/n_in, format='(F0.6)'), $
      string(ndvi_t, format=f4), string(ndvi_c, format=f4), string(ndvi_box, format=f4), $
      string(mt[114], format='(F0.2)'), string(mc[114], format='(F0.2)'), string(mt[77], format='(F0.2)'), string(mc[77], format='(F0.2)')], ',')
    printf, lun2, strjoin([strtrim(ids[i],2), names[i], 'plot_tif', strtrim(string(mt, format='(F0.2)'), 2)], ',')
    printf, lun2, strjoin([strtrim(ids[i],2), names[i], 'orthomosaic', strtrim(string(mc, format='(F0.2)'), 2)], ',')

    k = (where(sel eq ids[i], ks))[0]
    if ks gt 0 then begin
      y0 = k * (H + gap)
      fig[*, 0:W-1, y0:y0+H-1] = cvp_rgb_tile(sub, rgb_b, rgb_max, W, H)
      fig[*, W+gap:2*W+gap-1, y0:y0+H-1] = cvp_rgb_tile(tif, rgb_b, rgb_max, W, H)
      dm = bytarr(3, W, H)
      g = byte(m * 200)                         ; inside polygon = light grey
      dm[0, 0:nc-1, 0:nr-1] = g > byte(dmask * 255) & dm[1, 0:nc-1, 0:nr-1] = g * (dmask eq 0) & dm[2, 0:nc-1, 0:nr-1] = g * (dmask eq 0)
      fig[*, 2*W+2*gap:3*W+2*gap-1, y0:y0+H-1] = dm
    endif
    if (i mod 16) eq 0 then print, 'plot ', ids[i], ' col0/row0 ', col0, row0, ' inside ', n_in, ' diff px ', n_diff_in, ' maxdiff ', maxd, $
       ' t=', long(systime(1)-t0), 's'
  endfor
  free_lun, lun & free_lun, lun2
  write_png, out_dir + '\plot_vs_cube_rgb_examples.png', fig
  openw, lun3, out_dir + '\band_max_abs_diff.csv', /GET_LUN
  printf, lun3, 'band,wavelength_nm,max_abs_diff_inside_all_plots'
  for b = 0, nb-1 do printf, lun3, strtrim(b+1,2), ',', string(wl[b], format='(F0.2)'), ',', strtrim(band_maxdiff[b],2)
  free_lun, lun3
  cube.Close
  print, 'done in ', long(systime(1)-t0), ' s'
  e.Close
end
