; Example figure only: 5 plots, panels = cube window | plot tif | agreement map. Row 0 at top.
function mf_rgb_tile, d, bands, maxes, W, H
  compile_opt idl2
  dims = size(d, /DIMENSIONS)
  img = bytarr(3, W, H)
  for k = 0, 2 do begin
    b = float(d[*, *, bands[k]])
    img[k, 0:dims[0]-1, 0:dims[1]-1] = byte(((b < maxes[k]) / maxes[k]) * 255)
  endfor
  return, img
end

pro make_fig
  compile_opt idl2
  plot_dir  = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass Dataset\NUE_I_DPIRD_Muresk_F_Gobi\2025-09-30\vnir'
  cube_file = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin'
  out_dir   = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\vnir_qc_2026-10-08'
  e = ENVI(/HEADLESS)
  cube = e.OpenRaster(cube_file)
  csr = cube.SPATIALREF & ctp = csr.TIE_POINT_MAP & cps = csr.PIXEL_SIZE
  nb = cube.NBANDS
  sel = [1, 33, 64, 97, 128]
  W = 368 & H = 375 & gap = 6
  nsel = n_elements(sel)
  rgb_b = [114, 77, 43] & rgb_max = [6000., 1500., 1300.]
  for v = 0, 1 do begin           ; v=0 false colour (QML 115/78/44), v=1 true colour (70/44/21)
    if v eq 1 then begin
      rgb_b = [69, 43, 20] & rgb_max = [1500., 1500., 1300.]
    endif
    fig = bytarr(3, 3*W + 2*gap, nsel*H + (nsel-1)*gap) + 40b
    for k = 0, nsel-1 do begin
      f = (file_search(plot_dir + '\plot_' + strtrim(sel[k],2) + '_*.tif'))[0]
      r = e.OpenRaster(f)
      sr = r.SPATIALREF & tp = sr.TIE_POINT_MAP
      nc = r.NCOLUMNS & nr = r.NROWS
      col0 = round((tp[0] - ctp[0]) / cps[0]) & row0 = round((ctp[1] - tp[1]) / cps[1])
      tif = r.GetData(INTERLEAVE='bsq')
      sub = cube.GetData(INTERLEAVE='bsq', SUB_RECT=[col0, row0, col0+nc-1, row0+nr-1])
      r.Close
      m = total(tif ne 0, 3) gt 0
      dmask = (total((long(tif) - long(sub)) ne 0, 3) gt 0) and m
      y0 = k * (H + gap)
      fig[*, 0:W-1, y0:y0+H-1] = mf_rgb_tile(sub, rgb_b, rgb_max, W, H)
      fig[*, W+gap:2*W+gap-1, y0:y0+H-1] = mf_rgb_tile(tif, rgb_b, rgb_max, W, H)
      dm = bytarr(3, W, H) + 25b
      g = byte(m * 190)
      dm[0, 0:nc-1, 0:nr-1] = g > byte(dmask * 255)
      dm[1, 0:nc-1, 0:nr-1] = g * (dmask eq 0)
      dm[2, 0:nc-1, 0:nr-1] = g * (dmask eq 0)
      fig[*, 2*W+2*gap:3*W+2*gap-1, y0:y0+H-1] = dm
    endfor
    name = v eq 0 ? '\plot_vs_cube_examples_falsecolour_115_78_44.png' : '\plot_vs_cube_examples_truecolour_70_44_21.png'
    write_png, out_dir + name, reverse(fig, 3)
  endfor
  file_delete, out_dir + '\plot_vs_cube_rgb_examples.png', /ALLOW_NONEXISTENT
  cube.Close
  e.Close
  print, 'figures written'
end
