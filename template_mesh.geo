// === 1. Parametreler ===
pillar_dia = 40e-6;
pillar_spacing = 60e-6;
chip_length = 600e-6;
chip_width = 1000e-6;
chip_height = 50e-6;
inlet_length = 50e-6;
outlet_length = 50e-6;

lc_pillars = pillar_dia / 4; // Mesh kalitesi
lc_channel = chip_width / 15;

// === 2. Ana Kanalı Çiz ===
p1 = newp; Point(p1) = {0, 0, 0, lc_channel};
p2 = newp; Point(p2) = {chip_length, 0, 0, lc_channel};
p3 = newp; Point(p3) = {chip_length, chip_width, 0, lc_channel};
p4 = newp; Point(p4) = {0, chip_width, 0, lc_channel};

l_bottom = newl; Line(l_bottom) = {p1, p2};
l_outlet = newl; Line(l_outlet) = {p2, p3};
l_top = newl; Line(l_top) = {p3, p4};
l_inlet = newl; Line(l_inlet) = {p4, p1};

// Loop Sırası ÖNEMLİDİR: Bottom, Outlet, Top, Inlet
ll_box = newll; Curve Loop(ll_box) = {l_bottom, l_outlet, l_top, l_inlet};

// === 3. Sütunları Hazırla ===
num_cols = Floor((chip_length - inlet_length - outlet_length) / pillar_spacing);
num_rows = Floor(chip_width / pillar_spacing);
start_x = inlet_length + ((chip_length - inlet_length - outlet_length) - (num_cols-1)*pillar_spacing)/2;
start_y = (chip_width - (num_rows-1)*pillar_spacing)/2;

hole_loops[] = {};
For i In {0:num_cols-1}
  For j In {0:num_rows-1}
    x = start_x + i * pillar_spacing;
    y = start_y + j * pillar_spacing;
    If (i % 2 != 0) y += pillar_spacing / 2; EndIf

    If (y > pillar_dia/2 && y < (chip_width - pillar_dia/2))
      p_cen = newp; Point(p_cen) = {x, y, 0, lc_pillars};
      p1 = newp; Point(p1) = {x + pillar_dia/2, y, 0, lc_pillars};
      p2 = newp; Point(p2) = {x, y + pillar_dia/2, 0, lc_pillars};
      p3 = newp; Point(p3) = {x - pillar_dia/2, y, 0, lc_pillars};
      p4 = newp; Point(p4) = {x, y - pillar_dia/2, 0, lc_pillars};
      c1 = newc; Circle(c1) = {p1, p_cen, p2};
      c2 = newc; Circle(c2) = {p2, p_cen, p3};
      c3 = newc; Circle(c3) = {p3, p_cen, p4};
      c4 = newc; Circle(c4) = {p4, p_cen, p1};
      ll_pillar = newll; Curve Loop(ll_pillar) = {c1, c2, c3, c4};
      hole_loops += {ll_pillar};
    EndIf
  EndFor
EndFor

// === 4. Yüzeyi Oluştur ===
s_fluid_base = news; 
Plane Surface(s_fluid_base) = {ll_box, hole_loops[]};

// === 5. 3D EXTRUDE (Kritik Kısım) ===
// Extrude komutu bir liste döndürür:
// out[0] = Üst Yüzey (Top)
// out[1] = Hacim (Volume)
// out[2] = İlk çizginin (l_bottom) duvarı
// out[3] = İkinci çizginin (l_outlet) duvarı
// out[4] = Üçüncü çizginin (l_top) duvarı
// out[5] = Dördüncü çizginin (l_inlet) duvarı
// out[6...] = Sütun duvarları
out[] = Extrude {0, 0, chip_height} { Surface{s_fluid_base}; Layers{5}; Recombine; };

// === 6. Fiziksel Gruplar (İsimlendirme) ===
Physical Volume("fluid") = {out[1]};

Physical Surface("frontAndBack") = {s_fluid_base, out[0]}; // Alt ve Üst
Physical Surface("walls") = {out[2], out[4]}; // Yan duvarlar
Physical Surface("outlet") = {out[3]};        // Çıkış
Physical Surface("inlet") = {out[5]};         // Giriş

// Sütunlar (Geri kalan her şey)
Physical Surface("pillars") = {out[6: #out[]-1]};
