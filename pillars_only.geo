// --- Parameters (These will be overwritten by Python) ---
pillar_dia = 50e-6;
pillar_spacing = 100e-6; // Center-to-center spacing

// --- Fixed Geometry Parameters ---
chip_length = 2000e-6;
chip_width = 1000e-6;
chip_height = 50e-6;
inlet_length = 200e-6; // Length before pillars start
outlet_length = 200e-6; // Length after pillars end

// --- Mesh Settings ---
lc_pillars = pillar_dia / 10; // Finer mesh around pillars
lc_far = chip_width / 20;     // Coarser mesh far away

// --- Derived Parameters ---
// Calculate number of rows and columns based on spacing
num_cols = Floor((chip_length - inlet_length - outlet_length) / pillar_spacing);
num_rows = Floor(chip_width / pillar_spacing);

// Calculate actual start positions to center the array
array_start_x = inlet_length + ((chip_length - inlet_length - outlet_length) - (num_cols-1)*pillar_spacing)/2;
array_start_y = (chip_width - (num_rows-1)*pillar_spacing)/2;

// --- Create Pillars ---
For i In {0:num_cols-1}
  For j In {0:num_rows-1}
    x = array_start_x + i * pillar_spacing;
    y = array_start_y + j * pillar_spacing;
    
    // Stagger every other column
    If (i % 2 != 0)
      y += pillar_spacing / 2;
    EndIf

    // Only create if it's solidly inside the chip width
    If (y > pillar_dia && y < (chip_width - pillar_dia))
      p_center = newp; Point(p_center) = {x, y, 0, lc_pillars};
      
      p1 = newp; Point(p1) = {x + pillar_dia/2, y, 0, lc_pillars};
      p2 = newp; Point(p2) = {x, y + pillar_dia/2, 0, lc_pillars};
      p3 = newp; Point(p3) = {x - pillar_dia/2, y, 0, lc_pillars};
      p4 = newp; Point(p4) = {x, y - pillar_dia/2, 0, lc_pillars};

      c1 = newc; Circle(c1) = {p1, p_center, p2};
      c2 = newc; Circle(c2) = {p2, p_center, p3};
      c3 = newc; Circle(c3) = {p3, p_center, p4};
      c4 = newc; Circle(c4) = {p4, p_center, p1};

      ll = newll; Curve Loop(ll) = {c1, c2, c3, c4};
      s = news; Plane Surface(s) = {ll};
      
      // Extrude to create 3D pillar
      Extrude {0, 0, chip_height} { Surface{s}; Layers{5}; Recombine; }
    EndIf
  EndFor
EndFor

// --- Physical Groups for OpenFOAM ---
// We need to group all the side walls of the pillars into one patch
Physical Surface("pillars") = {Surface{:}}; // This grabs ALL surfaces created so far (which are just pillars)
