import subprocess
import pandas as pd
import pyvista as pv
import numpy as np
import os
import re
import shutil

# --- AYARLAR ---
DATASET_DIR_NAME = "ai_dataset_3D_Snappy" 
CSV_FILENAME = "final_training_data.csv"

base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir, DATASET_DIR_NAME)
npy_files_dir = os.path.join(dataset_dir, "npy_files")
output_csv_path = os.path.join(dataset_dir, CSV_FILENAME)

os.makedirs(npy_files_dir, exist_ok=True)

# --- Yardımcı: WSS Hesaplatıcı ---
def generate_wss_field(run_folder_name):
    uid, gid = os.getuid(), os.getgid()
    case_path = os.path.join(DATASET_DIR_NAME, run_folder_name)
    bash_cmd = f"source /opt/openfoam11/etc/bashrc && foamPostProcess -func wallShearStress -case {case_path}"
    full_cmd = ["docker", "run", "--rm", "--user", f"{uid}:{gid}", "-v", f"{base_dir}:/data", "-w", "/data", "--entrypoint", "/bin/bash", "openfoam/openfoam11-paraview510", "-c", bash_cmd]
    try:
        subprocess.run(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except: return False

# --- YARDIMCI: "Kutuyu Açma" Fonksiyonu (Hata Düzeltici) ---
def get_data_block(dataset):
    """
    Eğer veri bir MultiBlock (kutu) ise, içindeki gerçek veriyi (UnstructuredGrid/PolyData) bulup çıkarır.
    """
    if dataset is None:
        return None
    
    # Eğer zaten veri ise, olduğu gibi döndür
    if not isinstance(dataset, pv.MultiBlock):
        return dataset
    
    # Eğer MultiBlock ise, içindeki blokları gez
    # Genellikle tek bir blok vardır, onu döndür
    if dataset.n_blocks > 0:
        # İlk dolu bloğu bulup döndür
        for i in range(dataset.n_blocks):
            block = dataset.get_block(i)
            if block is not None and not isinstance(block, pv.MultiBlock):
                return block
            elif isinstance(block, pv.MultiBlock):
                # İç içe kutu varsa, recursive (tekrar) çağır
                inner = get_data_block(block)
                if inner is not None: return inner
    
    return None

# --- Ana Döngü ---
print(f"--- VERİ HASADI BAŞLIYOR ---")
print(f"Klasör: {dataset_dir}")

try:
    all_folders = sorted([f for f in os.listdir(dataset_dir) if f.startswith("run_")])
except: print("Klasör bulunamadı."); exit()

print(f"Toplam {len(all_folders)} klasör tarandı.")

results = []
dummy_foam = os.path.join(dataset_dir, "temp.foam")
with open(dummy_foam, 'w') as f: f.write("FoamFile { version 2.0; format ascii; class dictionary; object case.foam; }")

for run_name in all_folders:
    run_path = os.path.join(dataset_dir, run_name)
    
    try:
        match = re.search(r"dia(\d+\.\d+)_gap(\d+\.\d+)", run_name)
        dia = float(match.group(1))
        gap = float(match.group(2))
    except: continue

    # 1. BASINÇ
    log_file = os.path.join(run_path, "log.run")
    pres = None
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in reversed(f.readlines()):
                if match := re.search(r"(areaAverage|average)\(inlet\) of p\s*=\s*(-?\d+\.?\d*e?[-+]?\d*)", line): 
                    pres = float(match.group(2)) * 1025
                    break
    
    if pres is None:
        print(f"[ATLA] {run_name}: Basınç yok.")
        continue

    # 2. WSS OLUŞTUR
    wss_generated = False
    for time_dir in os.listdir(run_path):
        if time_dir.replace('.', '', 1).isdigit():
            if os.path.exists(os.path.join(run_path, time_dir, "wallShearStress")):
                wss_generated = True; break
    
    if not wss_generated:
        generate_wss_field(run_name)

    # 3. WSS OKU ve KAYDET
    try:
        case_foam = os.path.join(run_path, "case.foam")
        shutil.copy(dummy_foam, case_foam)
        
        reader = pv.OpenFOAMReader(case_foam)
        if not reader.time_values: os.remove(case_foam); continue
        
        reader.set_active_time_value(reader.time_values[-1])
        mesh_root = reader.read()
        
        wss_vecs = None
        
        # --- HATAYI ÇÖZEN KISIM ---
        # Hiyerarşiyi manuel gezmek yerine hedef odaklı gidiyoruz
        
        # A. Önce 'pillars' yamasını bulmaya çalış
        target_mesh = None
        try:
            # PyVista'da MultiBlock sözlük gibi davranır
            if 'boundary' in mesh_root.keys():
                boundary = mesh_root['boundary']
                if 'pillars' in boundary.keys():
                    # 'pillars' bloğunu al ve "Kutuyu Aç" fonksiyonuna gönder
                    target_mesh = get_data_block(boundary['pillars'])
        except: pass

        # B. Eğer pillars bulunamazsa veya boşsa, ana mesh'e (internalMesh) bak
        if target_mesh is None:
             # Ana mesh'i "Kutudan Çıkar"
             target_mesh = get_data_block(mesh_root)

        # C. Şimdi veriyi kontrol et (Artık target_mesh'in bir Mesh nesnesi olduğundan eminiz)
        if target_mesh is not None:
            if 'wallShearStress' in target_mesh.array_names:
                wss_vecs = target_mesh['wallShearStress']
            elif 'U' in target_mesh.point_data:
                # Yedek plan: Hesapla
                target_mesh = target_mesh.compute_derivative(scalars="U")
                wss_calc = target_mesh.wall_shear_stress(viscosity=1.35e-6)
                if 'wallShearStress' in wss_calc.point_data:
                    wss_vecs = wss_calc['wallShearStress']
        
        if wss_vecs is not None:
            npy_name = f"{run_name}_wss.npy"
            np.save(os.path.join(npy_files_dir, npy_name), wss_vecs)
            
            mags = np.linalg.norm(wss_vecs, axis=1)
            results.append({
                "run_name": run_name, "dia_um": dia, "gap_um": gap,
                "pressure_Pa": pres, 
                "wss_max": np.max(mags), "wss_mean": np.mean(mags), "wss_std": np.std(mags),
                "npy_file": npy_name
            })
            print(f"[OK] {run_name} | P={pres:.2f} | WSS_Max={np.max(mags):.4f}")
        else:
            print(f"[HATA] {run_name}: WSS verisi bulunamadı.")

        os.remove(case_foam)

    except Exception as e:
        print(f"[HATA] {run_name}: {e}")
        if os.path.exists(case_foam): os.remove(case_foam)

if os.path.exists(dummy_foam): os.remove(dummy_foam)

if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ İŞLEM TAMAM: {len(results)} veri kaydedildi.")
else:
    print("\n❌ Veri bulunamadı.")
