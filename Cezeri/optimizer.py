import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import re
import pyvista as pv
import joblib
import subprocess 
import sys 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

csv_path = "final_dataset.csv"
output_dir = "final_project_output"
validation_dir = "validation_run_auto"

viscosity = 1.35e-6
density = 1025
inlet_velocity = 0.01

base_dir = os.getcwd()
val_dir = os.path.join(base_dir, validation_dir)
os.makedirs(output_dir, exist_ok=True)

gmsh_exec = shutil.which("gmsh") or "gmsh"

print("\n" + "="*60)
print("CEZERİ'ye Hoşgeldiniz!")
print("\n" + "="*60)

print("Model yükleniyor, lütfen bekleyiniz...")

if not os.path.exists(csv_path):
    print(f"HATA: Veri dosyası bulunamadı: {CSV_PATH}"); exit()

df = pd.read_csv(csv_path)

#özellik çıkarımı
df['resistance_factor'] = 1.0 / (df['gap_um'] ** 3)
df['blockage_ratio']  = df['dia_um'] / (df['dia_um'] + df['gap_um'])
df['aspect_ratio'] = df['gap_um'] / df['dia_um']

X = df[['dia_um', 'gap_um', 'resistance_factor', 'blockage_ratio', 'aspect_ratio']].values
#Veri varyasyonu için logoritmik dönüşümün yapılması 
y_p_log = np.log1p (df['pressure_Pa'].values)
y_wss = df['wss_max'].values

scaler_X = StandardScaler()
X_s = scaler_X.fit_transform(X)

scaler_yw = StandardScaler()
y_w_s = scaler_yw.fit_transform(y_wss.reshape(-1, 1)).flatten()

scaler_yp = StandardScaler()
y_p_s = scaler_yp.fit_transform(y_p_log.reshape(-1, 1)).flatten()

X_train, X_test, yp_train, yp_test, yw_train, yw_test = train_test_split(
        X_s, y_p_s, y_w_s, test_size=0.15, random_state=42
)

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
# basınç modelinin eğitimi
gp_p = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)
gp_p.fit(X_train, yp_train)
#WSS modelinin eğitimi
gp_w = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)
gp_w.fit(X_train, yw_train)

#modellerin başarı skorlerı
pp_pred = gp_p.predict(X_test)
ww_pred = gp_w.predict(X_test)

p_pred_real = np.expm1(scaler_yp.inverse_transform(pp_pred.reshape(-1, 1)).flatten())
p_test_real = np.expm1(scaler_yp.inverse_transform(yp_test.reshape(-1, 1)).flatten())
r2_p = r2_score(p_test_real, p_pred_real)

w_pred_real = scaler_yw.inverse_transform(ww_pred.reshape(-1, 1)).flatten()
w_test_real = scaler_yw.inverse_transform(yw_test.reshape(-1, 1)).flatten()
r2_w = r2_score(w_test_real, w_pred_real)

print("\n Modeller başarıyla yüklendi.")
print(f"Basınç R^2: {r2_p:.4f}")
print(f"WSS R^2: {r2_w:.2f}")

#sonuçları aldıktan sonra test-train ayırmadan tüm veri ile modeli yeniden eğitiyoruz maksimum başarı için

gp_p.fit(X_s, y_p_s)
gp_w.fit(X_s, y_w_s)

try:
    print("\n---- Hedef Belirleme ----")
    print("\n - Eksik belirttiğiniz değerler varsayılan olarak belirlenecektir (15 mikron | 16000 Pa)")
    t_in = input("-> Hedef tanecik çapı (mikron): ")
    target_size = float(t_in) if t_in.strip() else 18.0
    p_in = input("-> Maksimum hidrodinamik basınç: ")
    max_pressure = float(p_in) if p_in.strip() else 16000.0
except: 
    target_size = 18.0; max_pressure = 16000.0

print(f"Hedef: {target_size}um  | Limit: {max_pressure} Pa")
print("---- Tasarım hesaplanıyor...")

class LiveChipProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=2, xl=np.array([30, 12]), xu=np.array([70, 35]))

    def _evaluate(self, x, out, *args, **kwargs):
        dia, gap = x[0], x[1]
        # özellik çıkarımı
        res = 1.0 / (gap**3)
        blo = dia / (dia + gap)
        asp = gap / dia 

        inp_s = scaler_X.transform(np.array([[dia, gap, res, blo, asp]]))

        pred_p_s = gp_p.predict(inp_s)[0]
        pred_w_s = gp_w.predict(inp_s)[0]

        pred_p = np.expm1(scaler_yp.inverse_transform([[pred_p_s]])[0][0])
        pred_w = scaler_yw.inverse_transform([[pred_w_s]])[0][0]

        out["F"] = [-pred_w, pred_p] # WSS maksimize edilir, Pa minimize edilir
        out["G"] = [pred_p - max_pressure, target_size - gap]

algorithm = NSGA2(pop_size=100, n_offsprings=40, eliminate_duplicates=True)
res = minimize(LiveChipProblem(), algorithm, ('n_gen', 100), seed=1, verbose=False)
if res.X is None: print(" Uygun tasarım bulunamadı!"); exit()

best_idx = np.argmin(res.F[:, 0])
best_design = res.X[best_idx]
dia_opt, gap_opt = best_design[0], best_design[1]

res_f = 1.0/(gap_opt**3); blo_f = dia_opt/(dia_opt+gap_opt); asp_f = gap_opt / dia_opt
inp_opt = scaler_X.transform(np.array([[dia_opt, gap_opt, res_f, blo_f, asp_f]]))

p_final = np.expm1(scaler_yp.inverse_transform([gp_p.predict(inp_opt)])[0][0])
w_final = scaler_yw.inverse_transform([gp_w.predict(inp_opt)])[0][0]

print(f"\n EN İDEAL TASARIM: Dia={dia_opt:.2f}um, Gap={gap_opt:.2f}um")
print(f"\n Hesaplanan Metrikler: P={p_final:.2f} Pa, WSS={w_final:.4f} Pa")































