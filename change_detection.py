#!/usr/bin/env python3
"""
C2C Vergleich v8 – verständlich kommentierte Endfassung
======================================================

Ziel des Skripts
----------------
Dieses Skript vergleicht zwei registrierte Punktwolken (Referenz und Vergleich)
und leitet daraus eine signierte Änderungsdarstellung ab.

Grundidee des Verfahrens
------------------------
1. Die Referenz- und Vergleichspunktwolke werden geladen.
2. Aus der Referenz wird ein DSM-artiges Raster aufgebaut.
3. Optional wird automatisch erkannt, ob die Z-Achse invertiert werden muss.
4. Für jeden Vergleichspunkt wird der nächste Referenzpunkt gesucht (C2C-Distanz).
5. Die C2C-Distanz wird mit Hilfe des DSM vorzeichenbehaftet:
   - positiv  = eher hinzugekommen / höher
   - negativ  = eher entfernt / niedriger
6. Die signierten Änderungen werden klassifiziert und exportiert als:
   - farbkodierte PLY-Datei
   - Histogramm der geänderten Punkte
   - Balkendiagramm der Klassenanteile
   - kompakte CSV mit Kernkennzahlen

Wichtiger Hinweis zur Methodik
------------------------------
Die signierte Distanz ist hier absichtlich ein Hybridmaß:
- Betrag   = C2C-Abstand zum nächsten Nachbarn
- Vorzeichen = aus dem DSM abgeleitete Richtung

Das Maß ist damit kein "reines" Höhen-Delta und auch keine klassische
normalengerichtete Oberflächendistanz. Es ist bewusst auf robuste
Änderungsdetektion ausgelegt.

" Crop erfolgt nicht auf beide separat, weil der Crop sonst auf zwei unterschiedlichen Bodenreferenzen basieren würde und die Vergleichbarkeit der Änderungsanalyse schlechter wäre."
"""

# ════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ════════════════════════════════════════════════════════════════════════

# Pfade zu den beiden Punktwolken
REFERENCE_POINT_CLOUD_PATH = r"C:\Users\Admin\Desktop\Photogrammetrie\pcd_registration\Registered_pcds\flight6A.ply"
COMPARISON_POINT_CLOUD_PATH = r"C:\Users\Admin\Desktop\Photogrammetrie\pcd_registration\Registered_pcds\flight6B.ply"

# Ausgabeordner und Tag für Dateinamen
OUTPUT_DIRECTORY = r"D:\Users\Admin\Desktop\Photogrammetrie\pcd_registration_exports\registed_pcds\c2c"
RUN_TAG = "XX--"

# Optionaler lokaler Boden-Crop
# Dieser Crop beschränkt die Auswertung auf einen Höhenbereich über einem lokal
# geschätzten Bodenmodell.
ENABLE_HEIGHT_CROP = False
HEIGHT_CROP_MIN_ABOVE_GROUND_M = 0.0
HEIGHT_CROP_MAX_ABOVE_GROUND_M = 1.5
GROUND_MODEL_GRID_CELL_SIZE_M = 0.1 # [1.0, ]
GROUND_MODEL_MEDIAN_FILTER_SIZE = 0.3 # [3,]

# Z-Achsen-Behandlung:
# - "auto" = automatische Erkennung per DSM-Rauheit
# - "flip" = Z-Achse immer invertieren
# - "keep" = Z-Achse unverändert lassen
Z_AXIS_HANDLING_MODE = "auto"

# Schwellwerte zur Klassifikation der Änderungsstärke
# |d| <= T1                 -> unchanged
# T1 < |d| <= T2            -> possible
# |d| > T2                  -> likely
CHANGE_THRESHOLD_T1_M = 0.10
CHANGE_THRESHOLD_T2_M = 0.20

# Rastergröße für flächenbezogene Statistiken
CHANGE_STATISTICS_GRID_CELL_SIZE_M = 1.0

# DSM-Auflösung für:
# - automatische Z-Flip-Erkennung
# - Vorzeichenbestimmung der Änderungsrichtung
DSM_GRID_CELL_SIZE_M = 0.05

# Farben der signierten Änderungsdarstellung (RGB im Bereich 0..1)
COLOR_LIKELY_REMOVED = (0.05, 0.20, 0.85)
COLOR_POSSIBLY_REMOVED = (0.45, 0.68, 0.96)
COLOR_UNCHANGED = (0.68, 0.68, 0.68)
COLOR_POSSIBLY_ADDED = (0.96, 0.58, 0.28)
COLOR_LIKELY_ADDED = (0.85, 0.08, 0.08)

# ════════════════════════════════════════════════════════════════════════
# IMPORTE
# ════════════════════════════════════════════════════════════════════════

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("Open3D fehlt. Installiere mit: pip install open3d")

try:
    from scipy.ndimage import distance_transform_edt, median_filter
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit("SciPy fehlt. Installiere mit: pip install scipy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warnung: Matplotlib nicht gefunden – Plots werden übersprungen.")

# ════════════════════════════════════════════════════════════════════════
# KLASSENBEZEICHNUNGEN
# ════════════════════════════════════════════════════════════════════════

UNSIGNED_CHANGE_CLASS_NAMES = {
    0: "unchanged",
    1: "possible",
    2: "likely",
}

SIGNED_CHANGE_CLASS_NAMES = {
    -2: "likely removed",
    -1: "possibly removed",
     0: "unchanged",
     1: "possibly added",
     2: "likely added",
}

# ════════════════════════════════════════════════════════════════════════
# ALLGEMEINE HILFSFUNKTIONEN
# ════════════════════════════════════════════════════════════════════════

def build_output_file_prefix(run_tag: str) -> str:
    """
    Erzeugt einen eindeutigen Dateinamen-Präfix aus Zeitstempel und Run-Tag.

    Beispiel:
        20260404_153501_FIN--
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    normalized_tag = run_tag.strip().replace(" ", "_") if run_tag else "run"
    return f"{timestamp}_{normalized_tag}"


def load_point_cloud_from_ply(file_path: str) -> o3d.geometry.PointCloud:
    """
    Lädt eine PLY-Punktwolke mit Open3D.

    Abbruchbedingungen:
    - Dateiendung ist nicht .ply
    - Datei enthält keine Punkte
    """
    if Path(file_path).suffix.lower() != ".ply":
        sys.exit(f"Nur PLY-Dateien werden unterstützt: {file_path}")

    print(f"  Lade: {file_path}")
    point_cloud = o3d.io.read_point_cloud(file_path)
    number_of_points = len(point_cloud.points)

    if number_of_points == 0:
        sys.exit(f"Keine Punkte in Datei: {file_path}")

    print(f"    -> {number_of_points:,} Punkte")
    return point_cloud

# ════════════════════════════════════════════════════════════════════════
# DSM-AUFBAU UND Z-FLIP-ERKENNUNG
# ════════════════════════════════════════════════════════════════════════

def build_reference_dsm_min_max_raster(
    reference_points_xyz: np.ndarray,
    comparison_points_xyz: np.ndarray,
    dsm_cell_size_m: float,
) -> dict:
    """
    Baut ein kombiniertes DSM-Raster aus der Referenzpunktwolke.

    Für jede Rasterzelle werden zwei Informationen gespeichert:
    - dsm_max: höchste Referenzoberfläche in der Zelle
    - dsm_min: niedrigste Referenzoberfläche in der Zelle

    Der XY-Extent wird so gewählt, dass sowohl Referenz als auch Vergleich
    vollständig abgedeckt sind.

    Leere Zellen werden mit dem jeweils nächstgelegenen belegten Rasterwert
    gefüllt, damit später keine Lücken in der Vorzeichenbestimmung entstehen.
    """
    reference_x = reference_points_xyz[:, 0]
    reference_y = reference_points_xyz[:, 1]
    reference_z = reference_points_xyz[:, 2]

    comparison_x = comparison_points_xyz[:, 0]
    comparison_y = comparison_points_xyz[:, 1]

    x_min_global = float(min(reference_x.min(), comparison_x.min()))
    y_min_global = float(min(reference_y.min(), comparison_y.min()))
    x_max_global = float(max(reference_x.max(), comparison_x.max()))
    y_max_global = float(max(reference_y.max(), comparison_y.max()))

    number_of_cells_x = int(np.floor((x_max_global - x_min_global) / dsm_cell_size_m)) + 2
    number_of_cells_y = int(np.floor((y_max_global - y_min_global) / dsm_cell_size_m)) + 2

    print(
        f"  DSM-Raster: {number_of_cells_x} × {number_of_cells_y} = "
        f"{number_of_cells_x * number_of_cells_y:,} Zellen  "
        f"(cell={dsm_cell_size_m:.3f}m)"
    )

    reference_cell_index_x = np.clip(
        np.floor((reference_x - x_min_global) / dsm_cell_size_m).astype(np.int32),
        0,
        number_of_cells_x - 1,
    )
    reference_cell_index_y = np.clip(
        np.floor((reference_y - y_min_global) / dsm_cell_size_m).astype(np.int32),
        0,
        number_of_cells_y - 1,
    )

    reference_z_float32 = reference_z.astype(np.float32)

    dsm_max_surface = np.full((number_of_cells_x, number_of_cells_y), -np.inf, dtype=np.float32)
    dsm_min_surface = np.full((number_of_cells_x, number_of_cells_y), np.inf, dtype=np.float32)

    np.maximum.at(dsm_max_surface, (reference_cell_index_x, reference_cell_index_y), reference_z_float32)
    np.minimum.at(dsm_min_surface, (reference_cell_index_x, reference_cell_index_y), reference_z_float32)

    empty_cells_in_max_surface = dsm_max_surface == -np.inf
    empty_cells_in_min_surface = dsm_min_surface == np.inf

    if np.any(empty_cells_in_max_surface):
        dsm_max_surface[empty_cells_in_max_surface] = np.nan
        dsm_max_surface = dsm_max_surface[
            tuple(
                distance_transform_edt(
                    empty_cells_in_max_surface,
                    return_distances=False,
                    return_indices=True,
                )
            )
        ]

    if np.any(empty_cells_in_min_surface):
        dsm_min_surface[empty_cells_in_min_surface] = np.nan
        dsm_min_surface = dsm_min_surface[
            tuple(
                distance_transform_edt(
                    empty_cells_in_min_surface,
                    return_distances=False,
                    return_indices=True,
                )
            )
        ]

    return {
        "x_min": x_min_global,
        "y_min": y_min_global,
        "cell_size": float(dsm_cell_size_m),
        "nx": number_of_cells_x,
        "ny": number_of_cells_y,
        "dsm_max": dsm_max_surface,
        "dsm_min": dsm_min_surface,
    }


def detect_whether_z_axis_is_flipped_from_dsm(dsm_raster: dict) -> bool:
    """
    Erkennt eine mögliche Invertierung der Z-Achse über die Rauheit des DSM.

    Heuristik:
    - dsm_max beschreibt die obere sichtbare Oberfläche.
    - dsm_min beschreibt die untere sichtbare Oberfläche.
    - Falls die obere Oberfläche deutlich glatter ist als die untere,
      ist das im Anwendungsfall ein Hinweis auf eine invertierte Z-Achse.

    Rückgabewert:
    - True  -> Z-Achse soll invertiert werden
    - False -> Z-Achse bleibt unverändert
    """
    upper_surface_roughness_std = float(np.nanstd(dsm_raster["dsm_max"]))
    lower_surface_roughness_std = float(np.nanstd(dsm_raster["dsm_min"]))

    should_flip_z_axis = upper_surface_roughness_std < lower_surface_roughness_std
    confidence_value = abs(upper_surface_roughness_std - lower_surface_roughness_std) / max(
        upper_surface_roughness_std + lower_surface_roughness_std,
        1e-10,
    )

    print(
        f"  DSM-Rauheit  oben (max): {upper_surface_roughness_std:.4f}m  |  "
        f"unten (min): {lower_surface_roughness_std:.4f}m  |  conf: {confidence_value:.2%}"
    )
    print(f"  -> Z {'invertiert  →  flip' if should_flip_z_axis else 'korrekt  →  keep'}")

    return should_flip_z_axis


def derive_working_dsm_surface_from_raw_dsm(dsm_raster: dict, z_axis_was_flipped: bool) -> dict:
    """
    Leitet aus dem Roh-DSM die tatsächlich zu verwendende Referenzoberfläche ab.

    Fall 1: Z bleibt unverändert
        -> oberste Referenzoberfläche = dsm_max

    Fall 2: Z wurde invertiert
        -> nach dem Flip wird aus dem früheren Minimum die neue obere Oberfläche
        -> deshalb working_surface = -dsm_min

    Diese Funktion baut nichts neu auf, sondern nutzt das bereits erzeugte Raster.
    """
    working_reference_surface = -dsm_raster["dsm_min"] if z_axis_was_flipped else dsm_raster["dsm_max"]

    return {
        "x_min": dsm_raster["x_min"],
        "y_min": dsm_raster["y_min"],
        "cell_size": dsm_raster["cell_size"],
        "nx": dsm_raster["nx"],
        "ny": dsm_raster["ny"],
        "surface": working_reference_surface,
    }


def rebuild_working_dsm_surface_from_reference_cloud(
    reference_point_cloud: o3d.geometry.PointCloud,
    comparison_points_xyz: np.ndarray,
    dsm_cell_size_m: float,
) -> dict:
    """
    Baut die verwendete Referenzoberfläche direkt aus einer bereits bearbeiteten
    Referenzwolke neu auf.

    Diese Funktion wird nur im Crop-Fall benötigt, weil sich dort Extent und
    Punktmenge nach dem Zuschnitt ändern können.
    """
    reference_points_xyz = np.asarray(reference_point_cloud.points)

    reference_x = reference_points_xyz[:, 0]
    reference_y = reference_points_xyz[:, 1]
    reference_z = reference_points_xyz[:, 2]

    comparison_x = comparison_points_xyz[:, 0]
    comparison_y = comparison_points_xyz[:, 1]

    x_min_global = float(min(reference_x.min(), comparison_x.min()))
    y_min_global = float(min(reference_y.min(), comparison_y.min()))
    x_max_global = float(max(reference_x.max(), comparison_x.max()))
    y_max_global = float(max(reference_y.max(), comparison_y.max()))

    number_of_cells_x = int(np.floor((x_max_global - x_min_global) / dsm_cell_size_m)) + 2
    number_of_cells_y = int(np.floor((y_max_global - y_min_global) / dsm_cell_size_m)) + 2

    reference_cell_index_x = np.clip(
        np.floor((reference_x - x_min_global) / dsm_cell_size_m).astype(np.int32),
        0,
        number_of_cells_x - 1,
    )
    reference_cell_index_y = np.clip(
        np.floor((reference_y - y_min_global) / dsm_cell_size_m).astype(np.int32),
        0,
        number_of_cells_y - 1,
    )

    working_reference_surface = np.full((number_of_cells_x, number_of_cells_y), -np.inf, dtype=np.float32)
    np.maximum.at(
        working_reference_surface,
        (reference_cell_index_x, reference_cell_index_y),
        reference_z.astype(np.float32),
    )

    empty_cells = working_reference_surface == -np.inf
    if np.any(empty_cells):
        working_reference_surface[empty_cells] = np.nan
        working_reference_surface = working_reference_surface[
            tuple(
                distance_transform_edt(
                    empty_cells,
                    return_distances=False,
                    return_indices=True,
                )
            )
        ]

    return {
        "x_min": x_min_global,
        "y_min": y_min_global,
        "cell_size": float(dsm_cell_size_m),
        "nx": number_of_cells_x,
        "ny": number_of_cells_y,
        "surface": working_reference_surface,
    }

# ════════════════════════════════════════════════════════════════════════
# Z-ACHSE INVERTIEREN / WIEDERHERSTELLEN
# ════════════════════════════════════════════════════════════════════════

def apply_optional_z_axis_flip(
    point_cloud: o3d.geometry.PointCloud,
    should_flip_z_axis: bool,
) -> o3d.geometry.PointCloud:
    """
    Spiegelt optional die Z-Koordinate einer Punktwolke.

    Falls Farben oder Normalen vorhanden sind, werden diese ebenfalls korrekt
    übernommen. Bei Normalen wird die Z-Komponente ebenfalls invertiert.
    """
    if not should_flip_z_axis:
        return point_cloud

    flipped_points_xyz = np.asarray(point_cloud.points).copy()
    flipped_points_xyz[:, 2] *= -1

    flipped_point_cloud = o3d.geometry.PointCloud()
    flipped_point_cloud.points = o3d.utility.Vector3dVector(flipped_points_xyz)

    if point_cloud.has_colors():
        flipped_point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors).copy())

    if point_cloud.has_normals():
        flipped_normals_xyz = np.asarray(point_cloud.normals).copy()
        flipped_normals_xyz[:, 2] *= -1
        flipped_point_cloud.normals = o3d.utility.Vector3dVector(flipped_normals_xyz)

    return flipped_point_cloud


def restore_original_z_axis_orientation(
    point_cloud: o3d.geometry.PointCloud,
    z_axis_was_flipped: bool,
) -> o3d.geometry.PointCloud:
    """
    Stellt die ursprüngliche Z-Orientierung wieder her.

    Technisch ist das identisch zu einem erneuten Flip.
    """
    return apply_optional_z_axis_flip(point_cloud, z_axis_was_flipped)

# ════════════════════════════════════════════════════════════════════════
# LOKALES BODENMODELL UND OPTIONALER HÖHENCROP
# ════════════════════════════════════════════════════════════════════════

def build_local_ground_model_from_reference(
    reference_point_cloud: o3d.geometry.PointCloud,
    ground_grid_cell_size_m: float = 1.0,
    median_filter_size: int = 3,
) -> dict:
    """
    Erzeugt ein einfaches lokales Bodenmodell aus der Referenzwolke.

    Vorgehen:
    - Das XY-Gebiet wird gerastert.
    - Für jede Zelle wird der niedrigste Z-Wert als Bodenansatz verwendet.
    - Leere Zellen werden mit dem nächsten belegten Bodenwert gefüllt.
    - Optional wird ein Medianfilter zur Glättung angewendet.

    Die Funktion wird nur benötigt, wenn ein lokaler Höhen-Crop aktiviert ist.
    """
    reference_points_xyz = np.asarray(reference_point_cloud.points)

    reference_x = reference_points_xyz[:, 0]
    reference_y = reference_points_xyz[:, 1]
    reference_z = reference_points_xyz[:, 2].astype(np.float32)

    x_min = float(reference_x.min())
    x_max = float(reference_x.max())
    y_min = float(reference_y.min())
    y_max = float(reference_y.max())

    number_of_cells_x = int(np.floor((x_max - x_min) / ground_grid_cell_size_m)) + 1
    number_of_cells_y = int(np.floor((y_max - y_min) / ground_grid_cell_size_m)) + 1

    print(
        f"  Lokales Bodenmodell: {number_of_cells_x} × {number_of_cells_y} Zellen "
        f"({ground_grid_cell_size_m:.2f}m)"
    )

    cell_index_x = np.clip(
        np.floor((reference_x - x_min) / ground_grid_cell_size_m).astype(np.int32),
        0,
        number_of_cells_x - 1,
    )
    cell_index_y = np.clip(
        np.floor((reference_y - y_min) / ground_grid_cell_size_m).astype(np.int32),
        0,
        number_of_cells_y - 1,
    )

    ground_height_raster = np.full((number_of_cells_x, number_of_cells_y), np.inf, dtype=np.float32)
    point_count_per_cell = np.zeros((number_of_cells_x, number_of_cells_y), dtype=np.int32)

    np.minimum.at(ground_height_raster, (cell_index_x, cell_index_y), reference_z)
    np.add.at(point_count_per_cell, (cell_index_x, cell_index_y), 1)

    valid_cells_mask = point_count_per_cell > 0
    number_of_empty_cells = int(np.sum(~valid_cells_mask))

    if not np.any(valid_cells_mask):
        sys.exit("Konnte kein Bodenmodell aufbauen.")

    ground_height_raster[~valid_cells_mask] = np.nan

    if number_of_empty_cells > 0:
        ground_height_raster = ground_height_raster[
            tuple(
                distance_transform_edt(
                    ~valid_cells_mask,
                    return_distances=False,
                    return_indices=True,
                )
            )
        ]

    if median_filter_size and median_filter_size > 1:
        ground_height_raster = median_filter(ground_height_raster, size=median_filter_size)

    print(
        f"    Belegte Zellen: {int(np.sum(valid_cells_mask)):,}  |  "
        f"Leere: {number_of_empty_cells:,}"
    )

    return {
        "x_min": x_min,
        "y_min": y_min,
        "cell_size": float(ground_grid_cell_size_m),
        "nx": number_of_cells_x,
        "ny": number_of_cells_y,
        "ground": ground_height_raster.astype(np.float32),
    }


def crop_point_cloud_by_height_above_local_ground(
    point_cloud: o3d.geometry.PointCloud,
    local_ground_model: dict,
    min_height_above_ground_m: float,
    max_height_above_ground_m: float,
    cloud_label: str = "",
) -> o3d.geometry.PointCloud:
    """
    Schneidet eine Punktwolke relativ zu einem lokalen Bodenmodell zu.

    Für jeden Punkt wird die Höhe über dem lokal geschätzten Boden berechnet.
    Behalten werden nur Punkte innerhalb des gewählten Höhenintervalls.
    """
    points_xyz = np.asarray(point_cloud.points)
    point_x = points_xyz[:, 0]
    point_y = points_xyz[:, 1]
    point_z = points_xyz[:, 2]

    cell_index_x = np.floor((point_x - local_ground_model["x_min"]) / local_ground_model["cell_size"]).astype(np.int32)
    cell_index_y = np.floor((point_y - local_ground_model["y_min"]) / local_ground_model["cell_size"]).astype(np.int32)

    points_outside_ground_model = (
        (cell_index_x < 0)
        | (cell_index_x >= local_ground_model["nx"])
        | (cell_index_y < 0)
        | (cell_index_y >= local_ground_model["ny"])
    )

    cell_index_x = np.clip(cell_index_x, 0, local_ground_model["nx"] - 1)
    cell_index_y = np.clip(cell_index_y, 0, local_ground_model["ny"] - 1)

    height_above_ground = point_z - local_ground_model["ground"][cell_index_x, cell_index_y]

    keep_point_mask = (
        (~points_outside_ground_model)
        & (height_above_ground >= min_height_above_ground_m)
        & (height_above_ground <= max_height_above_ground_m)
    )

    cropped_point_cloud = point_cloud.select_by_index(np.where(keep_point_mask)[0].tolist())

    number_of_points_before_crop = len(points_xyz)
    number_of_points_after_crop = len(cropped_point_cloud.points)
    optional_label_text = f" [{cloud_label}]" if cloud_label else ""

    print(
        f"  Boden-Crop{optional_label_text}: [{min_height_above_ground_m:.2f}, "
        f"{max_height_above_ground_m:.2f}] m  "
        f"{number_of_points_before_crop:,} -> {number_of_points_after_crop:,} "
        f"({number_of_points_after_crop / number_of_points_before_crop * 100:.1f}%)"
    )

    if number_of_points_after_crop == 0:
        sys.exit(f"Nach Boden-Crop{optional_label_text} sind keine Punkte übrig.")

    return cropped_point_cloud

# ════════════════════════════════════════════════════════════════════════
# C2C-DISTANZEN
# ════════════════════════════════════════════════════════════════════════

def compute_c2c_nearest_neighbor_distances(
    reference_point_cloud: o3d.geometry.PointCloud,
    comparison_point_cloud: o3d.geometry.PointCloud,
) -> np.ndarray:
    """
    Berechnet für jeden Punkt der Vergleichswolke den Abstand zum nächstgelegenen
    Punkt der Referenzwolke.

    Das Ergebnis ist ein eindimensionales Array mit den absoluten C2C-Distanzen.
    """
    reference_points_xyz = np.asarray(reference_point_cloud.points)
    comparison_points_xyz = np.asarray(comparison_point_cloud.points)

    print(f"\n  Baue KD-Tree ({len(reference_points_xyz):,} Referenzpunkte)...")
    start_time = time.time()

    reference_kd_tree = cKDTree(reference_points_xyz)

    print(f"  Berechne Abstände für {len(comparison_points_xyz):,} Punkte...")
    nearest_neighbor_distances, _ = reference_kd_tree.query(comparison_points_xyz, k=1, workers=-1)

    print(f"    -> fertig in {time.time() - start_time:.2f}s")
    return nearest_neighbor_distances

# ════════════════════════════════════════════════════════════════════════
# SIGNIERTE ÄNDERUNGSMETRIK
# ════════════════════════════════════════════════════════════════════════

def compute_signed_change_indicator_from_c2c_and_dsm(
    comparison_points_xyz: np.ndarray,
    absolute_c2c_distances_m: np.ndarray,
    working_reference_surface_dsm: dict,
) -> np.ndarray:
    """
    Berechnet den signierten Änderungsindikator.

    Definition:
    - Betrag   = absolute C2C-Distanz
    - Vorzeichen = Vorzeichen von (Vergleichs-Z - Referenzoberfläche im DSM)

    Damit gilt:
    - positiver Wert -> Vergleichspunkt liegt über der Referenzoberfläche
    - negativer Wert -> Vergleichspunkt liegt unter der Referenzoberfläche

    Hinweis:
    Dies ist absichtlich ein Hybridmaß und kein reines Höhen-Delta.
    """
    start_time = time.time()

    comparison_x = comparison_points_xyz[:, 0]
    comparison_y = comparison_points_xyz[:, 1]
    comparison_z = comparison_points_xyz[:, 2]

    dsm_x_min = working_reference_surface_dsm["x_min"]
    dsm_y_min = working_reference_surface_dsm["y_min"]
    dsm_cell_size = working_reference_surface_dsm["cell_size"]
    dsm_number_of_cells_x = working_reference_surface_dsm["nx"]
    dsm_number_of_cells_y = working_reference_surface_dsm["ny"]
    reference_surface_z = working_reference_surface_dsm["surface"]

    comparison_cell_index_x = np.clip(
        np.floor((comparison_x - dsm_x_min) / dsm_cell_size).astype(np.int32),
        0,
        dsm_number_of_cells_x - 1,
    )
    comparison_cell_index_y = np.clip(
        np.floor((comparison_y - dsm_y_min) / dsm_cell_size).astype(np.int32),
        0,
        dsm_number_of_cells_y - 1,
    )

    sign_of_change = np.sign(comparison_z - reference_surface_z[comparison_cell_index_x, comparison_cell_index_y])
    signed_change_indicator_m = absolute_c2c_distances_m * sign_of_change

    print(f"    -> Signed DSM fertig in {time.time() - start_time:.2f}s")
    return signed_change_indicator_m
# ════════════════════════════════════════════════════════════════════════
# KLASSIFIKATION DER ÄNDERUNGEN
# ════════════════════════════════════════════════════════════════════════

def classify_unsigned_change_magnitude(
    distances_m: np.ndarray,
    threshold_t1_m: float,
    threshold_t2_m: float,
) -> np.ndarray:
    """
    Klassifiziert die absolute Änderungsstärke ohne Richtungsinformation.

    Ausgabe:
    - 0 = unchanged
    - 1 = possible
    - 2 = likely
    """
    absolute_distances_m = np.abs(distances_m)

    unsigned_change_classes = np.zeros(len(distances_m), dtype=np.int8)
    unsigned_change_classes[absolute_distances_m > threshold_t1_m] = 1
    unsigned_change_classes[absolute_distances_m > threshold_t2_m] = 2

    return unsigned_change_classes


def classify_signed_change_direction_and_magnitude(
    signed_change_indicator_m: np.ndarray,
    threshold_t1_m: float,
    threshold_t2_m: float,
) -> np.ndarray:
    """
    Klassifiziert signierte Änderungen nach Richtung und Stärke.

    Ausgabe:
    -2 = likely removed
    -1 = possibly removed
     0 = unchanged
     1 = possibly added
     2 = likely added
    """
    signed_change_classes = np.zeros(len(signed_change_indicator_m), dtype=np.int8)

    signed_change_classes[signed_change_indicator_m > threshold_t1_m] = 1
    signed_change_classes[signed_change_indicator_m > threshold_t2_m] = 2
    signed_change_classes[signed_change_indicator_m < -threshold_t1_m] = -1
    signed_change_classes[signed_change_indicator_m < -threshold_t2_m] = -2

    return signed_change_classes

# ════════════════════════════════════════════════════════════════════════
# STATISTIK-HILFSFUNKTIONEN
# ════════════════════════════════════════════════════════════════════════

def compute_distribution_statistics_for_subset(values: np.ndarray) -> dict:
    """
    Berechnet einfache Verteilungskennzahlen für ein Teilset von Werten.

    Falls das Teilset leer ist, werden NaN-Werte zurückgegeben, damit die
    weitere Verarbeitung nicht fehlschlägt.
    """
    if len(values) == 0:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan}

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compute_raster_based_change_area_statistics(
    point_xy_coordinates: np.ndarray,
    change_classes: np.ndarray,
    raster_cell_size_m: float,
) -> dict:
    """
    Schätzt flächenbezogene Änderungskennzahlen auf Rasterbasis.

    Vorgehen:
    - Alle Punkte werden in ein 2D-Raster projiziert.
    - Pro Rasterzelle wird die maximale absolute Änderungsstufe gespeichert.
    - Dadurch kann eine grobe betroffene Fläche abgeschätzt werden.

    Hinweis:
    Dies ist eine Raster-Proxy-Fläche und keine exakte Objektfläche.
    """
    point_x = point_xy_coordinates[:, 0]
    point_y = point_xy_coordinates[:, 1]

    raster_index_x = np.floor((point_x - float(point_x.min())) / raster_cell_size_m).astype(np.int64)
    raster_index_y = np.floor((point_y - float(point_y.min())) / raster_cell_size_m).astype(np.int64)

    combined_cell_ids = raster_index_x * 1_000_003 + raster_index_y
    unique_cell_ids, inverse_cell_indices = np.unique(combined_cell_ids, return_inverse=True)

    number_of_cells = len(unique_cell_ids)
    absolute_change_classes = np.abs(change_classes).astype(np.int8)

    maximum_change_class_per_cell = np.zeros(number_of_cells, dtype=np.int8)
    np.maximum.at(maximum_change_class_per_cell, inverse_cell_indices, absolute_change_classes)

    raster_cell_area_m2 = raster_cell_size_m ** 2

    return {
        "total_cells": number_of_cells,
        "unchanged": int(np.sum(maximum_change_class_per_cell == 0)),
        "possible": int(np.sum(maximum_change_class_per_cell == 1)),
        "likely": int(np.sum(maximum_change_class_per_cell == 2)),
        "changed": int(np.sum(maximum_change_class_per_cell >= 1)),
        "changed_pct": 100.0 * np.sum(maximum_change_class_per_cell >= 1) / number_of_cells if number_of_cells else 0.0,
        "likely_pct": 100.0 * np.sum(maximum_change_class_per_cell == 2) / number_of_cells if number_of_cells else 0.0,
        "total_area": number_of_cells * raster_cell_area_m2,
        "changed_area": np.sum(maximum_change_class_per_cell >= 1) * raster_cell_area_m2,
        "likely_area": np.sum(maximum_change_class_per_cell == 2) * raster_cell_area_m2,
    }

# ════════════════════════════════════════════════════════════════════════
# DETAILSTATISTIKEN FÜR DEN LAUF
# ════════════════════════════════════════════════════════════════════════

def compute_absolute_distance_statistics(absolute_or_unsigned_distances_m: np.ndarray) -> dict:
    """
    Berechnet klassische Distanzstatistiken für die absoluten C2C-Abstände.

    Diese Statistik ist bewusst ausführlicher als die später exportierte Core-CSV,
    weil sie auch für die Konsolenausgabe nützlich ist.
    """
    absolute_distances_m = np.abs(absolute_or_unsigned_distances_m)

    statistics = {
        "Anzahl Punkte": len(absolute_or_unsigned_distances_m),
        "Mittlerer Abstand": np.mean(absolute_distances_m),
        "Median Abstand": np.median(absolute_distances_m),
        "Standardabweichung": np.std(absolute_distances_m),
        "RMS": np.sqrt(np.mean(absolute_or_unsigned_distances_m ** 2)),
        "Min Abstand": np.min(absolute_distances_m),
        "Max Abstand": np.max(absolute_distances_m),
        "Perzentil  5%": np.percentile(absolute_distances_m, 5),
        "Perzentil 25%": np.percentile(absolute_distances_m, 25),
        "Perzentil 50%": np.percentile(absolute_distances_m, 50),
        "Perzentil 75%": np.percentile(absolute_distances_m, 75),
        "Perzentil 90%": np.percentile(absolute_distances_m, 90),
        "Perzentil 95%": np.percentile(absolute_distances_m, 95),
        "Perzentil 98%": np.percentile(absolute_distances_m, 98),
        "Perzentil 99%": np.percentile(absolute_distances_m, 99),
    }

    for threshold_m in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]:
        percentage_below_threshold = 100.0 * np.sum(absolute_distances_m <= threshold_m) / len(absolute_distances_m)
        statistics[f"Punkte <= {threshold_m:.3f}m"] = f"{percentage_below_threshold:.2f}%"

    return statistics


def compute_unsigned_change_statistics(
    absolute_c2c_distances_m: np.ndarray,
    unsigned_change_classes: np.ndarray,
    threshold_t1_m: float,
    threshold_t2_m: float,
    point_xy_coordinates: np.ndarray,
    raster_cell_size_m: float,
) -> dict:
    """
    Berechnet Statistiken zur unsignierten Änderungsstärke.

    Dazu gehören:
    - Klassenanteile auf Punktebene
    - Verteilungskennzahlen pro Klasse
    - Rasterbasierte Flächenkennzahlen
    """
    absolute_distances_m = np.abs(absolute_c2c_distances_m)
    number_of_points = len(unsigned_change_classes)

    statistics = {
        "T1 (unchanged/possible)": threshold_t1_m,
        "T2 (possible/likely)": threshold_t2_m,
    }

    for class_id, class_name in UNSIGNED_CHANGE_CLASS_NAMES.items():
        class_mask = unsigned_change_classes == class_id
        class_count = int(np.sum(class_mask))

        statistics[f"Klasse {class_id} ({class_name}) Anzahl"] = class_count
        statistics[f"Klasse {class_id} ({class_name}) Anteil"] = (
            f"{100.0 * class_count / number_of_points:.2f}%" if number_of_points else "0.00%"
        )

        class_distribution_statistics = compute_distribution_statistics_for_subset(absolute_distances_m[class_mask])
        for metric_name, metric_value in class_distribution_statistics.items():
            statistics[f"Klasse {class_id} ({class_name}) {metric_name}"] = metric_value

    number_of_changed_points = int(np.sum(unsigned_change_classes >= 1))
    number_of_likely_changed_points = int(np.sum(unsigned_change_classes == 2))

    statistics["Changed Ratio (possible+likely)"] = (
        f"{100.0 * number_of_changed_points / number_of_points:.2f}%" if number_of_points else "0.00%"
    )
    statistics["Changed Ratio (nur likely)"] = (
        f"{100.0 * number_of_likely_changed_points / number_of_points:.2f}%" if number_of_points else "0.00%"
    )

    raster_area_statistics = compute_raster_based_change_area_statistics(
        point_xy_coordinates,
        unsigned_change_classes,
        raster_cell_size_m,
    )

    statistics.update({
        "Raster Zellgrösse [m]": raster_cell_size_m,
        "Raster Gesamtzellen": raster_area_statistics["total_cells"],
        "Raster unchanged Zellen": raster_area_statistics["unchanged"],
        "Raster possible Zellen": raster_area_statistics["possible"],
        "Raster likely Zellen": raster_area_statistics["likely"],
        "Raster changed Zellen (possible+likely)": raster_area_statistics["changed"],
        "Raster changed Anteil": f"{raster_area_statistics['changed_pct']:.2f}%",
        "Raster likely Anteil": f"{raster_area_statistics['likely_pct']:.2f}%",
        "Raster Gesamtfläche [m^2]": raster_area_statistics["total_area"],
        "Raster changed Fläche [m^2]": raster_area_statistics["changed_area"],
        "Raster likely Fläche [m^2]": raster_area_statistics["likely_area"],
    })

    return statistics


def compute_signed_change_statistics(
    signed_change_indicator_m: np.ndarray,
    signed_change_classes: np.ndarray,
    threshold_t1_m: float,
    threshold_t2_m: float,
    point_xy_coordinates: np.ndarray,
    raster_cell_size_m: float,
) -> dict:
    """
    Berechnet Statistiken für die signierte Änderungsdarstellung.

    Dazu gehören:
    - Klassenanteile pro signed Klasse
    - Verteilungskennzahlen pro signed Klasse
    - zusammenfassende positive / negative Änderungsanteile
    - mittlere signed Änderung
    - rasterbasierte positive und negative Änderungsflächen
    """
    number_of_points = len(signed_change_classes)
    absolute_signed_change_indicator_m = np.abs(signed_change_indicator_m)

    statistics = {
        "Signed Methode": "dsm",
        "T1 (unchanged/possible)": threshold_t1_m,
        "T2 (possible/likely)": threshold_t2_m,
    }

    for class_id, class_name in SIGNED_CHANGE_CLASS_NAMES.items():
        class_mask = signed_change_classes == class_id
        class_count = int(np.sum(class_mask))

        statistics[f"[Signed] Klasse {class_id:+d} ({class_name}) Anzahl"] = class_count
        statistics[f"[Signed] Klasse {class_id:+d} ({class_name}) Anteil"] = (
            f"{100.0 * class_count / number_of_points:.2f}%" if number_of_points else "0.00%"
        )

        class_distribution_statistics = compute_distribution_statistics_for_subset(
            absolute_signed_change_indicator_m[class_mask]
        )
        for metric_name, metric_value in class_distribution_statistics.items():
            statistics[f"[Signed] Klasse {class_id:+d} ({class_name}) {metric_name}"] = metric_value

    number_of_added_points = int(np.sum(signed_change_classes >= 1))
    number_of_removed_points = int(np.sum(signed_change_classes <= -1))
    number_of_likely_added_points = int(np.sum(signed_change_classes == 2))
    number_of_likely_removed_points = int(np.sum(signed_change_classes == -2))

    statistics["[Signed] Added   Ratio (cls +1/+2)"] = (
        f"{100.0 * number_of_added_points / number_of_points:.2f}%" if number_of_points else "0.00%"
    )
    statistics["[Signed] Removed Ratio (cls -1/-2)"] = (
        f"{100.0 * number_of_removed_points / number_of_points:.2f}%" if number_of_points else "0.00%"
    )
    statistics["[Signed] Likely Added   Ratio"] = (
        f"{100.0 * number_of_likely_added_points / number_of_points:.2f}%" if number_of_points else "0.00%"
    )
    statistics["[Signed] Likely Removed Ratio"] = (
        f"{100.0 * number_of_likely_removed_points / number_of_points:.2f}%" if number_of_points else "0.00%"
    )

    statistics["[Signed] Mittlere signed Distanz (alle)"] = float(np.mean(signed_change_indicator_m))
    statistics["[Signed] Mittlere signed Distanz (|d|>T1)"] = (
        float(np.mean(signed_change_indicator_m[absolute_signed_change_indicator_m > threshold_t1_m]))
        if np.any(absolute_signed_change_indicator_m > threshold_t1_m)
        else np.nan
    )

    point_x = point_xy_coordinates[:, 0]
    point_y = point_xy_coordinates[:, 1]

    raster_index_x = np.floor((point_x - float(point_x.min())) / raster_cell_size_m).astype(np.int64)
    raster_index_y = np.floor((point_y - float(point_y.min())) / raster_cell_size_m).astype(np.int64)
    combined_cell_ids = raster_index_x * 1_000_003 + raster_index_y

    unique_cell_ids, inverse_cell_indices = np.unique(combined_cell_ids, return_inverse=True)
    number_of_cells = len(unique_cell_ids)

    maximum_positive_change_class_per_cell = np.full(number_of_cells, -99, dtype=np.int8)
    minimum_negative_change_class_per_cell = np.full(number_of_cells, 99, dtype=np.int8)

    np.maximum.at(maximum_positive_change_class_per_cell, inverse_cell_indices, signed_change_classes)
    np.minimum.at(minimum_negative_change_class_per_cell, inverse_cell_indices, signed_change_classes)

    number_of_cells_with_positive_change = int(np.sum(maximum_positive_change_class_per_cell >= 1))
    number_of_cells_with_negative_change = int(np.sum(minimum_negative_change_class_per_cell <= -1))
    raster_cell_area_m2 = raster_cell_size_m ** 2

    statistics.update({
        "[Signed] Raster Zellgrösse [m]": raster_cell_size_m,
        "[Signed] Raster Gesamtzellen": number_of_cells,
        "[Signed] Raster added Zellen (cls +1/+2)": number_of_cells_with_positive_change,
        "[Signed] Raster removed Zellen (cls -1/-2)": number_of_cells_with_negative_change,
        "[Signed] Raster added   Fläche [m^2]": number_of_cells_with_positive_change * raster_cell_area_m2,
        "[Signed] Raster removed Fläche [m^2]": number_of_cells_with_negative_change * raster_cell_area_m2,
    })

    return statistics

# ════════════════════════════════════════════════════════════════════════
# KONSOLENAUSGABE DER STATISTIKEN
# ════════════════════════════════════════════════════════════════════════

def print_absolute_distance_statistics(statistics: dict) -> None:
    """Gibt die allgemeinen Distanzstatistiken formatiert in der Konsole aus."""
    print("\n" + "═" * 60)
    print("  C2C VERGLEICH — DISTANZ-STATISTIKEN")
    print("═" * 60)

    for metric_name in [
        "Anzahl Punkte",
        "Mittlerer Abstand",
        "Median Abstand",
        "Standardabweichung",
        "RMS",
        "Min Abstand",
        "Max Abstand",
    ]:
        metric_value = statistics[metric_name]
        if isinstance(metric_value, float):
            print(f"  {metric_name:<28}: {metric_value:>12.6f} m")
        else:
            print(f"  {metric_name:<28}: {metric_value:>12,}")

    print("\n  Perzentile:")
    for metric_name, metric_value in statistics.items():
        if metric_name.startswith("Perzentil"):
            print(f"  {metric_name:<28}: {metric_value:>12.6f} m")

    print("\n  Schwellwerte:")
    for metric_name, metric_value in statistics.items():
        if metric_name.startswith("Punkte <="):
            print(f"  {metric_name:<28}: {metric_value:>12s}")

    print()


def print_unsigned_change_statistics(statistics: dict) -> None:
    """Gibt die unsignierten Änderungsstatistiken formatiert in der Konsole aus."""
    print("\n" + "═" * 60)
    print("  UNSIGNED CHANGE-KLASSIFIKATION — STATISTIKEN")
    print("═" * 60)
    print(
        f"  T1 = {statistics['T1 (unchanged/possible)']:.4f} m  |  "
        f"T2 = {statistics['T2 (possible/likely)']:.4f} m\n"
    )

    print(f"  {'Kls':<5} {'Name':<12} {'Anzahl':>12} {'Anteil':>10} {'Mean':>8} {'Median':>8}")
    print("  " + "-" * 57)

    for class_id, class_name in UNSIGNED_CHANGE_CLASS_NAMES.items():
        format_or_na = lambda value: f"{value:.4f}" if not np.isnan(value) else "n/a"
        print(
            f"  {class_id:<5} {class_name:<12} "
            f"{statistics[f'Klasse {class_id} ({class_name}) Anzahl']:>12,} "
            f"{statistics[f'Klasse {class_id} ({class_name}) Anteil']:>10} "
            f"{format_or_na(statistics[f'Klasse {class_id} ({class_name}) mean']):>8} "
            f"{format_or_na(statistics[f'Klasse {class_id} ({class_name}) median']):>8}"
        )

    raster_cell_size_m = statistics["Raster Zellgrösse [m]"]
    changed_area_m2 = statistics["Raster changed Fläche [m^2]"]

    print(f"\n  Changed (pos+lik): {statistics['Changed Ratio (possible+likely)']}")
    print(
        f"  Raster ({raster_cell_size_m:.1f}m): "
        f"{statistics['Raster changed Zellen (possible+likely)']:,} Zellen "
        f"({statistics['Raster changed Anteil']}) | {changed_area_m2:,.1f} m²\n"
    )


def print_signed_change_statistics(statistics: dict) -> None:
    """Gibt die signierten Änderungsstatistiken formatiert in der Konsole aus."""
    print("\n" + "═" * 60)
    print("  SIGNED CHANGE-MAP — STATISTIKEN  [DSM]")
    print("═" * 60)
    print(
        f"  T1 = {statistics['T1 (unchanged/possible)']:.4f} m  |  "
        f"T2 = {statistics['T2 (possible/likely)']:.4f} m\n"
    )

    print(f"  {'Kls':>4} {'Name':<18} {'Anzahl':>12} {'Anteil':>10} {'Mean':>8} {'Median':>8}")
    print("  " + "-" * 62)

    for class_id, class_name in SIGNED_CHANGE_CLASS_NAMES.items():
        class_key_prefix = f"[Signed] Klasse {class_id:+d} ({class_name})"
        format_or_na = lambda value: f"{value:.4f}" if not np.isnan(value) else "n/a"
        class_symbol = "▲" if class_id > 0 else ("▼" if class_id < 0 else "─")

        print(
            f"  {class_symbol}{class_id:>3} {class_name:<18} "
            f"{statistics[f'{class_key_prefix} Anzahl']:>12,} "
            f"{statistics[f'{class_key_prefix} Anteil']:>10} "
            f"{format_or_na(statistics[f'{class_key_prefix} mean']):>8} "
            f"{format_or_na(statistics[f'{class_key_prefix} median']):>8}"
        )

    mean_signed_change_m = statistics["[Signed] Mittlere signed Distanz (alle)"]
    raster_cell_size_m = statistics["[Signed] Raster Zellgrösse [m]"]
    added_area_m2 = statistics["[Signed] Raster added   Fläche [m^2]"]
    removed_area_m2 = statistics["[Signed] Raster removed Fläche [m^2]"]

    print(
        f"\n  Netto Ø signed:  {mean_signed_change_m:+.6f} m  "
        f"({'Netto-Zunahme' if mean_signed_change_m > 0 else 'Netto-Abnahme'})"
    )
    print(
        f"  Raster ({raster_cell_size_m:.1f}m): added "
        f"{statistics['[Signed] Raster added Zellen (cls +1/+2)']:,} ({added_area_m2:,.1f} m²) | "
        f"removed {statistics['[Signed] Raster removed Zellen (cls -1/-2)']:,} ({removed_area_m2:,.1f} m²)\n"
    )

# ════════════════════════════════════════════════════════════════════════
# EXPORTFUNKTIONEN
# ════════════════════════════════════════════════════════════════════════

def write_rgb_point_cloud_to_ply(
    output_file_path: str,
    points_xyz: np.ndarray,
    point_colors_rgb: np.ndarray,
) -> None:
    """
    Schreibt eine farbcodierte Punktwolke als ASCII-PLY.

    Erwartet:
    - points_xyz      : N x 3 Koordinaten
    - point_colors_rgb: N x 3 Farben im Bereich 0..1
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    point_colors_rgb = np.asarray(point_colors_rgb, dtype=np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)

    output_point_cloud = o3d.geometry.PointCloud()
    output_point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    output_point_cloud.colors = o3d.utility.Vector3dVector(np.clip(point_colors_rgb, 0, 1))

    o3d.io.write_point_cloud(output_file_path, output_point_cloud, write_ascii=True)


def export_signed_change_map_as_rgb_point_clouds(
    output_file_path_all_points: str,
    output_file_path_changed_points_only: str,
    output_points_xyz: np.ndarray,
    signed_change_classes: np.ndarray,
    color_likely_removed: tuple,
    color_possibly_removed: tuple,
    color_unchanged: tuple,
    color_possibly_added: tuple,
    color_likely_added: tuple,
) -> None:
    """
    Exportiert die signierte Änderungsdarstellung als zwei PLY-Dateien:

    1. alle Punkte mit Farbkodierung
    2. nur geänderte Punkte (|Klasse| >= 1)

    Farben:
    - blau  = entfernt / niedriger
    - grau  = unverändert
    - rot/orange = hinzugekommen / höher
    """
    signed_change_classes = np.asarray(signed_change_classes, dtype=np.int8)
    all_point_colors_rgb = np.zeros((len(output_points_xyz), 3), dtype=np.float64)

    all_point_colors_rgb[signed_change_classes == -2] = color_likely_removed
    all_point_colors_rgb[signed_change_classes == -1] = color_possibly_removed
    all_point_colors_rgb[signed_change_classes == 0] = color_unchanged
    all_point_colors_rgb[signed_change_classes == 1] = color_possibly_added
    all_point_colors_rgb[signed_change_classes == 2] = color_likely_added

    write_rgb_point_cloud_to_ply(output_file_path_all_points, output_points_xyz, all_point_colors_rgb)
    print(f"  Signed RGB [alle]: {os.path.basename(output_file_path_all_points)}  ({len(output_points_xyz):,} Punkte)")

    changed_point_mask = np.abs(signed_change_classes) >= 1
    write_rgb_point_cloud_to_ply(
        output_file_path_changed_points_only,
        output_points_xyz[changed_point_mask],
        all_point_colors_rgb[changed_point_mask],
    )
    print(
        f"  Signed RGB [chg]:  {os.path.basename(output_file_path_changed_points_only)}  "
        f"({int(np.sum(changed_point_mask)):,} Punkte)"
    )


def export_histogram_of_changed_signed_points(
    signed_change_indicator_m: np.ndarray,
    signed_change_classes: np.ndarray,
    threshold_t1_m: float,
    threshold_t2_m: float,
    output_file_path: str,
    color_likely_removed: tuple,
    color_possibly_removed: tuple,
    color_possibly_added: tuple,
    color_likely_added: tuple,
) -> None:
    """
    Exportiert ein Histogramm nur für geänderte Punkte, also für |d| > T1.

    Zweck:
    - Die unveränderten Punkte werden ausgeblendet.
    - Dadurch wird die Verteilung der positiven und negativen Änderungen klarer sichtbar.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    changed_point_mask = np.abs(signed_change_indicator_m) > threshold_t1_m
    if not np.any(changed_point_mask):
        print("  Signed-Histogramm [changed only] übersprungen: keine Punkte > T1")
        return

    signed_values_changed_only = signed_change_indicator_m[changed_point_mask]
    signed_classes_changed_only = signed_change_classes[changed_point_mask]

    symmetric_histogram_range = float(
        max(abs(signed_values_changed_only.min()), abs(signed_values_changed_only.max())) * 1.05
    )
    histogram_bins = np.linspace(-symmetric_histogram_range, symmetric_histogram_range, 81)

    figure, axis = plt.subplots(figsize=(10, 5))

    histogram_class_info = [
        (-2, "likely removed", color_likely_removed),
        (-1, "possibly removed", color_possibly_removed),
        (1, "possibly added", color_possibly_added),
        (2, "likely added", color_likely_added),
    ]

    for class_id, class_name, class_color in histogram_class_info:
        class_mask = signed_classes_changed_only == class_id
        if np.any(class_mask):
            axis.hist(
                signed_values_changed_only[class_mask],
                bins=histogram_bins,
                alpha=0.75,
                color=class_color,
                label=f"{class_id:+d} {class_name}  n={np.sum(class_mask):,}",
                edgecolor="none",
            )

    axis.axvline(0.0, color="black", linestyle="-", linewidth=1.0)
    axis.axvline(threshold_t1_m, color="black", linestyle="--", linewidth=1.0, label=f"±T1={threshold_t1_m:.3f}m")
    axis.axvline(-threshold_t1_m, color="black", linestyle="--", linewidth=1.0)
    axis.axvline(threshold_t2_m, color="gray", linestyle=":", linewidth=1.0, label=f"±T2={threshold_t2_m:.3f}m")
    axis.axvline(-threshold_t2_m, color="gray", linestyle=":", linewidth=1.0)

    mean_signed_value_changed_only = float(np.mean(signed_values_changed_only))
    axis.axvline(
        mean_signed_value_changed_only,
        color="red",
        linestyle="-.",
        linewidth=1.5,
        label=f"Mittel: {mean_signed_value_changed_only:+.4f} m",
    )

    axis.set_xlabel("signed d [m]")
    axis.set_ylabel("Anzahl Punkte")
    axis.set_title(f"Signed Change – nur geänderte Punkte (|d| > {threshold_t1_m:.3f} m)")
    axis.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file_path, dpi=150)
    plt.close()
    print(f"  Signed-Hist [chg]: {os.path.basename(output_file_path)}")


def export_signed_class_share_bar_chart(
    signed_change_classes: np.ndarray,
    output_file_path: str,
    color_likely_removed: tuple,
    color_possibly_removed: tuple,
    color_unchanged: tuple,
    color_possibly_added: tuple,
    color_likely_added: tuple,
) -> None:
    """
    Exportiert ein einfaches horizontales Balkendiagramm der Klassenanteile.

    Dieses Diagramm beantwortet die Frage:
        Wie viel Prozent der Punkte gehören zu welcher signed Klasse?
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    number_of_points_total = len(signed_change_classes)
    if number_of_points_total == 0:
        print("  Klassen-Anteile übersprungen: keine Punkte vorhanden")
        return

    chart_class_info = [
        (2, "+2 likely added", color_likely_added),
        (1, "+1 possibly added", color_possibly_added),
        (0, "+0 unchanged", color_unchanged),
        (-1, "-1 possibly removed", color_possibly_removed),
        (-2, "-2 likely removed", color_likely_removed),
    ]

    class_names = []
    class_percentages = []
    class_counts = []
    class_colors = []

    for class_id, class_name, class_color in chart_class_info:
        class_count = int(np.sum(signed_change_classes == class_id))
        class_percentage = 100.0 * class_count / number_of_points_total

        class_names.append(class_name)
        class_percentages.append(class_percentage)
        class_counts.append(class_count)
        class_colors.append(class_color)

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.barh(class_names, class_percentages, color=class_colors, edgecolor="white", linewidth=0.5)

    for bar, class_percentage, class_count in zip(bars, class_percentages, class_counts):
        axis.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{class_percentage:.1f}%  ({class_count:,})",
            va="center",
            fontsize=9,
        )

    axis.set_xlabel("Anteil [%]")
    axis.set_title("Klassen-Anteile")
    axis.set_xlim(0, max(class_percentages) * 1.5 + 2)

    plt.tight_layout()
    plt.savefig(output_file_path, dpi=150)
    plt.close()
    print(f"  Klassen-Plot:      {os.path.basename(output_file_path)}")


def export_core_metrics_csv(
    absolute_distance_statistics: dict,
    unsigned_change_statistics: dict,
    signed_change_statistics: dict,
    output_file_path: str,
    config: SimpleNamespace,
    result_mode_label: str,
) -> None:
    """
    Exportiert genau die Kernkennzahlen, die später im Ergebnisteil benötigt werden.

    Die CSV ist bewusst klein gehalten und soll alle Datensätze leicht
    vergleichbar machen.
    """
    core_metric_row = {
        "Referenz": os.path.basename(config.reference),
        "Vergleich": os.path.basename(config.compared),
        "Modus": result_mode_label,
        "T1_m": config.change_t1,
        "T2_m": config.change_t2,
        "Punkte_n": absolute_distance_statistics["Anzahl Punkte"],
        "MedianAbs_m": absolute_distance_statistics["Median Abstand"],
        "P95Abs_m": absolute_distance_statistics["Perzentil 95%"],
        "Changed_pct": unsigned_change_statistics["Changed Ratio (possible+likely)"],
        "LikelyAdded_pct": signed_change_statistics["[Signed] Likely Added   Ratio"],
        "LikelyRemoved_pct": signed_change_statistics["[Signed] Likely Removed Ratio"],
    }

    with open(output_file_path, "w", encoding="utf-8", newline="") as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=list(core_metric_row.keys()), delimiter=";")
        csv_writer.writeheader()
        csv_writer.writerow(core_metric_row)

    print(f"  Core-CSV:          {os.path.basename(output_file_path)}")

# ════════════════════════════════════════════════════════════════════════
# ZENTRALE VERARBEITUNGSPIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_c2c_change_detection_pipeline(
    reference_point_cloud: o3d.geometry.PointCloud,
    comparison_point_cloud: o3d.geometry.PointCloud,
    config: SimpleNamespace,
    output_file_prefix: str,
    working_reference_surface_dsm: dict,
    processing_label: str = "",
    export_point_cloud_with_original_z=None,
):
    """
    Führt die eigentliche C2C-Änderungsanalyse für ein Punktwolkenpaar aus.

    Ablauf innerhalb der Pipeline:
    1. C2C-Distanzen berechnen
    2. Distanzstatistiken berechnen
    3. unsignierte Änderung klassifizieren und auswerten
    4. signierte Änderung mit DSM-Vorzeichen bestimmen und auswerten
    5. Exporte schreiben

    Rückgabe:
    - diverse Statistik-Dictionaries
    - Distanzarrays
    - Klassenarrays
    """
    processing_tag_text = f" [{processing_label}]" if processing_label else ""

    if export_point_cloud_with_original_z is None:
        export_point_cloud_with_original_z = comparison_point_cloud

    comparison_points_xyz = np.asarray(comparison_point_cloud.points)
    export_points_xyz = np.asarray(export_point_cloud_with_original_z.points)
    optional_suffix = f"_{processing_label}" if processing_label else ""
    output_directory = config.output_dir

    # 1. C2C-Distanzen
    print(f"\n━━━ C2C-Abstände{processing_tag_text} ━━━")
    absolute_c2c_distances_m = compute_c2c_nearest_neighbor_distances(
        reference_point_cloud,
        comparison_point_cloud,
    )
    absolute_c2c_distances_for_signing_m = np.abs(absolute_c2c_distances_m)

    # 2. Distanzstatistiken
    print(f"\n━━━ Distanz-Statistiken{processing_tag_text} ━━━")
    absolute_distance_statistics = compute_absolute_distance_statistics(absolute_c2c_distances_m)
    print_absolute_distance_statistics(absolute_distance_statistics)

    # 3. Unsignierte Änderung
    print(f"\n━━━ Unsigned Change{processing_tag_text} ━━━")
    unsigned_change_classes = classify_unsigned_change_magnitude(
        absolute_c2c_distances_m,
        config.change_t1,
        config.change_t2,
    )
    unsigned_change_statistics = compute_unsigned_change_statistics(
        absolute_c2c_distances_m,
        unsigned_change_classes,
        config.change_t1,
        config.change_t2,
        export_points_xyz[:, :2],
        config.change_grid_size,
    )
    print_unsigned_change_statistics(unsigned_change_statistics)

    # 4. Signierte Änderung
    print(
        f"\n━━━ Signed Change{processing_tag_text}  "
        f"[DSM, cell={working_reference_surface_dsm['cell_size']:.3f}m] ━━━"
    )
    signed_change_indicator_m = compute_signed_change_indicator_from_c2c_and_dsm(
        comparison_points_xyz,
        absolute_c2c_distances_for_signing_m,
        working_reference_surface_dsm,
    )
    signed_change_classes = classify_signed_change_direction_and_magnitude(
        signed_change_indicator_m,
        config.change_t1,
        config.change_t2,
    )
    signed_change_statistics = compute_signed_change_statistics(
        signed_change_indicator_m,
        signed_change_classes,
        config.change_t1,
        config.change_t2,
        export_points_xyz[:, :2],
        config.change_grid_size,
    )
    print_signed_change_statistics(signed_change_statistics)

    # 5. Exporte
    print(f"\n━━━ Exports{processing_tag_text} ━━━")

    export_signed_change_map_as_rgb_point_clouds(
        output_file_path_all_points=os.path.join(
            output_directory,
            f"{output_file_prefix}_signed_changemap_rgb{optional_suffix}.ply",
        ),
        output_file_path_changed_points_only=os.path.join(
            output_directory,
            f"{output_file_prefix}_signed_changemap_changed{optional_suffix}.ply",
        ),
        output_points_xyz=export_points_xyz,
        signed_change_classes=signed_change_classes,
        color_likely_removed=config.signed_color_likely_removed,
        color_possibly_removed=config.signed_color_possible_removed,
        color_unchanged=config.signed_color_unchanged,
        color_possibly_added=config.signed_color_possible_added,
        color_likely_added=config.signed_color_likely_added,
    )

    export_histogram_of_changed_signed_points(
        signed_change_indicator_m,
        signed_change_classes,
        config.change_t1,
        config.change_t2,
        os.path.join(output_directory, f"{output_file_prefix}_signed_histogram_changed_only{optional_suffix}.png"),
        config.signed_color_likely_removed,
        config.signed_color_possible_removed,
        config.signed_color_possible_added,
        config.signed_color_likely_added,
    )

    export_signed_class_share_bar_chart(
        signed_change_classes,
        os.path.join(output_directory, f"{output_file_prefix}_signed_class_shares{optional_suffix}.png"),
        config.signed_color_likely_removed,
        config.signed_color_possible_removed,
        config.signed_color_unchanged,
        config.signed_color_possible_added,
        config.signed_color_likely_added,
    )

    export_core_metrics_csv(
        absolute_distance_statistics,
        unsigned_change_statistics,
        signed_change_statistics,
        os.path.join(output_directory, f"{output_file_prefix}_core_metrics{optional_suffix}.csv"),
        config,
        processing_label or "full",
    )

    return (
        absolute_distance_statistics,
        unsigned_change_statistics,
        signed_change_statistics,
        absolute_c2c_distances_m,
        signed_change_indicator_m,
        unsigned_change_classes,
        signed_change_classes,
    )

# ════════════════════════════════════════════════════════════════════════
# HAUPTPROGRAMM
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Startpunkt des Skripts."""
    config = SimpleNamespace(
        reference=REFERENCE_POINT_CLOUD_PATH,
        compared=COMPARISON_POINT_CLOUD_PATH,
        output_dir=OUTPUT_DIRECTORY,
        run_tag=RUN_TAG,
        z_handling=Z_AXIS_HANDLING_MODE,
        crop_on_height=ENABLE_HEIGHT_CROP,
        crop_z_min=HEIGHT_CROP_MIN_ABOVE_GROUND_M,
        crop_z_max=HEIGHT_CROP_MAX_ABOVE_GROUND_M,
        ground_grid_size=GROUND_MODEL_GRID_CELL_SIZE_M,
        ground_smoothing_size=GROUND_MODEL_MEDIAN_FILTER_SIZE,
        change_t1=CHANGE_THRESHOLD_T1_M,
        change_t2=CHANGE_THRESHOLD_T2_M,
        change_grid_size=CHANGE_STATISTICS_GRID_CELL_SIZE_M,
        dsm_cell_size=DSM_GRID_CELL_SIZE_M,
        signed_color_likely_removed=COLOR_LIKELY_REMOVED,
        signed_color_possible_removed=COLOR_POSSIBLY_REMOVED,
        signed_color_unchanged=COLOR_UNCHANGED,
        signed_color_possible_added=COLOR_POSSIBLY_ADDED,
        signed_color_likely_added=COLOR_LIKELY_ADDED,
    )

    for input_file_path in (config.reference, config.compared):
        if not os.path.exists(input_file_path):
            sys.exit(f"Datei nicht gefunden: {input_file_path}")

    if config.change_t1 >= config.change_t2:
        sys.exit(
            f"T1 ({config.change_t1}) muss kleiner als T2 ({config.change_t2}) sein."
        )

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  C2C VERGLEICH v8  +  DSM-FIRST  (ein Raster-Pass)     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    output_file_prefix = build_output_file_prefix(config.run_tag)
    print(f"\n  Run:     {output_file_prefix}")
    print(f"  Ausgabe: {config.output_dir}")
    print(f"  T1 / T2: {config.change_t1:.3f} m / {config.change_t2:.3f} m")
    print(f"  DSM:     {config.dsm_cell_size:.3f} m")

    os.makedirs(config.output_dir, exist_ok=True)

    # Phase 1: Punktwolken laden
    print("\n━━━ Phase 1: Punktwolken laden ━━━")
    reference_point_cloud = load_point_cloud_from_ply(config.reference)
    comparison_point_cloud = load_point_cloud_from_ply(config.compared)

    # Phase 2: DSM aufbauen und Z-Achse prüfen
    print("\n━━━ Phase 2: DSM aufbauen + Z-Achse prüfen ━━━")
    reference_points_xyz_raw = np.asarray(reference_point_cloud.points)
    comparison_points_xyz_raw = np.asarray(comparison_point_cloud.points)

    if config.z_handling == "flip":
        should_flip_z_axis = True
        print("  Z-Handling: flip  (manuell)")
    elif config.z_handling == "keep":
        should_flip_z_axis = False
        print("  Z-Handling: keep  (manuell)")
    else:
        print("  Z-Handling: auto  →  DSM-Rauheit-Methode")
        raw_dsm_raster = build_reference_dsm_min_max_raster(
            reference_points_xyz_raw,
            comparison_points_xyz_raw,
            config.dsm_cell_size,
        )
        should_flip_z_axis = detect_whether_z_axis_is_flipped_from_dsm(raw_dsm_raster)

    if config.z_handling == "auto":
        working_reference_surface_dsm = derive_working_dsm_surface_from_raw_dsm(
            raw_dsm_raster,
            should_flip_z_axis,
        )
        print(
            "  Working-DSM aus bestehendem Raster abgeleitet  "
            f"({'−dsm_min' if should_flip_z_axis else 'dsm_max'})"
        )
    else:
        raw_dsm_raster = build_reference_dsm_min_max_raster(
            reference_points_xyz_raw,
            comparison_points_xyz_raw,
            config.dsm_cell_size,
        )
        working_reference_surface_dsm = derive_working_dsm_surface_from_raw_dsm(
            raw_dsm_raster,
            should_flip_z_axis,
        )
        print(
            "  Working-DSM frisch aufgebaut  "
            f"({'−dsm_min' if should_flip_z_axis else 'dsm_max'})"
        )

    # Phase 3: Z-Flip anwenden
    reference_point_cloud = apply_optional_z_axis_flip(reference_point_cloud, should_flip_z_axis)
    comparison_point_cloud = apply_optional_z_axis_flip(comparison_point_cloud, should_flip_z_axis)
    print(f"  Z {'invertiert' if should_flip_z_axis else 'unverändert'}")

    # Phase 4: Optionaler Crop oder volle Wolke
    if config.crop_on_height:
        print("\n━━━ Phase 3: Lokaler Boden-Crop ━━━")

        local_ground_model = build_local_ground_model_from_reference(
            reference_point_cloud,
            config.ground_grid_size,
            config.ground_smoothing_size,
        )

        cropped_reference_point_cloud = crop_point_cloud_by_height_above_local_ground(
            reference_point_cloud,
            local_ground_model,
            config.crop_z_min,
            config.crop_z_max,
            "ref",
        )
        cropped_comparison_point_cloud = crop_point_cloud_by_height_above_local_ground(
            comparison_point_cloud,
            local_ground_model,
            config.crop_z_min,
            config.crop_z_max,
            "cmp",
        )

        print("  Rebuild Working-DSM für gecropte Wolke...")
        cropped_working_reference_surface_dsm = rebuild_working_dsm_surface_from_reference_cloud(
            cropped_reference_point_cloud,
            np.asarray(cropped_comparison_point_cloud.points),
            config.dsm_cell_size,
        )

        comparison_point_cloud_for_export = restore_original_z_axis_orientation(
            cropped_comparison_point_cloud,
            should_flip_z_axis,
        )

        results = run_c2c_change_detection_pipeline(
            cropped_reference_point_cloud,
            cropped_comparison_point_cloud,
            config,
            output_file_prefix,
            working_reference_surface_dsm=cropped_working_reference_surface_dsm,
            processing_label="crop",
            export_point_cloud_with_original_z=comparison_point_cloud_for_export,
        )
    else:
        print("\n━━━ Phase 3: C2C auf voller Wolke ━━━")

        comparison_point_cloud_for_export = restore_original_z_axis_orientation(
            comparison_point_cloud,
            should_flip_z_axis,
        )

        results = run_c2c_change_detection_pipeline(
            reference_point_cloud,
            comparison_point_cloud,
            config,
            output_file_prefix,
            working_reference_surface_dsm=working_reference_surface_dsm,
            processing_label="full",
            export_point_cloud_with_original_z=comparison_point_cloud_for_export,
        )

    (
        _,
        _,
        _,
        absolute_c2c_distances_m,
        signed_change_indicator_m,
        _,
        signed_change_classes,
    ) = results

    absolute_distances_m = np.abs(absolute_c2c_distances_m)
    number_of_points = len(absolute_c2c_distances_m)

    number_of_unchanged_points = int(np.sum(signed_change_classes == 0))
    number_of_possibly_added_points = int(np.sum(signed_change_classes == 1))
    number_of_likely_added_points = int(np.sum(signed_change_classes == 2))
    number_of_possibly_removed_points = int(np.sum(signed_change_classes == -1))
    number_of_likely_removed_points = int(np.sum(signed_change_classes == -2))

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ERGEBNIS                                               ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Punkte   : {number_of_points:>12,}                            ║")
    print(f"║  Ø |d|    : {np.mean(absolute_distances_m):>12.6f} m                        ║")
    print(f"║  Ø signed : {np.mean(signed_change_indicator_m):>+12.6f} m  [DSM]                ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  T1={config.change_t1:.3f}m  T2={config.change_t2:.3f}m                            ║")
    print(f"║  ▲ likely  added  : {number_of_likely_added_points:>10,}  ({100.0 * number_of_likely_added_points / number_of_points:5.1f}%)          ║")
    print(f"║  ▲ possibly added : {number_of_possibly_added_points:>10,}  ({100.0 * number_of_possibly_added_points / number_of_points:5.1f}%)          ║")
    print(f"║  ─ unchanged      : {number_of_unchanged_points:>10,}  ({100.0 * number_of_unchanged_points / number_of_points:5.1f}%)          ║")
    print(f"║  ▼ possibly remov.: {number_of_possibly_removed_points:>10,}  ({100.0 * number_of_possibly_removed_points / number_of_points:5.1f}%)          ║")
    print(f"║  ▼ likely  remov. : {number_of_likely_removed_points:>10,}  ({100.0 * number_of_likely_removed_points / number_of_points:5.1f}%)          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()