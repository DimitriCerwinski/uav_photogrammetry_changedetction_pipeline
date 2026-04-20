#!/usr/bin/env python3
"""
TEASER++ + ICP Registrierung
Ausführlich kommentierte und lesbar umgeschriebene Version
==========================================================

Ziel dieses Skripts
-------------------
Dieses Skript registriert zwei Punktwolken miteinander:

- eine Quellpunktwolke (SRC = source)
- eine Zielpunktwolke   (TGT = target)

Die Registrierung läuft in mehreren Schritten ab:

1. Windows startet das Skript.
2. Das Skript schreibt einen Python-Runner nach WSL.
3. In WSL werden die Punktwolken geladen und optional skaliert.
4. Danach werden beide Punktwolken gefiltert.
5. Aus FPFH-Features werden Korrespondenzen erzeugt.
6. TEASER++ berechnet eine robuste Grobregistrierung.
7. Optional wird diese Grobregistrierung mit ICP verfeinert.
8. Anschließend werden Kennzahlen, Diagramme, JSON und PLY-Dateien exportiert.

Warum dieses Skript so aufgebaut ist
------------------------------------
TEASER++ und Open3D laufen in deinem Setup unter WSL/Linux stabiler.
Darum dient der Windows-Teil nur als Starter, während die eigentliche
Punktwolkenverarbeitung in einem WSL-Runner stattfindet.

Zusätzliche Auswertungsexporte
------------------------------
Diese Version exportiert zusätzlich:

- eine kompakte CSV mit Kernmetriken
- ein Diagramm "TEASER vs. FINAL"
- eine CDF der Restabstände
- eine JSON-Datei mit allen wichtigen Laufdetails
- die final registrierten PLY-Dateien

Finale Kernmetriken
-------------------
Die CSV enthält bewusst nur wenige, aber wichtige Größen:

- final_stage
- n_corrs
- final_fitness
- final_trim30
- p95_final
- final_rmse
- S_total
- runtime_s

Hinweis zur Logik "final"
--------------------------
"final" meint immer das Ergebnis, das am Ende tatsächlich verwendet wird:

- wenn ICP akzeptiert wird:      TEASER + ICP
- wenn ICP abgelehnt wird:       nur TEASER
- wenn ICP übersprungen wird:    nur TEASER

Damit sind die exportierten Kennwerte immer definiert und nie unnötig "N/A".
"""

import datetime
import json
import os
import pprint
import re
import subprocess
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# Name der WSL-Distribution.
WSL_DISTRIBUTION_NAME = "Ubuntu-22.04"

# Arbeitsordner in WSL. Dort wird der temporäre Runner abgelegt.
WSL_WORKING_DIRECTORY = "/home/dc/photogrammetry/pcd_registration"

# Pfad zur virtuellen Python-Umgebung in WSL.
WSL_VENV_ACTIVATE = f"{WSL_WORKING_DIRECTORY}/.venv/bin/activate"

# ── Eingabedaten ──────────────────────────────────────────────────────
# Quellpunktwolke (wird auf Zielpunktwolke registriert)
WINDOWS_SOURCE_POINT_CLOUD = (
    r"C:\Users\Admin\Desktop\Photogrammetrie\pcd_registration\Flight_No6A.ply"
)

# Zielpunktwolke (Referenz / Master)
WINDOWS_TARGET_POINT_CLOUD = (
    r"C:\Users\Admin\Desktop\Photogrammetrie\pcd_registration\Flight_No6B.ply"
)

# Exportordner für JSON, CSV, Diagramme und PLY-Dateien.
WINDOWS_EXPORT_DIRECTORY = (
    r"D:\Users\Admin\Desktop\Photogrammetrie\pcd_registration_exports"
)

# Frei wählbarer Runtag für Dateinamen.
RUN_TAG = "f6a-on-f6b (uncroped)"

# ── Master-Modus ──────────────────────────────────────────────────────
# True:
#   TGT gilt als Master und wird NICHT skaliert.
#   Das ist für Zeitreihen praktisch, wenn alle weiteren Datensätze auf
#   dieselbe Referenz registriert werden sollen.
# False:
#   klassische paarweise Registrierung: SRC und TGT können beide skaliert
#   werden.
USE_TARGET_AS_MASTER = False

# ── Referenzmaß der Zielpunktwolke ────────────────────────────────────
# Wird ignoriert, wenn USE_TARGET_AS_MASTER = True.
TARGET_REFERENCE_REAL_LENGTH_METERS = 1.0
TARGET_REFERENCE_LENGTH_IN_CLOUD_UNITS = 0.38018

# ── Referenzmaß der Quellpunktwolke ───────────────────────────────────
# Diese Werte werden benutzt, um die Quellpunktwolke in Meter umzurechnen.
# Wenn die Schätzung nicht perfekt ist, kann Scale-Retry versuchen,
# den Quellmaßstab nachträglich zu verbessern.
SOURCE_REFERENCE_REAL_LENGTH_METERS = 1.0
SOURCE_REFERENCE_LENGTH_IN_CLOUD_UNITS = 0.38594

# ── Hauptparameter ────────────────────────────────────────────────────
# Voxelgröße für Downsampling, Featurebildung, Metriken und ICP.
VOXEL_SIZE_METERS = 0.2

# TEASER++ Noise Bound in Metern.
TEASER_NOISE_BOUND_METERS = 0.07

# Maximale Anzahl verwendeter Korrespondenzen.
MAXIMUM_NUMBER_OF_CORRESPONDENCES = 5000

# Ob TEASER++ Skalierung mit schätzen darf.
ENABLE_SCALE_ESTIMATION = True

# Ob am Ende PLY-Dateien des finalen Ergebnisses exportiert werden.
EXPORT_FINAL_ALIGNED_POINT_CLOUDS = True

# ── Scale-Retry ───────────────────────────────────────────────────────
# Wenn aktiviert und ICP nicht akzeptiert wird, wird der geschätzte
# Gesamtskalierungsfaktor benutzt, um den Maßstab von SRC zu korrigieren,
# und der gesamte Lauf wird wiederholt.
ENABLE_SCALE_RETRY = False
MAXIMUM_SCALE_RETRY_COUNT = 2

# ── Vorverarbeitung / Filter ──────────────────────────────────────────
# Z-Clip entfernt extrem tiefe und extrem hohe Punkte per Quantil.
ENABLE_Z_CLIP = True
Z_CLIP_LOWER_QUANTILE = 0.01
Z_CLIP_UPPER_QUANTILE = 0.995

# Statistischer Ausreißerfilter.
STATISTICAL_OUTLIER_NB_NEIGHBORS = 30
STATISTICAL_OUTLIER_STD_RATIO = 2.0

# ── Korrespondenzbildung ──────────────────────────────────────────────
MINIMUM_REQUIRED_CORRESPONDENCES = 100
CORRESPONDENCE_GRID_CELL_SIZE_MULTIPLIER = 6.0
MAXIMUM_CORRESPONDENCES_PER_GRID_CELL = 6

# ── Normalen und FPFH ─────────────────────────────────────────────────
NORMAL_SEARCH_RADIUS_MULTIPLIER = 2.0
NORMAL_SEARCH_MAX_NEIGHBORS = 30
FPFH_SEARCH_RADIUS_MULTIPLIER = 5.0
FPFH_SEARCH_MAX_NEIGHBORS = 100

# ── ICP-Entscheidungsparameter ────────────────────────────────────────
# ICP wird nur versucht, wenn die TEASER-Lösung bereits genügend Überlappung
# zeigt. Ansonsten würde ICP meist nur Zeit kosten oder schlechte Ergebnisse
# produzieren.
MINIMUM_FITNESS_FOR_ICP = 0.35

# Distanzgrenze für die Fitness in align_metrics().
ALIGNMENT_MAX_DISTANCE_MULTIPLIER = 2.0

# ICP-internes Gating und Suchdistanzen.
ICP_GATE_DISTANCE_MULTIPLIER = 2.0
ICP_POINT_TO_POINT_MAX_DISTANCE_MULTIPLIER = 2.0
ICP_POINT_TO_PLANE_MAX_DISTANCE_MULTIPLIER = 1.0
ICP_POINT_TO_POINT_MAX_ITERATIONS = 40
ICP_POINT_TO_PLANE_MAX_ITERATIONS = 50

# ICP wird nur akzeptiert, wenn der gute Kernbereich nicht schlechter wird.
ICP_ACCEPT_TRIM30_FACTOR = 1.05

# ── TEASER++ Parameter ────────────────────────────────────────────────
TEASER_CBAR2 = 1.0
TEASER_ROTATION_GNC_FACTOR = 1.4
TEASER_ROTATION_MAX_ITERATIONS = 100
TEASER_ROTATION_COST_THRESHOLD = 1e-12


# ══════════════════════════════════════════════════════════════════════
# ALLGEMEINE HILFSFUNKTIONEN (Windows-Seite)
# ══════════════════════════════════════════════════════════════════════

def convert_windows_path_to_wsl_path(path_string: str) -> str:
    """
    Wandelt einen Windows-Pfad in einen WSL-kompatiblen Pfad um.

    Beispiel:
        C:/Users/Admin/file.ply
    wird zu:
        /mnt/c/Users/Admin/file.ply
    """
    cleaned_path = path_string.strip()

    # Falls bereits ein Linux-/WSL-Pfad übergeben wurde, direkt zurückgeben.
    if cleaned_path.startswith("/"):
        return cleaned_path

    # Spezialfall: \\wsl$\...
    match = re.match(r"^\\\\wsl\$\\[^\\]+\\(.*)$", cleaned_path, re.IGNORECASE)
    if match:
        return "/" + match.group(1).replace("\\", "/").lstrip("/")

    absolute_path = os.path.abspath(cleaned_path).replace("\\", "/")

    # Standardfall: Laufwerksbuchstabe vorhanden.
    if len(absolute_path) >= 2 and absolute_path[1] == ":":
        drive_letter = absolute_path[0].lower()
        remaining_path = absolute_path[2:].lstrip("/")
        return f"/mnt/{drive_letter}/{remaining_path}"

    return absolute_path


def make_filename_safe(text: str) -> str:
    """Erzeugt aus einem freien Tag einen sicheren Dateinamenbestandteil."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40]


# ══════════════════════════════════════════════════════════════════════
# WSL-RUNNER-CODE
# ══════════════════════════════════════════════════════════════════════
#
# Dieser Code lebt im selben Skript, wird aber nur im WSL-Runner-Modus mit
# den Linux-/WSL-Abhängigkeiten initialisiert.
#
import copy
import csv
import time

np = None
o3d = None
teaserpp_python = None
plt = None
MATPLOTLIB_AVAILABLE = False
CONFIG = {}


def initialize_wsl_dependencies():
    """
    Lädt WSL-/Linux-Abhängigkeiten erst zur Laufzeit im Runner-Modus.

    Wichtig:
    Der Windows-Starter darf diese Module NICHT beim Start importieren,
    weil sie in der Regel nur in WSL installiert sind.
    """
    global np, o3d, teaserpp_python, plt, MATPLOTLIB_AVAILABLE

    import numpy as _np
    import open3d as _o3d
    import teaserpp_python as _teaserpp_python

    np = _np
    o3d = _o3d
    teaserpp_python = _teaserpp_python

    try:
        import matplotlib as _matplotlib
        _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
        MATPLOTLIB_AVAILABLE = True
    except Exception:
        MATPLOTLIB_AVAILABLE = False
        print("[WARN] Matplotlib nicht verfügbar – Diagramme werden übersprungen.", flush=True)


# ══════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN: KONFIGURATION UND BASISOPERATIONEN
# ══════════════════════════════════════════════════════════════════════

def get_config_value(key, default=None):
    """Kurzer Zugriff auf Konfigurationswerte."""
    return CONFIG.get(key, default)


def scale_point_cloud_to_metric_units(point_cloud, scale_factor):
    """
    Multipliziert alle Punktkoordinaten mit einem Faktor.

    Hintergrund:
    In photogrammetrischen Punktwolken liegt der Maßstab häufig nicht exakt
    in Metern vor. Mit dem Faktor 1 / one_meter_* wird die Wolke so skaliert,
    dass 1 reale Meter möglichst korrekt abgebildet werden.
    """
    points = np.asarray(point_cloud.points) * scale_factor
    point_cloud.points = o3d.utility.Vector3dVector(points)


def remove_non_finite_points(point_cloud):
    """
    Entfernt NaN- und Inf-Punkte.

    Solche Punkte können Feature-Bildung, Normalenschätzung und ICP stören.
    """
    point_cloud_copy = copy.deepcopy(point_cloud)
    result = point_cloud_copy.remove_non_finite_points()
    return result[0] if isinstance(result, tuple) else result


def apply_z_quantile_clip(point_cloud, lower_quantile=0.01, upper_quantile=0.995):
    """
    Entfernt Punkte anhand der Z-Verteilung.

    Das ist praktisch, wenn ganz unten oder ganz oben wenige extreme Punkte
    liegen, die für die Registrierung nur stören.
    """
    z_values = np.asarray(point_cloud.points)[:, 2]
    z_lower, z_upper = np.quantile(z_values, [lower_quantile, upper_quantile])
    valid_indices = np.where((z_values >= z_lower) & (z_values <= z_upper))[0]
    return point_cloud.select_by_index(valid_indices)


def apply_statistical_outlier_filter(point_cloud, nb_neighbors=30, std_ratio=2.0):
    """
    Entfernt statistische Ausreißer.

    Open3D vergleicht hier lokale Punktabstände und verwirft Punkte,
    die deutlich aus der Nachbarschaft herausfallen.
    """
    result = point_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return result[0] if isinstance(result, tuple) else result


def transfer_colors_from_original_to_filtered(original_point_cloud, filtered_point_cloud):
    """
    Überträgt Farben von der Originalwolke auf die gefilterte Wolke.

    Nach Filterschritten können Farben verloren gehen, wenn nur Punkte
    selektiert werden. Für Visualisierung und PLY-Export ist es nützlich,
    die Farben zu erhalten.
    """
    if not original_point_cloud.has_colors():
        return filtered_point_cloud

    search_tree = o3d.geometry.KDTreeFlann(original_point_cloud)
    original_colors = np.asarray(original_point_cloud.colors)
    new_colors = []

    for point in np.asarray(filtered_point_cloud.points):
        _, nearest_index, _ = search_tree.search_knn_vector_3d(point, 1)
        new_colors.append(original_colors[nearest_index[0]])

    filtered_point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(new_colors))
    return filtered_point_cloud


def filter_point_cloud_with_optional_color_transfer(point_cloud):
    """
    Führt alle Vorverarbeitungsschritte auf einer Punktwolke aus.

    Reihenfolge:
    1. optionaler Z-Clip
    2. statistischer Ausreißerfilter
    3. Farbübertragung
    """
    filtered_point_cloud = point_cloud

    if get_config_value("enable_z_clip", True):
        filtered_point_cloud = apply_z_quantile_clip(
            filtered_point_cloud,
            float(get_config_value("z_clip_lower_quantile", 0.01)),
            float(get_config_value("z_clip_upper_quantile", 0.995)),
        )

    filtered_point_cloud = apply_statistical_outlier_filter(
        filtered_point_cloud,
        int(get_config_value("statistical_outlier_nb_neighbors", 30)),
        float(get_config_value("statistical_outlier_std_ratio", 2.0)),
    )

    if point_cloud.has_colors():
        filtered_point_cloud = transfer_colors_from_original_to_filtered(
            point_cloud,
            filtered_point_cloud,
        )

    return filtered_point_cloud


def print_point_cloud_info(label, point_cloud):
    """
    Gibt eine kurze Übersicht über die Punktwolke in der Konsole aus.

    Besonders nützlich, um direkt zu sehen:
    - wie viele Punkte vorhanden sind
    - wie groß die Punktwolke grob ist
    - ob Farben vorhanden sind
    """
    if point_cloud.is_empty():
        print(f"[{label}] EMPTY!", flush=True)
        return

    extent = point_cloud.get_axis_aligned_bounding_box().get_extent()
    print(
        f"[{label}] pts={len(point_cloud.points):,}  "
        f"extent={extent[0]:.2f}x{extent[1]:.2f}x{extent[2]:.2f} m  "
        f"colors={point_cloud.has_colors()}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════
# FEATURE-BILDUNG UND KORRESPONDENZEN
# ══════════════════════════════════════════════════════════════════════

def preprocess_point_cloud_for_fpfh(point_cloud, voxel_size):
    """
    Downsample + Normalen + FPFH.

    Warum diese Reihenfolge?
    - Downsampling reduziert Rechenzeit und Rauschen.
    - FPFH benötigt Normalen.
    - FPFH liefert lokale Deskriptoren für robustes Matching.
    """
    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)

    normal_radius = float(get_config_value("normal_search_radius_multiplier", 2.0)) * voxel_size
    normal_max_neighbors = int(get_config_value("normal_search_max_neighbors", 30))
    downsampled_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_neighbors,
        )
    )

    fpfh_radius = float(get_config_value("fpfh_search_radius_multiplier", 5.0)) * voxel_size
    fpfh_max_neighbors = int(get_config_value("fpfh_search_max_neighbors", 100))
    fpfh_feature = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius,
            max_nn=fpfh_max_neighbors,
        ),
    )

    return downsampled_cloud, fpfh_feature


def build_teaser_correspondences(source_cloud, target_cloud, voxel_size, max_correspondences):
    """
    Erzeugt Korrespondenzen zwischen SRC und TGT auf Basis von FPFH.

    Ablauf:
    1. Beide Wolken werden für FPFH vorbereitet.
    2. Open3D erzeugt Merkmalskorrespondenzen.
    3. Diese werden nach Merkmalsabstand sortiert.
    4. Ein Gitter-basiertes Sampling verhindert, dass zu viele Korrespondenzen
       aus demselben lokalen Bereich stammen.
    5. Ergebnis sind 3xN Punktmatrizen für TEASER++.
    """
    source_downsampled, source_fpfh = preprocess_point_cloud_for_fpfh(source_cloud, voxel_size)
    target_downsampled, target_fpfh = preprocess_point_cloud_for_fpfh(target_cloud, voxel_size)

    correspondences = np.asarray(
        o3d.pipelines.registration.correspondences_from_features(
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
        )
    )

    if correspondences.shape[0] < int(get_config_value("minimum_required_correspondences", 100)):
        return None, None, None, 0

    source_feature_array = np.asarray(source_fpfh.data).T
    target_feature_array = np.asarray(target_fpfh.data).T
    feature_distances = np.linalg.norm(
        source_feature_array[correspondences[:, 0]] - target_feature_array[correspondences[:, 1]],
        axis=1,
    )

    correspondences = correspondences[np.argsort(feature_distances)]

    source_points = np.asarray(source_downsampled.points)
    grid_cell_size = float(get_config_value("correspondence_grid_cell_size_multiplier", 6.0)) * voxel_size
    maximum_per_cell = int(get_config_value("maximum_correspondences_per_grid_cell", 6))

    selected_indices = []
    used_cells = {}

    for correspondence_index, (source_index, _) in enumerate(correspondences):
        cell_key = tuple(np.floor(source_points[source_index] / grid_cell_size).astype(int))
        if used_cells.get(cell_key, 0) < maximum_per_cell:
            used_cells[cell_key] = used_cells.get(cell_key, 0) + 1
            selected_indices.append(correspondence_index)
            if len(selected_indices) >= max_correspondences:
                break

    correspondences = correspondences[selected_indices]

    if correspondences.shape[0] < int(get_config_value("minimum_required_correspondences", 100)):
        return None, None, None, 0

    source_points_for_teaser = np.asarray(source_downsampled.points)[correspondences[:, 0]].T
    target_points_for_teaser = np.asarray(target_downsampled.points)[correspondences[:, 1]].T

    return (
        source_points_for_teaser,
        target_points_for_teaser,
        (source_downsampled, target_downsampled),
        correspondences.shape[0],
    )


# ══════════════════════════════════════════════════════════════════════
# TEASER++ UND ICP
# ══════════════════════════════════════════════════════════════════════

def solve_registration_with_teaser(source_points_3xN, target_points_3xN, noise_bound, estimate_scaling=True):
    """
    Führt die robuste Grobregistrierung mit TEASER++ aus.

    Ergebnis:
    - vollständige 4x4-Transformationsmatrix
    - von TEASER geschätzter Skalierungsfaktor
    """
    solver_parameters = teaserpp_python.RobustRegistrationSolver.Params()
    solver_parameters.cbar2 = float(get_config_value("teaser_cbar2", 1.0))
    solver_parameters.noise_bound = float(noise_bound)
    solver_parameters.estimate_scaling = bool(estimate_scaling)
    solver_parameters.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_parameters.rotation_gnc_factor = float(get_config_value("teaser_rotation_gnc_factor", 1.4))
    solver_parameters.rotation_max_iterations = int(get_config_value("teaser_rotation_max_iterations", 100))
    solver_parameters.rotation_cost_threshold = float(get_config_value("teaser_rotation_cost_threshold", 1e-12))

    solver = teaserpp_python.RobustRegistrationSolver(solver_parameters)
    solver.solve(source_points_3xN, target_points_3xN)

    solution = solver.getSolution()
    estimated_scale = float(solution.scale)
    estimated_rotation = np.array(solution.rotation)
    estimated_translation = np.array(solution.translation).reshape(3)

    transformation = np.eye(4)
    transformation[:3, :3] = estimated_scale * estimated_rotation
    transformation[:3, 3] = estimated_translation

    return transformation, estimated_scale


def refine_registration_with_icp(source_aligned_after_teaser, target_cloud, voxel_size):
    """
    Führt eine zweistufige ICP-Verfeinerung aus:

    1. Point-to-Point mit Skalierung
    2. Point-to-Plane rigid

    Warum zweistufig?
    - Die erste Stufe korrigiert verbleibende Grobabweichungen.
    - Die zweite Stufe verfeinert die Lage an Oberflächen genauer.
    """
    voxel = float(voxel_size)

    gate_distance_multiplier = float(get_config_value("icp_gate_distance_multiplier", 2.0))
    point_to_point_max_distance_multiplier = float(get_config_value("icp_point_to_point_max_distance_multiplier", 2.0))
    point_to_plane_max_distance_multiplier = float(get_config_value("icp_point_to_plane_max_distance_multiplier", 1.0))
    point_to_point_max_iterations = int(get_config_value("icp_point_to_point_max_iterations", 40))
    point_to_plane_max_iterations = int(get_config_value("icp_point_to_plane_max_iterations", 50))

    source_downsampled = copy.deepcopy(source_aligned_after_teaser).voxel_down_sample(voxel)
    target_downsampled = copy.deepcopy(target_cloud).voxel_down_sample(voxel)

    # Gating:
    # Nur Punkte in der Nähe der Zielwolke bleiben erhalten.
    # Das verhindert, dass weit entfernte Bereiche ICP verschlechtern.
    target_tree = o3d.geometry.KDTreeFlann(target_downsampled)
    squared_gate_distance = (gate_distance_multiplier * voxel) ** 2
    kept_indices = [
        index
        for index, point in enumerate(np.asarray(source_downsampled.points))
        if target_tree.search_knn_vector_3d(point, 1)[2][0] <= squared_gate_distance
    ]
    source_downsampled = source_downsampled.select_by_index(kept_indices)
    print(
        f"[ICP] gated src: {len(source_downsampled.points):,} pts "
        f"(gate={gate_distance_multiplier * voxel:.2f} m)",
        flush=True,
    )

    # ICP Stufe 1: Point-to-Point mit Skalierung.
    icp_stage_1 = o3d.pipelines.registration.registration_icp(
        source_downsampled,
        target_downsampled,
        point_to_point_max_distance_multiplier * voxel,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=point_to_point_max_iterations
        ),
    )
    print(
        f"[ICP-Scale]  fit={icp_stage_1.fitness:.4f}  rmse={icp_stage_1.inlier_rmse:.4f}",
        flush=True,
    )

    source_after_stage_1 = copy.deepcopy(source_downsampled)
    source_after_stage_1.transform(icp_stage_1.transformation)

    # Für Point-to-Plane werden Normalen benötigt.
    for point_cloud in [source_after_stage_1, target_downsampled]:
        point_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(get_config_value("normal_search_radius_multiplier", 2.0)) * voxel,
                max_nn=int(get_config_value("normal_search_max_neighbors", 30)),
            )
        )

    try:
        point_to_plane_estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=voxel)
        )
    except Exception:
        point_to_plane_estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    # ICP Stufe 2: Point-to-Plane, rigid.
    icp_stage_2 = o3d.pipelines.registration.registration_icp(
        source_after_stage_1,
        target_downsampled,
        point_to_plane_max_distance_multiplier * voxel,
        np.eye(4),
        point_to_plane_estimator,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=point_to_plane_max_iterations
        ),
    )
    print(
        f"[ICP-Rigid]  fit={icp_stage_2.fitness:.4f}  rmse={icp_stage_2.inlier_rmse:.4f}",
        flush=True,
    )

    total_icp_transformation = icp_stage_2.transformation @ icp_stage_1.transformation

    class ICPResult:
        transformation = total_icp_transformation
        fitness = icp_stage_2.fitness
        inlier_rmse = icp_stage_2.inlier_rmse

    return ICPResult()


# ══════════════════════════════════════════════════════════════════════
# METRIKEN UND DIAGRAMME
# ══════════════════════════════════════════════════════════════════════

def compute_alignment_metrics(source_aligned, target_cloud, voxel_size):
    """
    Berechnet robuste Distanzkennzahlen für eine ausgerichtete SRC gegen TGT.

    Wichtige Ausgaben:
    - trim30: guter Kernbereich
    - median
    - p95: oberer Fehlerbereich / Tail
    - fitness: Anteil Punkte innerhalb eines Distanz-Gates
    """
    source_downsampled = source_aligned.voxel_down_sample(voxel_size)
    target_downsampled = target_cloud.voxel_down_sample(voxel_size)

    target_tree = o3d.geometry.KDTreeFlann(target_downsampled)
    max_distance = float(get_config_value("alignment_max_distance_multiplier", 2.0)) * voxel_size

    nearest_neighbor_distances = []
    for point in np.asarray(source_downsampled.points):
        number_of_neighbors, _, squared_distances = target_tree.search_knn_vector_3d(point, 1)
        if number_of_neighbors:
            nearest_neighbor_distances.append(np.sqrt(squared_distances[0]))

    distances = np.asarray(nearest_neighbor_distances)
    if distances.size == 0:
        return {}

    sorted_distances = np.sort(distances)
    number_of_core_distances = max(1, int(0.3 * len(sorted_distances)))

    return {
        "trim30": float(np.mean(sorted_distances[:number_of_core_distances])),
        "median": float(np.median(distances)),
        "p95": float(np.quantile(distances, 0.95)),
        "fitness": float(np.mean(distances <= max_distance)),
    }


def compute_residual_distances(source_aligned, target_cloud, voxel_size):
    """
    Berechnet alle NN-Restabstände zwischen SRC und TGT.

    Diese Verteilung wird später für:
    - die finale RMSE
n    - die CDF-Grafik
    verwendet.
    """
    source_downsampled = source_aligned.voxel_down_sample(voxel_size)
    target_downsampled = target_cloud.voxel_down_sample(voxel_size)

    if source_downsampled.is_empty() or target_downsampled.is_empty():
        return np.asarray([], dtype=float)

    target_tree = o3d.geometry.KDTreeFlann(target_downsampled)
    nearest_neighbor_distances = []

    for point in np.asarray(source_downsampled.points):
        number_of_neighbors, _, squared_distances = target_tree.search_knn_vector_3d(point, 1)
        if number_of_neighbors:
            nearest_neighbor_distances.append(np.sqrt(squared_distances[0]))

    return np.asarray(nearest_neighbor_distances, dtype=float)


def compute_rmse_from_distances(distances):
    """Berechnet die RMSE aus einer Distanzverteilung."""
    if distances.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(distances ** 2)))


def save_core_metrics_csv(output_path, row_dictionary):
    """Speichert die kompakten Kernmetriken als 1-Zeilen-CSV."""
    with open(output_path, "w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(row_dictionary.keys()), delimiter=";")
        writer.writeheader()
        writer.writerow(row_dictionary)
    print(f"[OUT] {output_path}", flush=True)


def save_stage_comparison_plot(output_path, teaser_metrics, final_metrics):
    """
    Erstellt ein Vergleichsdiagramm zwischen:
    - TEASER-Ergebnis
    - finalem Ergebnis

    Links:
        fitness
    Rechts:
        trim30 und p95
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, (axis_fitness, axis_errors) = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Fitness-Vergleich ---
    fitness_values = [
        float(teaser_metrics.get("fitness", np.nan)),
        float(final_metrics.get("fitness", np.nan)),
    ]
    fitness_bars = axis_fitness.bar(["TEASER", "FINAL"], fitness_values, edgecolor="black")

    for bar, value in zip(fitness_bars, fitness_values):
        if np.isfinite(value):
            axis_fitness.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis_fitness.set_title("Fitness")
    axis_fitness.set_ylabel("Anteil Inlier / Überlappung")
    if np.any(np.isfinite(fitness_values)):
        axis_fitness.set_ylim(0, max(1.0, np.nanmax(fitness_values) * 1.15))
    else:
        axis_fitness.set_ylim(0, 1.0)

    # --- Fehlermaße ---
    labels = ["trim30", "p95"]
    x_positions = np.arange(len(labels))
    bar_width = 0.36

    teaser_error_values = [
        float(teaser_metrics.get("trim30", np.nan)),
        float(teaser_metrics.get("p95", np.nan)),
    ]
    final_error_values = [
        float(final_metrics.get("trim30", np.nan)),
        float(final_metrics.get("p95", np.nan)),
    ]

    teaser_bars = axis_errors.bar(
        x_positions - bar_width / 2,
        teaser_error_values,
        bar_width,
        label="TEASER",
        edgecolor="black",
    )
    final_bars = axis_errors.bar(
        x_positions + bar_width / 2,
        final_error_values,
        bar_width,
        label="FINAL",
        edgecolor="black",
    )

    for bar_group in [teaser_bars, final_bars]:
        for bar in bar_group:
            height = bar.get_height()
            if np.isfinite(height):
                axis_errors.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axis_errors.set_xticks(x_positions)
    axis_errors.set_xticklabels(labels)
    axis_errors.set_ylabel("Restabstand [m]")
    axis_errors.set_title("Fehlermetriken")
    axis_errors.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OUT] {output_path}", flush=True)


def save_residual_cdf_plot(output_path, teaser_residuals, final_residuals):
    """
    Erstellt eine CDF (kumulative Verteilung) der Restabstände.

    Interpretation:
    Je weiter links/oben eine Kurve liegt, desto besser.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, axis = plt.subplots(figsize=(8, 5))

    if teaser_residuals.size > 0:
        sorted_teaser = np.sort(teaser_residuals)
        teaser_cdf = np.arange(1, len(sorted_teaser) + 1) / len(sorted_teaser)
        axis.plot(sorted_teaser, teaser_cdf, label="TEASER")

    if final_residuals.size > 0:
        sorted_final = np.sort(final_residuals)
        final_cdf = np.arange(1, len(sorted_final) + 1) / len(sorted_final)
        axis.plot(sorted_final, final_cdf, label="FINAL")

    axis.set_xlabel("Restabstand [m]")
    axis.set_ylabel("Kumulative Häufigkeit")
    axis.set_title("CDF der Restabstände")
    axis.set_ylim(0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OUT] {output_path}", flush=True)


def export_registration_metrics_and_plots(
    output_directory,
    output_prefix,
    number_of_correspondences,
    teaser_metrics,
    final_metrics,
    teaser_residuals,
    final_residuals,
    final_stage,
    total_scale_factor,
    runtime_seconds,
):
    """
    Exportiert die reduzierten Kernmetriken und die zwei Diagramme.

    Diese Funktion ist bewusst kompakt gehalten, damit alle wichtigen
    Ergebnisse an einer zentralen Stelle geschrieben werden.
    """
    core_metrics_row = {
        "final_stage": final_stage,
        "n_corrs": int(number_of_correspondences),
        "final_fitness": float(final_metrics.get("fitness", np.nan)),
        "final_trim30": float(final_metrics.get("trim30", np.nan)),
        "p95_final": float(final_metrics.get("p95", np.nan)),
        "final_rmse": compute_rmse_from_distances(final_residuals),
        "S_total": float(total_scale_factor),
        "runtime_s": float(runtime_seconds),
    }

    save_core_metrics_csv(
        output_directory / f"{output_prefix}__core_metrics.csv",
        core_metrics_row,
    )

    save_stage_comparison_plot(
        output_directory / f"{output_prefix}__registration_stage_comparison.png",
        teaser_metrics,
        final_metrics,
    )

    save_residual_cdf_plot(
        output_directory / f"{output_prefix}__registration_residual_cdf.png",
        teaser_residuals,
        final_residuals,
    )

    return core_metrics_row


# ══════════════════════════════════════════════════════════════════════
# HAUPTPIPELINE
# ══════════════════════════════════════════════════════════════════════

def wsl_runner_main(configuration=None):
    global CONFIG
    initialize_wsl_dependencies()

    if configuration is not None:
        CONFIG = configuration

    run_start_time = time.time()
    config = CONFIG

    run_id = config.get("run_id") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("run_name", run_id)
    output_directory = Path(config["windows_output_directory_wsl"])
    use_target_as_master = bool(config.get("use_target_as_master", False))

    output_directory.mkdir(parents=True, exist_ok=True)

    # ── Startmaßstäbe für SRC und TGT ────────────────────────────────
    maximum_retry_count = config.get("maximum_scale_retry_count", 2) if config.get("enable_scale_retry") else 0
    current_one_meter_source = config["one_meter_source"]
    current_one_meter_target = config["one_meter_target"]

    print("=" * 60, flush=True)
    if use_target_as_master:
        print(f"  Registrierung (Master-Mode + Scale-Retry)  [{run_name}]", flush=True)
        print("  TGT = MASTER – wird NICHT skaliert oder transformiert", flush=True)
    else:
        print(f"  Registrierung (Scale-Retry)  [{run_name}]", flush=True)
    print(
        f"  Scale-Retry: {'aktiviert, max ' + str(maximum_retry_count) + ' Versuche' if maximum_retry_count > 0 else 'deaktiviert'}",
        flush=True,
    )
    print("=" * 60, flush=True)
    print(f"  SRC: {config['source_point_cloud_wsl']}", flush=True)
    print(f"  TGT: {config['target_point_cloud_wsl']}", flush=True)
    print(
        f"  Referenzmaß SRC: {config['source_reference_length_in_cloud_units']} cloud-units = "
        f"{config['source_reference_real_length_meters']} m  ->  one_meter_src = {current_one_meter_source:.6f}",
        flush=True,
    )
    if not use_target_as_master:
        print(
            f"  Referenzmaß TGT: {config['target_reference_length_in_cloud_units']} cloud-units = "
            f"{config['target_reference_real_length_meters']} m  ->  one_meter_tgt = {current_one_meter_target:.6f}",
            flush=True,
        )
    else:
        print("  Referenzmaß TGT: Master (bereits metrisch, keine Skalierung)", flush=True)
    print("=" * 60, flush=True)

    # ── Eingabedateien prüfen ────────────────────────────────────────
    for label, point_cloud_path in [("SRC", config["source_point_cloud_wsl"]), ("TGT", config["target_point_cloud_wsl"] )]:
        path_object = Path(point_cloud_path)
        if not path_object.exists():
            raise FileNotFoundError(f"{label} nicht gefunden: {point_cloud_path}")
        print(
            f"[OK] {label}: {path_object.name}  ({path_object.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )

    # ── Punktwolken laden ────────────────────────────────────────────
    print("\n── Lade PCDs ──", flush=True)
    source_raw = remove_non_finite_points(o3d.io.read_point_cloud(config["source_point_cloud_wsl"]))
    target_raw = remove_non_finite_points(o3d.io.read_point_cloud(config["target_point_cloud_wsl"]))

    if source_raw.is_empty() or target_raw.is_empty():
        raise RuntimeError("Eine oder beide PCDs sind leer!")

    print(
        f"[INFO] Farben vorhanden – SRC: {source_raw.has_colors()}  TGT: {target_raw.has_colors()}",
        flush=True,
    )

    # ── Variablen für Retry und bestes Ergebnis ──────────────────────
    icp_accepted = False
    attempt_log = []
    best_result = None

    # ── Hauptloop für optionalen Scale-Retry ─────────────────────────
    for attempt_index in range(maximum_retry_count + 1):
        print(f"\n{'#' * 60}", flush=True)
        print(
            f"  VERSUCH {attempt_index}/{maximum_retry_count}  "
            f"one_meter_src={current_one_meter_source:.6f}  "
            f"one_meter_tgt={current_one_meter_target:.6f}",
            flush=True,
        )
        print(f"{'#' * 60}", flush=True)

        # Für jeden Versuch werden frische Kopien verwendet.
        source_current = copy.deepcopy(source_raw)
        target_current = copy.deepcopy(target_raw)

        # SRC wird immer gemäß Referenzmaß skaliert.
        scale_point_cloud_to_metric_units(source_current, 1.0 / current_one_meter_source)

        # TGT wird nur skaliert, wenn es NICHT als Master fixiert ist.
        if use_target_as_master:
            print("[MASTER] TGT wird nicht skaliert (gilt als metrisch)", flush=True)
        else:
            scale_point_cloud_to_metric_units(target_current, 1.0 / current_one_meter_target)

        if attempt_index == 0:
            print_point_cloud_info("src_metric", source_current)
            print_point_cloud_info("tgt_metric", target_current)
            print("\n── Filter ──", flush=True)
            print(
                f"[FILTER] z_clip={config.get('enable_z_clip')}  "
                f"lo={config.get('z_clip_lower_quantile')}  "
                f"hi={config.get('z_clip_upper_quantile')}",
                flush=True,
            )
            print(
                f"[FILTER] stat_outlier nb={config.get('statistical_outlier_nb_neighbors')}  "
                f"std={config.get('statistical_outlier_std_ratio')}",
                flush=True,
            )

        source_filtered = filter_point_cloud_with_optional_color_transfer(source_current)
        target_filtered = filter_point_cloud_with_optional_color_transfer(target_current)

        if attempt_index == 0:
            print_point_cloud_info("src_filtered", source_filtered)
            print_point_cloud_info("tgt_filtered", target_filtered)

        voxel_size = float(config["voxel_size_meters"])
        teaser_noise_bound = float(config["teaser_noise_bound_meters"])

        # ── Korrespondenzen aufbauen ────────────────────────────────
        print(f"\n── FPFH  voxel={voxel_size:.2f} m ──", flush=True)
        (
            source_points_for_teaser,
            target_points_for_teaser,
            _,
            number_of_correspondences,
        ) = build_teaser_correspondences(
            source_filtered,
            target_filtered,
            voxel_size,
            int(config["maximum_number_of_correspondences"]),
        )

        if source_points_for_teaser is None:
            print("[ERROR] Zu wenige Korrespondenzen! Abbruch.", flush=True)
            attempt_log.append(
                {
                    "attempt": attempt_index,
                    "error": "no_correspondences",
                    "one_meter_source": current_one_meter_source,
                    "one_meter_target": current_one_meter_target,
                }
            )
            break

        print(f"[CORRS] {number_of_correspondences}", flush=True)

        # ── TEASER++ Grobregistrierung ──────────────────────────────
        print(f"\n── TEASER++  noise_bound={teaser_noise_bound:.3f} ──", flush=True)
        teaser_transformation, teaser_scale = solve_registration_with_teaser(
            source_points_for_teaser,
            target_points_for_teaser,
            teaser_noise_bound,
            config["enable_scale_estimation"],
        )
        print(f"  scale={teaser_scale:.5f}", flush=True)
        print(
            f"  T =\n{np.array2string(teaser_transformation, precision=5, suppress_small=True)}",
            flush=True,
        )

        source_after_teaser = copy.deepcopy(source_filtered)
        source_after_teaser.transform(teaser_transformation)

        teaser_metrics = compute_alignment_metrics(source_after_teaser, target_filtered, voxel_size)
        print(f"\n── Metriken nach TEASER ──\n  {teaser_metrics}", flush=True)

        # Standardmäßig gilt TEASER zunächst als finales Ergebnis.
        total_transformation = teaser_transformation.copy()
        icp_information = None
        icp_accepted = False
        final_stage = "teaser_only"
        final_metrics = dict(teaser_metrics)
        final_source_aligned = copy.deepcopy(source_after_teaser)

        # ── ICP nur bei genügend Überlappung ────────────────────────
        if teaser_metrics.get("fitness", 0) >= float(config.get("minimum_fitness_for_icp", 0.35)):
            print("\n── ICP Verfeinerung ──", flush=True)
            icp_result = refine_registration_with_icp(source_after_teaser, target_filtered, voxel_size)

            source_after_icp = copy.deepcopy(source_after_teaser)
            source_after_icp.transform(icp_result.transformation)

            icp_metrics = compute_alignment_metrics(source_after_icp, target_filtered, voxel_size)
            print(f"  Metriken nach ICP: {icp_metrics}", flush=True)

            # ICP-Akzeptanzlogik:
            # 1. guter Kernbereich darf nicht schlechter werden
            # 2. globale fitness muss sich verbessern
            core_region_ok = (
                icp_metrics.get("trim30", 1.0)
                <= float(config.get("icp_accept_trim30_factor", 1.05))
                * teaser_metrics.get("trim30", 1.0)
            )
            global_improvement_ok = icp_metrics.get("fitness", 0.0) > teaser_metrics.get("fitness", 0.0)

            if core_region_ok and global_improvement_ok:
                print("[ICP] *** AKZEPTIERT ***", flush=True)
                icp_accepted = True
                final_source_aligned = source_after_icp
                total_transformation = icp_result.transformation @ total_transformation
                icp_information = {
                    "accepted": True,
                    "fitness": icp_result.fitness,
                    "rmse": icp_result.inlier_rmse,
                }
                final_stage = "teaser_plus_icp"
                final_metrics = dict(icp_metrics)
            else:
                print("[ICP] ABGELEHNT (keine Verbesserung)", flush=True)
                icp_information = {
                    "accepted": False,
                    "metrics_teaser": teaser_metrics,
                    "metrics_after_icp": icp_metrics,
                }
                final_stage = "teaser_only"
                final_metrics = dict(teaser_metrics)
        else:
            print("[ICP] übersprungen (zu geringe Überlappung)", flush=True)
            icp_information = {
                "accepted": False,
                "reason": "low_fitness_after_teaser",
            }
            final_stage = "teaser_only"
            final_metrics = dict(teaser_metrics)

        # Gesamtmaßstab aus finaler 4x4-Transformation ableiten.
        total_scale_factor = float(np.cbrt(abs(np.linalg.det(total_transformation[:3, :3]))))
        print(f"\n[SCALE] Gesamt-Skalierung Versuch {attempt_index}: {total_scale_factor:.6f}", flush=True)

        # Versuch vollständig protokollieren.
        attempt_log.append(
            {
                "attempt": attempt_index,
                "one_meter_source": current_one_meter_source,
                "one_meter_target": current_one_meter_target,
                "voxel_size": voxel_size,
                "noise_bound": teaser_noise_bound,
                "S_total": total_scale_factor,
                "teaser_scale": teaser_scale,
                "icp_accepted": icp_accepted,
                "final_stage": final_stage,
                "metrics_teaser": teaser_metrics,
                "metrics_final": final_metrics,
                "n_corrs": number_of_correspondences,
            }
        )

        # Bestes Ergebnis merken.
        best_result = {
            "source_after_teaser": source_after_teaser,
            "source_aligned_final": final_source_aligned,
            "target_filtered": target_filtered,
            "T_total": total_transformation,
            "S_total": total_scale_factor,
            "icp_info": icp_information,
            "teaser_scale": teaser_scale,
            "n_corrs": number_of_correspondences,
            "final_stage": final_stage,
            "teaser_metrics": teaser_metrics,
            "final_metrics": final_metrics,
        }

        if icp_accepted:
            print(f"\n[SCALE-RETRY] Erfolg bei Versuch {attempt_index}!", flush=True)
            break

        # ── Optionaler Scale-Retry ─────────────────────────────────
        if attempt_index < maximum_retry_count:
            previous_one_meter_source = current_one_meter_source
            current_one_meter_source = current_one_meter_source * total_scale_factor
            print(
                f"\n[SCALE-RETRY] Scale-Korrektur Versuch {attempt_index} -> {attempt_index + 1}:",
                flush=True,
            )
            print(
                f"[SCALE-RETRY]   one_meter_src: {previous_one_meter_source:.6f} -> {current_one_meter_source:.6f}  "
                f"(Faktor: {total_scale_factor:.6f})",
                flush=True,
            )
        else:
            print(
                f"\n[SCALE-RETRY] Max. Versuche ({maximum_retry_count}) erreicht – bestes Ergebnis wird gespeichert.",
                flush=True,
            )

    # ── Falls gar kein Ergebnis zustande kam ──────────────────────────
    result = best_result
    if result is None:
        print("\n[RESULT] FEHLGESCHLAGEN – keine Korrespondenzen!", flush=True)
        error_payload = {
            "run_id": run_id,
            "run_name": run_name,
            "version": "commented_registration_runner",
            "icp_accepted": False,
            "error": "no_correspondences",
            "attempt_log": attempt_log,
            "runtime_s": round(time.time() - run_start_time, 1),
        }
        json_output_path = output_directory / f"{run_name}__result.json"
        json_output_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        print(f"[OUT] {json_output_path}", flush=True)
        return

    # ── Finale Restabstände und Exporte ───────────────────────────────
    runtime_seconds = round(time.time() - run_start_time, 1)

    teaser_residuals = compute_residual_distances(
        result["source_after_teaser"],
        result["target_filtered"],
        voxel_size,
    )
    final_residuals = compute_residual_distances(
        result["source_aligned_final"],
        result["target_filtered"],
        voxel_size,
    )

    core_metrics = export_registration_metrics_and_plots(
        output_directory=output_directory,
        output_prefix=run_name,
        number_of_correspondences=result["n_corrs"],
        teaser_metrics=result["teaser_metrics"],
        final_metrics=result["final_metrics"],
        teaser_residuals=teaser_residuals,
        final_residuals=final_residuals,
        final_stage=result["final_stage"],
        total_scale_factor=result["S_total"],
        runtime_seconds=runtime_seconds,
    )

    # ── Konsolenausgabe der wichtigsten Ergebnisse ────────────────────
    print(f"\n[RESULT] final_stage: {result['final_stage']}", flush=True)
    print(f"[RESULT] ICP akzeptiert: {icp_accepted}", flush=True)
    print(f"[RESULT] S_total: {result['S_total']:.6f}", flush=True)
    print(f"[RESULT] Finale one_meter_src: {current_one_meter_source:.6f}", flush=True)
    print(f"[RESULT] Versuche: {len(attempt_log)}", flush=True)
    print(f"[RESULT] final_fitness: {core_metrics['final_fitness']:.4f}", flush=True)
    print(f"[RESULT] final_trim30: {core_metrics['final_trim30']:.4f} m", flush=True)
    print(f"[RESULT] p95_final: {core_metrics['p95_final']:.4f} m", flush=True)
    print(f"[RESULT] final_rmse: {core_metrics['final_rmse']:.4f} m", flush=True)

    # ── Maßstabs-Hinweise ─────────────────────────────────────────────
    if use_target_as_master:
        print("\n[SCALE] Master-Mode: TGT ist fix.", flush=True)
        print(
            f"[SCALE] Finale one_meter_src für nächsten Lauf: {current_one_meter_source:.6f} "
            f"(= SOURCE_REFERENCE_LENGTH_IN_CLOUD_UNITS anpassen)",
            flush=True,
        )
    else:
        print(
            f"[SCALE] Vorschlag one_meter_tgt (wenn SRC metrisch): "
            f"{current_one_meter_target * result['S_total']:.6f}",
            flush=True,
        )
        print(
            f"[SCALE] Vorschlag one_meter_src (wenn TGT metrisch): {current_one_meter_source:.6f}",
            flush=True,
        )

    # ── JSON-Export mit allen Details ────────────────────────────────
    json_payload = {
        "run_id": run_id,
        "run_name": run_name,
        "version": "commented_registration_runner",
        "use_target_as_master": use_target_as_master,
        "source": config["source_point_cloud_wsl"],
        "target": config["target_point_cloud_wsl"],
        "source_reference_real_length_meters": config["source_reference_real_length_meters"],
        "source_reference_length_in_cloud_units": config["source_reference_length_in_cloud_units"],
        "initial_one_meter_source": config["one_meter_source"],
        "final_one_meter_source": current_one_meter_source,
        "target_reference_real_length_meters": config["target_reference_real_length_meters"] if not use_target_as_master else None,
        "target_reference_length_in_cloud_units": config["target_reference_length_in_cloud_units"] if not use_target_as_master else None,
        "initial_one_meter_target": config["one_meter_target"],
        "final_one_meter_target": current_one_meter_target,
        "voxel_size": voxel_size,
        "noise_bound": teaser_noise_bound,
        "noise_ratio": round(teaser_noise_bound / voxel_size, 4) if voxel_size > 0 else 0,
        "enable_z_clip": config.get("enable_z_clip"),
        "z_clip_lower_quantile": config.get("z_clip_lower_quantile"),
        "z_clip_upper_quantile": config.get("z_clip_upper_quantile"),
        "statistical_outlier_nb_neighbors": config.get("statistical_outlier_nb_neighbors"),
        "statistical_outlier_std_ratio": config.get("statistical_outlier_std_ratio"),
        "minimum_required_correspondences": config.get("minimum_required_correspondences"),
        "minimum_fitness_for_icp": config.get("minimum_fitness_for_icp"),
        "T_total": result["T_total"].tolist(),
        "S_total": result["S_total"],
        "icp_accepted": icp_accepted,
        "final_stage": result["final_stage"],
        "teaser_metrics": result["teaser_metrics"],
        "final_metrics": result["final_metrics"],
        "core_metrics": core_metrics,
        "attempt_log": attempt_log,
        "icp": result["icp_info"],
        "runtime_s": runtime_seconds,
    }

    json_output_path = output_directory / f"{run_name}__result.json"
    json_output_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(f"\n[OUT] {json_output_path}", flush=True)

    # ── PLY-Export des finalen Ergebnisses ────────────────────────────
    if config.get("export_aligned"):
        aligned_source_output_path = output_directory / f"{run_name}__src_aligned.ply"
        target_output_path = output_directory / (
            f"{run_name}__tgt_master.ply" if use_target_as_master else f"{run_name}__tgt_scaled.ply"
        )
        merged_output_path = output_directory / f"{run_name}__merged.ply"

        o3d.io.write_point_cloud(str(aligned_source_output_path), result["source_aligned_final"], write_ascii=False)
        o3d.io.write_point_cloud(str(target_output_path), result["target_filtered"], write_ascii=False)
        o3d.io.write_point_cloud(
            str(merged_output_path),
            result["target_filtered"] + result["source_aligned_final"],
            write_ascii=False,
        )

        print(f"[OUT] {aligned_source_output_path}", flush=True)
        print(f"[OUT] {target_output_path}", flush=True)
        print(f"[OUT] {merged_output_path}", flush=True)
        print(f"[EXPORT] Final stage exported: {result['final_stage']}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(
        f"  Fertig in {time.time() - run_start_time:.1f} s  "
        f"ICP: {'OK' if icp_accepted else 'FAIL'}  "
        f"Versuche: {len(attempt_log)}",
        flush=True,
    )
    print(f"  Ausgabe: {output_directory}", flush=True)
    if use_target_as_master:
        print("  Master-Mode: *src_aligned.ply wurde auf *tgt_master.ply registriert", flush=True)
    else:
        print("  Für Change Detection: *src_aligned.ply  vs  *tgt_scaled.ply", flush=True)
    print("=" * 60, flush=True)

# ══════════════════════════════════════════════════════════════════════
# WINDOWS-STARTER
# ══════════════════════════════════════════════════════════════════════

def copy_text_to_wsl_file(text_content: str, destination_path_in_wsl: str) -> None:
    """
    Schreibt Textinhalt von Windows nach WSL in eine Datei.

    Das wird hier bewusst generisch genutzt, damit sowohl die Konfiguration
    als auch dieses Skript selbst sauber nach WSL kopiert werden können –
    ohne irgendeine String-Injection des Runner-Codes.
    """
    subprocess.run(
        [
            "wsl",
            "-d",
            WSL_DISTRIBUTION_NAME,
            "--",
            "bash",
            "-lc",
            f"mkdir -p '{Path(destination_path_in_wsl).parent.as_posix()}' && cat > '{destination_path_in_wsl}'",
        ],
        input=text_content,
        text=True,
        encoding="utf-8",
        check=True,
    )


def run_windows_launcher():
    """
    Windows-Einstiegspunkt.

    Aufgaben des Windows-Teils:
    1. Eingabepfade prüfen
    2. Konfigurationswerte zusammenstellen
    3. Dieses Skript + Konfiguration nach WSL kopieren
    4. Dasselbe Skript in WSL im Runner-Modus starten
    """
    export_directory = Path(WINDOWS_EXPORT_DIRECTORY)
    export_directory.mkdir(parents=True, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{make_filename_safe(RUN_TAG)}__{run_id}" if RUN_TAG.strip() else run_id

    print(f"[Windows] run={run_name}")
    print(f"[Windows] USE_TARGET_AS_MASTER={USE_TARGET_AS_MASTER}")
    print(f"[Windows] ENABLE_SCALE_RETRY={ENABLE_SCALE_RETRY}  MAX={MAXIMUM_SCALE_RETRY_COUNT}")

    # ── Maßstabsfaktoren aus Referenzmaßen ableiten ───────────────────
    one_meter_source = SOURCE_REFERENCE_LENGTH_IN_CLOUD_UNITS / SOURCE_REFERENCE_REAL_LENGTH_METERS

    if USE_TARGET_AS_MASTER:
        one_meter_target = 1.0
        print(
            f"[Windows] SRC Scale: {SOURCE_REFERENCE_LENGTH_IN_CLOUD_UNITS} cu / {SOURCE_REFERENCE_REAL_LENGTH_METERS} m "
            f"-> one_meter_src = {one_meter_source:.6f}"
        )
        print("[Windows] TGT = Master (metrisch, one_meter_tgt = 1.0)")
    else:
        one_meter_target = TARGET_REFERENCE_LENGTH_IN_CLOUD_UNITS / TARGET_REFERENCE_REAL_LENGTH_METERS
        print(
            f"[Windows] SRC Scale: {SOURCE_REFERENCE_LENGTH_IN_CLOUD_UNITS} cu / {SOURCE_REFERENCE_REAL_LENGTH_METERS} m "
            f"-> one_meter_src = {one_meter_source:.6f}"
        )
        print(
            f"[Windows] TGT Scale: {TARGET_REFERENCE_LENGTH_IN_CLOUD_UNITS} cu / {TARGET_REFERENCE_REAL_LENGTH_METERS} m "
            f"-> one_meter_tgt = {one_meter_target:.6f}"
        )

    # ── Eingabedateien prüfen ─────────────────────────────────────────
    for label, path_string in [("SRC", WINDOWS_SOURCE_POINT_CLOUD), ("TGT", WINDOWS_TARGET_POINT_CLOUD)]:
        path_object = Path(path_string)
        if not path_object.is_file():
            raise FileNotFoundError(f"[Windows] {label} nicht gefunden: {path_object}")
        print(f"[Windows] {label}: {path_object.name}  ({path_object.stat().st_size / 1e6:.1f} MB)")

    # ── Konfiguration für WSL zusammenbauen ───────────────────────────
    configuration = {
        # Dateipfade
        "source_point_cloud_wsl": convert_windows_path_to_wsl_path(WINDOWS_SOURCE_POINT_CLOUD),
        "target_point_cloud_wsl": convert_windows_path_to_wsl_path(WINDOWS_TARGET_POINT_CLOUD),
        "windows_output_directory_wsl": convert_windows_path_to_wsl_path(str(export_directory)),
        "wsl_working_directory": WSL_WORKING_DIRECTORY,

        # Master-Logik
        "use_target_as_master": USE_TARGET_AS_MASTER,

        # Referenzmaße für Logging und Maßstab
        "source_reference_real_length_meters": SOURCE_REFERENCE_REAL_LENGTH_METERS,
        "source_reference_length_in_cloud_units": SOURCE_REFERENCE_LENGTH_IN_CLOUD_UNITS,
        "target_reference_real_length_meters": TARGET_REFERENCE_REAL_LENGTH_METERS,
        "target_reference_length_in_cloud_units": TARGET_REFERENCE_LENGTH_IN_CLOUD_UNITS,

        # Startmaßstäbe
        "one_meter_source": one_meter_source,
        "one_meter_target": one_meter_target,

        # Scale-Retry
        "enable_scale_retry": ENABLE_SCALE_RETRY,
        "maximum_scale_retry_count": MAXIMUM_SCALE_RETRY_COUNT,

        # Registrierung
        "voxel_size_meters": VOXEL_SIZE_METERS,
        "teaser_noise_bound_meters": TEASER_NOISE_BOUND_METERS,
        "maximum_number_of_correspondences": MAXIMUM_NUMBER_OF_CORRESPONDENCES,
        "enable_scale_estimation": ENABLE_SCALE_ESTIMATION,
        "export_aligned": EXPORT_FINAL_ALIGNED_POINT_CLOUDS,

        # Filter
        "enable_z_clip": ENABLE_Z_CLIP,
        "z_clip_lower_quantile": Z_CLIP_LOWER_QUANTILE,
        "z_clip_upper_quantile": Z_CLIP_UPPER_QUANTILE,
        "statistical_outlier_nb_neighbors": STATISTICAL_OUTLIER_NB_NEIGHBORS,
        "statistical_outlier_std_ratio": STATISTICAL_OUTLIER_STD_RATIO,

        # Korrespondenzen
        "minimum_required_correspondences": MINIMUM_REQUIRED_CORRESPONDENCES,
        "correspondence_grid_cell_size_multiplier": CORRESPONDENCE_GRID_CELL_SIZE_MULTIPLIER,
        "maximum_correspondences_per_grid_cell": MAXIMUM_CORRESPONDENCES_PER_GRID_CELL,

        # Normalen und FPFH
        "normal_search_radius_multiplier": NORMAL_SEARCH_RADIUS_MULTIPLIER,
        "normal_search_max_neighbors": NORMAL_SEARCH_MAX_NEIGHBORS,
        "fpfh_search_radius_multiplier": FPFH_SEARCH_RADIUS_MULTIPLIER,
        "fpfh_search_max_neighbors": FPFH_SEARCH_MAX_NEIGHBORS,

        # ICP-Entscheidungen
        "alignment_max_distance_multiplier": ALIGNMENT_MAX_DISTANCE_MULTIPLIER,
        "minimum_fitness_for_icp": MINIMUM_FITNESS_FOR_ICP,
        "icp_gate_distance_multiplier": ICP_GATE_DISTANCE_MULTIPLIER,
        "icp_point_to_point_max_distance_multiplier": ICP_POINT_TO_POINT_MAX_DISTANCE_MULTIPLIER,
        "icp_point_to_plane_max_distance_multiplier": ICP_POINT_TO_PLANE_MAX_DISTANCE_MULTIPLIER,
        "icp_point_to_point_max_iterations": ICP_POINT_TO_POINT_MAX_ITERATIONS,
        "icp_point_to_plane_max_iterations": ICP_POINT_TO_PLANE_MAX_ITERATIONS,
        "icp_accept_trim30_factor": ICP_ACCEPT_TRIM30_FACTOR,

        # TEASER++
        "teaser_cbar2": TEASER_CBAR2,
        "teaser_rotation_gnc_factor": TEASER_ROTATION_GNC_FACTOR,
        "teaser_rotation_max_iterations": TEASER_ROTATION_MAX_ITERATIONS,
        "teaser_rotation_cost_threshold": TEASER_ROTATION_COST_THRESHOLD,

        # Laufdaten
        "run_id": run_id,
        "run_name": run_name,
    }

    # ── Dieses Skript + Konfiguration nach WSL schreiben ──────────────
    runner_directory_in_wsl = f"{WSL_WORKING_DIRECTORY}/runs/{run_name}"
    runner_script_path_in_wsl = f"{runner_directory_in_wsl}/runner.py"
    runner_config_path_in_wsl = f"{runner_directory_in_wsl}/config.json"
    console_log_path_in_wsl = f"{runner_directory_in_wsl}/console.txt"

    current_script_path = Path(__file__).resolve()
    current_script_content = current_script_path.read_text(encoding="utf-8")
    configuration_json = json.dumps(configuration, indent=2, ensure_ascii=False)

    print(f"[Windows] Kopiere Skript nach WSL: {runner_script_path_in_wsl}")
    copy_text_to_wsl_file(current_script_content, runner_script_path_in_wsl)

    print(f"[Windows] Kopiere Konfiguration nach WSL: {runner_config_path_in_wsl}")
    copy_text_to_wsl_file(configuration_json, runner_config_path_in_wsl)

    # ── WSL-Runner starten ────────────────────────────────────────────
    command = [
        "wsl",
        "-d",
        WSL_DISTRIBUTION_NAME,
        "--",
        "bash",
        "-lc",
        (
            f"set -euo pipefail; "
            f"mkdir -p '{runner_directory_in_wsl}'; "
            f"source '{WSL_VENV_ACTIVATE}'; "
            f"PYTHONUNBUFFERED=1 python3 -u '{runner_script_path_in_wsl}' "
            f"--wsl-runner --config '{runner_config_path_in_wsl}' 2>&1 | tee '{console_log_path_in_wsl}'"
        ),
    ]

    print("[Windows] Starte WSL ...")
    result = subprocess.run(command)
    sys.exit(result.returncode)


def parse_arguments():
    """
    Parst die optionalen CLI-Argumente.

    Standardfall:
        python teaser.py
    -> startet den Windows-Launcher

    WSL-intern:
        python3 runner.py --wsl-runner --config /pfad/config.json
    -> startet die eigentliche Punktwolkenpipeline
    """
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--wsl-runner",
        action="store_true",
        help="Startet den internen WSL-Runner anstelle des Windows-Launchers.",
    )
    argument_parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Pfad zu einer JSON-Konfiguration für den WSL-Runner.",
    )
    return argument_parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.wsl_runner:
        if not arguments.config:
            raise ValueError("--config fehlt für --wsl-runner")
        configuration = json.loads(Path(arguments.config).read_text(encoding="utf-8"))
        wsl_runner_main(configuration)
        return

    run_windows_launcher()


if __name__ == "__main__":
    main()
