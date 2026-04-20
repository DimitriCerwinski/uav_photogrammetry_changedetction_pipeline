#!/usr/bin/env python3
"""
pycolmap SfM + MVS Pipeline
===========================

Ziel dieser Datei
-----------------
Dieses Skript startet eine vollständige Rekonstruktionspipeline mit pycolmap:

1. Feature-Extraktion
2. Bild-Matching
3. Sparse Reconstruction (SfM)
4. Dense Reconstruction (MVS)
5. Export der Punktwolken
6. Export von Kernmetriken und Diagrammen

Die Datei ist absichtlich in zwei Bereiche getrennt:

- Windows-Teil:
  Startet das Skript aus Windows heraus und kopiert es nach WSL.
- WSL/Linux-Teil:
  Führt die eigentliche Rekonstruktion mit pycolmap auf Linux/WSL aus.

Die Syntax ist bewusst ausgeschrieben und ausführlich kommentiert,
damit der Ablauf später in der Arbeit oder beim Nachlesen leichter
verstanden werden kann.
"""

import os
import sys
import subprocess
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# KONFIGURATION – HIER ANPASSEN
# ══════════════════════════════════════════════════════════════════════

# Windows-Ordner mit den Eingabebildern.
# Dies ist der Datensatz, der rekonstruiert werden soll.
WINDOWS_IMAGE_DIRECTORY = r"C:\Users\Admin\Desktop\Photogrammetrie\dataset\flight_No7A"

# Beispiel für alternativen Datensatz:
# WINDOWS_IMAGE_DIRECTORY = r"C:\Users\Admin\Desktop\Photogrammetrie\dataset\Flight_No6B"

# Zielpfad in WSL, in den dieses Skript vor der Ausführung kopiert wird.
WSL_TARGET_SCRIPT_PATH = "/home/dc/photogrammetry/pycolmap/sfm.py"

# Windows-Zielordner für alle Exporte:
# - sparse.ply
# - dense fused .ply
# - CSV mit Kernmetriken
# - Diagramme als PNG
WINDOWS_EXPORT_DIRECTORY = r"C:\Users\Admin\Desktop\Photogrammetrie\pycolmap_exports"

# Dense-Verhalten:
# True  -> Dense-Workspace wird zu Beginn gelöscht.
# False -> Bereits vorhandene Depth Maps dürfen erhalten bleiben.
CLEAN_DENSE_RESULTS_ON_START = False

# StereoFusion: bewusst konservativ für stabilen Lauf.
# Einheit: Gigabyte
FUSION_CACHE_SIZE_GB = 32
# Bildkante 
FUSION_ATTEMPTS = [
    (2400, 4),
    (2000, 2),
    (1600, 1),
]


# ══════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN – PFADKONVERTIERUNG
# ══════════════════════════════════════════════════════════════════════

def convert_windows_path_to_wsl_path(path_on_windows: str) -> str:
    """
    Wandelt einen Windows-Pfad in einen WSL-Pfad um.

    Beispiel:
        C:/Users/Admin/data/images
    wird zu:
        /mnt/c/Users/Admin/data/images
    """
    absolute_path = os.path.abspath(path_on_windows).replace("\\", "/")
    drive_letter = absolute_path[0].lower()
    path_without_drive = absolute_path[2:]
    return f"/mnt/{drive_letter}/{path_without_drive}"


# ══════════════════════════════════════════════════════════════════════
# WINDOWS-STARTER
# ══════════════════════════════════════════════════════════════════════

if os.name == "nt":

    def copy_current_script_to_wsl() -> None:
        """
        Kopiert diese Python-Datei aus Windows in den definierten WSL-Zielpfad.
        Dadurch wird sichergestellt, dass immer die aktuelle Version in WSL liegt.
        """
        current_script_path_in_wsl = convert_windows_path_to_wsl_path(__file__)
        target_script_path_in_wsl = WSL_TARGET_SCRIPT_PATH

        command = [
            "wsl",
            "bash",
            "-lc",
            f"mkdir -p $(dirname {target_script_path_in_wsl}) && cp {current_script_path_in_wsl} {target_script_path_in_wsl}",
        ]

        print("[Windows] Kopiere Skript nach WSL:", target_script_path_in_wsl)
        subprocess.run(command, check=True)


    def run_pipeline_from_windows() -> None:
        """
        Windows-Einstiegspunkt.

        Ablauf:
        1. Prüfen, ob der Bildordner existiert.
        2. Skript nach WSL kopieren.
        3. pycolmap-Pipeline innerhalb von WSL starten.
        """
        if not os.path.isdir(WINDOWS_IMAGE_DIRECTORY):
            print(f"[Windows] Bildordner existiert nicht: {WINDOWS_IMAGE_DIRECTORY}")
            sys.exit(1)

        normalized_windows_image_directory = os.path.abspath(WINDOWS_IMAGE_DIRECTORY)
        print("[Windows] Verwende Bildordner:", normalized_windows_image_directory)

        copy_current_script_to_wsl()

        windows_path_for_argument = normalized_windows_image_directory.replace("\\", "/")
        wsl_command_string = (
            "cd ~/photogrammetry/pycolmap && "
            "source .venv/bin/activate && "
            f"python {WSL_TARGET_SCRIPT_PATH} --win_images \"{windows_path_for_argument}\""
        )

        command = ["wsl", "bash", "-lc", wsl_command_string]
        print("[Windows] Starte in WSL:", " ".join(command))
        result = subprocess.run(command)
        sys.exit(result.returncode)


    if __name__ == "__main__":
        run_pipeline_from_windows()


# ══════════════════════════════════════════════════════════════════════
# WSL / LINUX-TEIL
# ══════════════════════════════════════════════════════════════════════
else:
    import argparse
    import csv
    import shutil
    import time
    from datetime import datetime

    import numpy as np
    import pycolmap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    # ──────────────────────────────────────────────────────────────
    # WSL-KONFIGURATION
    # ──────────────────────────────────────────────────────────────

    def convert_windows_like_path_to_wsl_path(path_text: str) -> Path:
        """
        Wandelt einen Pfadtext in ein Path-Objekt innerhalb von WSL um.

        Diese Funktion wird genutzt, wenn ein Windows-Pfad als Argument
        in WSL ankommt.
        """
        normalized_path = path_text.replace("\\", "/")

        if len(normalized_path) >= 2 and normalized_path[1] == ":":
            drive_letter = normalized_path[0].lower()
            path_without_drive = normalized_path[2:].lstrip("/")
            return Path(f"/mnt/{drive_letter}/{path_without_drive}")

        return Path(normalized_path)


    WSL_EXPORT_DIRECTORY = convert_windows_like_path_to_wsl_path(WINDOWS_EXPORT_DIRECTORY)

    # Kamera- und Matching-Einstellungen.
    CAMERA_MODEL_NAME = "OPENCV"

    # Mögliche Optionen:
    # - "exhaustive"
    # - "sequential"
    # - "spatial"
    MATCHING_STRATEGY = "exhaustive"

    # Nur relevant, wenn MATCHING_STRATEGY == "spatial"
    SPATIAL_MAX_NEIGHBORS = 50
    SPATIAL_MIN_NEIGHBORS = 10
    SPATIAL_MAX_DISTANCE_METERS = 150.0


    # ──────────────────────────────────────────────────────────────
    # GERÄTEAUSWAHL (CPU / CUDA)
    # ──────────────────────────────────────────────────────────────

    def select_pycolmap_device():
        """
        Ermittelt, ob CUDA-Geräte verfügbar sind.

        Rückgabe:
            (device, number_of_cuda_devices)
        """
        if hasattr(pycolmap, "get_num_cuda_devices"):
            number_of_cuda_devices = pycolmap.get_num_cuda_devices()

            if number_of_cuda_devices and number_of_cuda_devices > 0:
                return pycolmap.Device.cuda, number_of_cuda_devices

            return pycolmap.Device.cpu, number_of_cuda_devices

        return pycolmap.Device.cpu, None


    # ──────────────────────────────────────────────────────────────
    # METRIKEN UND DIAGRAMME
    # ──────────────────────────────────────────────────────────────

    def count_input_images_recursively(image_directory: Path) -> int:
        """
        Zählt rekursiv alle Bilddateien im Eingabedatensatz.

        Diese Zahl ist wichtig, um später zu sehen,
        wie viele Bilder insgesamt vorhanden waren.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

        return sum(
            1
            for file_path in image_directory.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions
        )


    def read_point_count_from_ply_header(ply_file_path: Path) -> int:
        """
        Liest die Punktanzahl direkt aus dem PLY-Header.

        Das funktioniert sowohl bei ASCII-PLY als auch bei binären PLY-Dateien,
        solange der Header standardkonform ist.
        """
        with open(ply_file_path, "rb") as file_handle:
            for raw_line in file_handle:
                decoded_line = raw_line.decode("ascii", errors="ignore").strip()

                if decoded_line.startswith("element vertex "):
                    return int(decoded_line.split()[-1])

                if decoded_line == "end_header":
                    break

        raise RuntimeError(f"Konnte Anzahl der Punkte nicht aus PLY lesen: {ply_file_path}")


    def extract_sparse_reprojection_errors(reconstruction) -> np.ndarray:
        """
        Sammelt die Reprojektionsfehler aller Sparse-3D-Punkte.

        Hinweis:
        Früher wurde hier 'has_error()' verwendet. Da dies je nach pycolmap-Version
        nicht immer verfügbar ist, wird das Attribut 'error' direkt und robust gelesen.
        """
        reprojection_errors = []

        for point_3d in reconstruction.points3D.values():
            point_error = getattr(point_3d, "error", None)

            if point_error is None:
                continue

            try:
                point_error = float(point_error)
            except Exception:
                continue

            if np.isfinite(point_error) and point_error >= 0:
                reprojection_errors.append(point_error)

        return np.asarray(reprojection_errors, dtype=np.float64)


    def extract_sparse_track_lengths(reconstruction) -> np.ndarray:
        """
        Sammelt die Track-Längen aller Sparse-3D-Punkte.

        Ein Track beschreibt, in wie vielen Bildern ein 3D-Punkt beobachtet wurde.
        """
        track_lengths = []

        for point_3d in reconstruction.points3D.values():
            track_object = getattr(point_3d, "track", None)
            if track_object is None:
                continue

            if hasattr(track_object, "length"):
                try:
                    track_lengths.append(int(track_object.length()))
                    continue
                except Exception:
                    pass

            track_elements = getattr(track_object, "elements", None)
            if track_elements is not None:
                try:
                    track_lengths.append(len(track_elements))
                    continue
                except Exception:
                    pass

        return np.asarray(track_lengths, dtype=np.int32)


    def save_histogram_as_png(values, output_path: Path, title: str, x_label: str, bins: int = 50) -> None:
        """
        Speichert ein einfaches Histogramm.

        Verwendet für:
        - Reprojektionsfehler
        - Track-Längen
        """
        values_array = np.asarray(values)

        if values_array.size == 0:
            print(f"[WSL] Histogramm übersprungen (keine Werte): {output_path.name}")
            return

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(values_array, bins=bins, edgecolor="black")
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Anzahl")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print("[WSL] Saved plot:", output_path)


    def save_registered_vs_unregistered_bar_chart(output_path: Path, number_of_registered_images: int, total_number_of_images: int) -> None:
        """
        Speichert ein Balkendiagramm für:
        - registrierte Bilder
        - nicht registrierte Bilder
        """
        number_of_unregistered_images = max(0, total_number_of_images - number_of_registered_images)

        labels = ["registered", "not registered"]
        values = [number_of_registered_images, number_of_unregistered_images]

        figure, axis = plt.subplots(figsize=(6, 4))
        bars = axis.bar(labels, values, edgecolor="black")

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1, total_number_of_images * 0.01),
                f"{value}",
                ha="center",
                va="bottom",
            )

        axis.set_title("Image registration result")
        axis.set_ylabel("Anzahl Bilder")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print("[WSL] Saved plot:", output_path)


    def save_sparse_vs_dense_point_count_bar_chart(output_path: Path, number_of_sparse_points: int, number_of_dense_points: int) -> None:
        """
        Speichert ein Balkendiagramm für Sparse- vs. Dense-Punktanzahl.

        Dieses Diagramm zeigt gut, wie stark die dichte Rekonstruktion
        die Punktzahl gegenüber der Sparse-Rekonstruktion erhöht.
        """
        labels = ["sparse points", "dense fused points"]
        values = [number_of_sparse_points, number_of_dense_points]

        figure, axis = plt.subplots(figsize=(6, 4))
        bars = axis.bar(labels, values, edgecolor="black")

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1, max(values) * 0.01),
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        axis.set_title("Sparse vs. dense point count")
        axis.set_ylabel("Anzahl Punkte")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print("[WSL] Saved plot:", output_path)


    def save_reconstruction_core_metrics_to_csv(output_path: Path, metrics_row: dict) -> None:
        """
        Speichert die Kernmetriken der Rekonstruktion als 1-Zeilen-CSV.

        Dadurch lassen sich später mehrere Läufe einfach vergleichen.
        """
        with open(output_path, "w", encoding="utf-8", newline="") as file_handle:
            csv_writer = csv.DictWriter(file_handle, fieldnames=list(metrics_row.keys()), delimiter=";")
            csv_writer.writeheader()
            csv_writer.writerow(metrics_row)

        print("[WSL] Saved metrics CSV:", output_path)


    def export_reconstruction_metrics_and_plots(
        reconstruction,
        image_directory: Path,
        fused_dense_ply_path: Path,
        export_directory: Path,
        export_prefix: str,
        total_runtime_seconds: float,
    ) -> None:
        """
        Exportiert das vollständige, bewusst schlanke Metrikpaket.

        Exportiert werden:
        - 1 CSV mit Kernmetriken
        - Histogramm der Reprojektionsfehler
        - Histogramm der Track-Längen
        - Balkendiagramm registriert vs. nicht registriert
        - Balkendiagramm Sparse- vs. Dense-Punktanzahl
        """
        total_number_of_input_images = count_input_images_recursively(image_directory)
        number_of_registered_images = int(reconstruction.num_reg_images())
        number_of_sparse_points = int(reconstruction.num_points3D())
        mean_reprojection_error_pixels = float(reconstruction.compute_mean_reprojection_error())
        mean_track_length = float(reconstruction.compute_mean_track_length())
        mean_observations_per_registered_image = float(reconstruction.compute_mean_observations_per_reg_image())
        number_of_dense_fused_points = int(read_point_count_from_ply_header(fused_dense_ply_path))

        registration_ratio_percent = (
            100.0 * number_of_registered_images / total_number_of_input_images
            if total_number_of_input_images > 0
            else 0.0
        )

        sparse_reprojection_errors = extract_sparse_reprojection_errors(reconstruction)
        sparse_track_lengths = extract_sparse_track_lengths(reconstruction)

        metrics_row = {
            "dataset": image_directory.name,
            "total_input_images_n": total_number_of_input_images,
            "registered_images_n": number_of_registered_images,
            "registration_ratio_pct": f"{registration_ratio_percent:.2f}",
            "sparse_points_n": number_of_sparse_points,
            "mean_reprojection_error_px": f"{mean_reprojection_error_pixels:.6f}",
            "mean_track_length": f"{mean_track_length:.6f}",
            "mean_observations_per_reg_image": f"{mean_observations_per_registered_image:.6f}",
            "dense_fused_points_n": number_of_dense_fused_points,
            "total_runtime_s": f"{total_runtime_seconds:.2f}",
        }

        save_reconstruction_core_metrics_to_csv(
            export_directory / f"{export_prefix}_reconstruction_core_metrics.csv",
            metrics_row,
        )

        save_histogram_as_png(
            sparse_reprojection_errors,
            export_directory / f"{export_prefix}_reprojection_error_histogram.png",
            title="Sparse reprojection error distribution",
            x_label="Reprojection error [px]",
            bins=50,
        )

        save_histogram_as_png(
            sparse_track_lengths,
            export_directory / f"{export_prefix}_track_length_histogram.png",
            title="Sparse track length distribution",
            x_label="Track length [images]",
            bins=50,
        )

        save_registered_vs_unregistered_bar_chart(
            export_directory / f"{export_prefix}_registered_vs_unregistered_images.png",
            number_of_registered_images=number_of_registered_images,
            total_number_of_images=total_number_of_input_images,
        )

        save_sparse_vs_dense_point_count_bar_chart(
            export_directory / f"{export_prefix}_sparse_vs_dense_points.png",
            number_of_sparse_points=number_of_sparse_points,
            number_of_dense_points=number_of_dense_fused_points,
        )

        print("[WSL] Reconstruction core metrics:")
        for key, value in metrics_row.items():
            print(f"    {key}: {value}")


    # ──────────────────────────────────────────────────────────────
    # DATEISYNC UND RUN-BEREINIGUNG
    # ──────────────────────────────────────────────────────────────

    def sync_images_to_local_cache(source_directory: Path, cache_image_directory: Path) -> None:
        """
        Kopiert bzw. synchronisiert Bilder mit rsync in einen lokalen WSL-Cache.

        Vorteil:
        Der eigentliche pycolmap-Run arbeitet dann lokal in WSL und nicht direkt
        auf dem Windows-Dateisystem.
        """
        cache_image_directory.mkdir(parents=True, exist_ok=True)

        command = [
            "rsync",
            "-a",
            "--info=stats2",
            f"{source_directory}/",
            f"{cache_image_directory}/",
        ]

        print("[WSL] Sync:", " ".join(command))
        subprocess.run(command, check=True)


    def clean_previous_run_outputs(work_directory: Path) -> None:
        """
        Entfernt alte Artefakte früherer Läufe.

        Gelöscht werden:
        - database.db
        - sparse/
        - dense/   (nur wenn CLEAN_DENSE_RESULTS_ON_START = True)
        """
        database_path = work_directory / "database.db"
        sparse_directory = work_directory / "sparse"
        dense_directory = work_directory / "dense"

        if database_path.exists():
            database_path.unlink()

        if sparse_directory.exists():
            shutil.rmtree(sparse_directory)

        if CLEAN_DENSE_RESULTS_ON_START and dense_directory.exists():
            shutil.rmtree(dense_directory)


    # ──────────────────────────────────────────────────────────────
    # DENSE-HILFSFUNKTIONEN
    # ──────────────────────────────────────────────────────────────

    def count_undistorted_images_in_dense_workspace(dense_directory: Path) -> int:
        image_directory = dense_directory / "images"
        if not image_directory.exists():
            return 0

        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        return sum(1 for file_path in image_directory.iterdir() if file_path.is_file() and file_path.suffix.lower() in valid_extensions)


    def get_depth_map_status(dense_directory: Path):
        depth_map_directory = dense_directory / "stereo" / "depth_maps"
        if not depth_map_directory.exists():
            return 0, 0

        number_of_photometric_maps = len(list(depth_map_directory.glob("*.photometric.bin")))
        number_of_geometric_maps = len(list(depth_map_directory.glob("*.geometric.bin")))
        return number_of_photometric_maps, number_of_geometric_maps


    def choose_dense_fusion_input_type(dense_directory: Path) -> str:
        """
        Wählt automatisch, welcher Typ von Depth Maps bevorzugt verwendet wird.
        """
        number_of_workspace_images = count_undistorted_images_in_dense_workspace(dense_directory)
        number_of_photometric_maps, number_of_geometric_maps = get_depth_map_status(dense_directory)

        if number_of_workspace_images > 0 and number_of_geometric_maps >= int(0.9 * number_of_workspace_images):
            return "geometric"

        if number_of_photometric_maps > 0:
            return "photometric"

        return "geometric"


    def is_process_killed_return_code(return_code: int) -> bool:
        """
        Prüft, ob ein Subprozess wahrscheinlich durch SIGKILL beendet wurde.
        """
        return return_code in (-9, 9, 137)


    def run_stereo_fusion_in_subprocess(
        dense_directory: Path,
        fused_ply_path: Path,
        input_type: str,
        max_image_size: int,
        num_threads: int,
        use_cache: bool = True,
        cache_size: int = FUSION_CACHE_SIZE_GB,
    ) -> int:
        """
        Startet die StereoFusion in einem separaten Python-Prozess.

        Das ist robuster, weil ein möglicher SIGKILL nicht das Hauptskript beendet.
        """
        worker_script_path = dense_directory / f"_fusion_worker_{input_type}_{max_image_size}_{num_threads}.py"

        worker_script_path.write_text(
            f"""\
from pathlib import Path
import pycolmap

dense_dir = Path(r\"{str(dense_directory)}\")
fused_ply = Path(r\"{str(fused_ply_path)}\")

fusion_options = pycolmap.StereoFusionOptions()
if hasattr(fusion_options, \"max_image_size\"):
    fusion_options.max_image_size = {max_image_size}
if hasattr(fusion_options, \"num_threads\"):
    fusion_options.num_threads = {num_threads}
if hasattr(fusion_options, \"use_cache\"):
    fusion_options.use_cache = {str(bool(use_cache))}
if hasattr(fusion_options, \"cache_size\"):
    fusion_options.cache_size = {cache_size}

pycolmap.stereo_fusion(
    output_path=str(fused_ply),
    workspace_path=str(dense_dir),
    workspace_format=\"COLMAP\",
    input_type=\"{input_type}\",
    options=fusion_options,
)

print(\"Wrote:\", fused_ply)
"""
        )

        command = [sys.executable, str(worker_script_path)]
        print("[WSL] Running fusion worker:", " ".join(command))
        result = subprocess.run(command)
        return result.returncode


    def run_stereo_fusion_via_colmap_cli(
        dense_directory: Path,
        fused_ply_path: Path,
        input_type: str,
        max_image_size: int,
        num_threads: int,
        use_cache: int = 1,
        cache_size: int = 64,
    ) -> int:
        """
        Fallback über die COLMAP-CLI, falls pycolmap-StereoFusion scheitert.
        """
        if shutil.which("colmap") is None:
            print("[WSL] COLMAP-CLI nicht gefunden (colmap). CLI-Fallback übersprungen.")
            return 127

        command = [
            "colmap",
            "stereo_fusion",
            "--workspace_path", str(dense_directory),
            "--workspace_format", "COLMAP",
            "--input_type", input_type,
            "--output_path", str(fused_ply_path),
            "--StereoFusion.max_image_size", str(max_image_size),
            "--StereoFusion.num_threads", str(num_threads),
            "--StereoFusion.use_cache", str(use_cache),
            "--StereoFusion.cache_size", str(cache_size),
        ]

        print("[WSL] Running COLMAP CLI fusion:", " ".join(command))
        result = subprocess.run(command)
        return result.returncode


    def run_safe_stereo_fusion(dense_directory: Path, fused_ply_path: Path) -> Path:
        """
        Führt StereoFusion robust aus.

        Strategie:
        1. Bereits vorhandene fused.ply wiederverwenden.
        2. Mehrere Versuche mit pycolmap.
        3. Falls nötig: Fallback auf COLMAP-CLI.
        """
        if fused_ply_path.exists() and fused_ply_path.stat().st_size > 0:
            print("[WSL] fused.ply existiert bereits, Fusion wird übersprungen:", fused_ply_path)
            return fused_ply_path

        selected_input_type = choose_dense_fusion_input_type(dense_directory)
        print("[WSL] Fusion input_type:", selected_input_type)

        if not FUSION_ATTEMPTS:
            raise RuntimeError("FUSION_ATTEMPTS ist leer. Bitte mindestens einen Fusion-Versuch in der Konfiguration definieren.")

        print("[WSL] Fusion attempts from config:", FUSION_ATTEMPTS)
        print("[WSL] Fusion cache size [GB]:", FUSION_CACHE_SIZE_GB)

        # Erst pycolmap-Subprozess versuchen.
        for max_image_size, number_of_threads in FUSION_ATTEMPTS:
            print(
                f"[WSL] Fusion attempt (pycolmap): max_image_size={max_image_size}, num_threads={number_of_threads}"
            )

            return_code = run_stereo_fusion_in_subprocess(
                dense_directory=dense_directory,
                fused_ply_path=fused_ply_path,
                input_type=selected_input_type,
                max_image_size=max_image_size,
                num_threads=number_of_threads,
                use_cache=True,
                cache_size=FUSION_CACHE_SIZE_GB,
            )

            if fused_ply_path.exists() and fused_ply_path.stat().st_size > 0:
                return fused_ply_path

            if is_process_killed_return_code(return_code):
                print(f"[WSL] Fusion worker wurde beendet (rc={return_code}). Neuer Versuch mit sichereren Einstellungen...")
            elif return_code != 0:
                print(f"[WSL] Fusion worker fehlgeschlagen (rc={return_code}). Neuer Versuch...")

        # Danach COLMAP-CLI als Fallback.
        for max_image_size, number_of_threads in FUSION_ATTEMPTS:
            print(
                f"[WSL] Fusion attempt (CLI): max_image_size={max_image_size}, num_threads={number_of_threads}"
            )

            return_code = run_stereo_fusion_via_colmap_cli(
                dense_directory=dense_directory,
                fused_ply_path=fused_ply_path,
                input_type=selected_input_type,
                max_image_size=max_image_size,
                num_threads=number_of_threads,
                use_cache=1,
                cache_size=FUSION_CACHE_SIZE_GB,
            )

            if fused_ply_path.exists() and fused_ply_path.stat().st_size > 0:
                return fused_ply_path

            print(f"[WSL] CLI fusion rc={return_code}")

        raise RuntimeError(
            "StereoFusion konnte nicht erfolgreich abgeschlossen werden. "
            f"Dense workspace bleibt erhalten unter: {dense_directory}"
        )

        # Danach COLMAP-CLI als Fallback.
        for max_image_size, number_of_threads in FUSION_ATTEMPTS:
            print(
                f"[WSL] Fusion attempt (CLI): max_image_size={max_image_size}, num_threads={number_of_threads}"
            )

            return_code = run_stereo_fusion_via_colmap_cli(
                dense_directory=dense_directory,
                fused_ply_path=fused_ply_path,
                input_type=selected_input_type,
                max_image_size=max_image_size,
                num_threads=number_of_threads,
                use_cache=1,
                cache_size=FUSION_CACHE_SIZE_GB,
            )

            if fused_ply_path.exists() and fused_ply_path.stat().st_size > 0:
                return fused_ply_path

            print(f"[WSL] CLI fusion rc={return_code}")

        raise RuntimeError(
            "StereoFusion konnte nicht erfolgreich abgeschlossen werden. "
            f"Dense workspace bleibt erhalten unter: {dense_directory}"
        )


    # ──────────────────────────────────────────────────────────────
    # DENSE-PIPELINE
    # ──────────────────────────────────────────────────────────────

    def run_dense_reconstruction(image_directory: Path, work_directory: Path, sparse_model_directory: Path) -> Path:
        """
        Führt die Dense-Rekonstruktion aus:
        1. Bilder entzerren
        2. PatchMatch Stereo
        3. StereoFusion
        """
        required_functions_are_available = (
            hasattr(pycolmap, "undistort_images")
            and hasattr(pycolmap, "patch_match_stereo")
            and hasattr(pycolmap, "stereo_fusion")
        )

        if not required_functions_are_available:
            raise RuntimeError(
                "pycolmap build hat keine Dense/MVS Bindings "
                "(undistort_images / patch_match_stereo / stereo_fusion)."
            )

        dense_directory = work_directory / "dense"
        dense_directory.mkdir(parents=True, exist_ok=True)

        # 1) Undistort nur dann ausführen, wenn der Dense-Workspace noch nicht vollständig existiert.
        if not (dense_directory / "images").exists() or not (dense_directory / "sparse").exists():
            print("[WSL] Bereite Dense-Workspace vor (undistort_images)...")

            if hasattr(pycolmap, "CopyType"):
                pycolmap.undistort_images(
                    output_path=str(dense_directory),
                    input_path=str(sparse_model_directory),
                    image_path=str(image_directory),
                    output_type="COLMAP",
                    copy_policy=pycolmap.CopyType.copy,
                )
            else:
                pycolmap.undistort_images(
                    output_path=str(dense_directory),
                    input_path=str(sparse_model_directory),
                    image_path=str(image_directory),
                    output_type="COLMAP",
                )
        else:
            print("[WSL] Dense-Workspace existiert bereits, undistort_images wird übersprungen:", dense_directory)

        # 2) PatchMatch nur dann ausführen, wenn Depth Maps noch nicht vollständig vorhanden sind.
        number_of_workspace_images = count_undistorted_images_in_dense_workspace(dense_directory)
        number_of_photometric_maps, number_of_geometric_maps = get_depth_map_status(dense_directory)

        print(
            f"[WSL] Workspace images={number_of_workspace_images} "
            f"depthmaps photometric={number_of_photometric_maps} geometric={number_of_geometric_maps}"
        )

        if number_of_workspace_images == 0:
            raise RuntimeError("Dense workspace images/ ist leer – undistort_images fehlgeschlagen?")

        depth_maps_are_incomplete = (
            number_of_photometric_maps < number_of_workspace_images
            or number_of_geometric_maps < number_of_workspace_images
        )

        if depth_maps_are_incomplete:
            print("[WSL] Starte PatchMatch Stereo (Depth Maps noch nicht vollständig)...")

            patch_match_options = pycolmap.PatchMatchOptions()

            if hasattr(patch_match_options, "geom_consistency"):
                patch_match_options.geom_consistency = True
            if hasattr(patch_match_options, "max_image_size"):
                patch_match_options.max_image_size = 2400
            if hasattr(patch_match_options, "num_threads"):
                patch_match_options.num_threads = max(1, os.cpu_count() or 1)

            pycolmap.patch_match_stereo(
                workspace_path=str(dense_directory),
                workspace_format="COLMAP",
                pmvs_option_name="option-all",
                options=patch_match_options,
            )
        else:
            print("[WSL] Depth Maps vollständig vorhanden, PatchMatch wird übersprungen.")

        # 3) Robuste StereoFusion.
        fused_ply_path = dense_directory / "fused.ply"
        fused_ply_path = run_safe_stereo_fusion(dense_directory=dense_directory, fused_ply_path=fused_ply_path)

        print("[WSL] Dense fused PLY (WSL):", fused_ply_path)
        return fused_ply_path


    # ──────────────────────────────────────────────────────────────
    # MATCHING-STRATEGIEN
    # ──────────────────────────────────────────────────────────────

    def run_image_matching(database_path: Path, device) -> None:
        """
        Führt das Bild-Matching mit der gewählten Strategie aus.

        Hinweise:
        - exhaustive: robust, aber langsamer
        - sequential: sinnvoll bei Video-/Flugreihenfolge
        - spatial: nutzt Positionspriors, nur sinnvoll wenn diese verlässlich sind
        """
        matching_strategy = MATCHING_STRATEGY.lower()

        if matching_strategy == "exhaustive":
            pycolmap.match_exhaustive(database_path=str(database_path), device=device)
            return

        if matching_strategy == "sequential":
            sequential_pairing_options = pycolmap.SequentialPairingOptions()
            sequential_pairing_options.overlap = 10

            pycolmap.match_sequential(
                database_path=str(database_path),
                options=sequential_pairing_options,
                device=device,
            )
            return

        if matching_strategy == "spatial":
            spatial_pairing_options = pycolmap.SpatialPairingOptions()
            spatial_pairing_options.max_num_neighbors = SPATIAL_MAX_NEIGHBORS
            spatial_pairing_options.min_num_neighbors = SPATIAL_MIN_NEIGHBORS
            spatial_pairing_options.max_distance = SPATIAL_MAX_DISTANCE_METERS
            spatial_pairing_options.ignore_z = True

            pycolmap.match_spatial(
                database_path=str(database_path),
                pairing_options=spatial_pairing_options,
                device=device,
            )
            return

        raise ValueError(f"Unbekannte Matching-Strategie: {MATCHING_STRATEGY}")


    # ──────────────────────────────────────────────────────────────
    # SPARSE-PIPELINE (SfM)
    # ──────────────────────────────────────────────────────────────

    def run_sparse_and_dense_reconstruction(image_directory: Path, work_directory: Path) -> None:
        """
        Führt die komplette Rekonstruktion aus.

        Gesamtpipeline:
        1. Features extrahieren
        2. Bilder matchen
        3. Sparse Mapping
        4. Sparse-Punktwolke exportieren
        5. Dense-Rekonstruktion erzeugen
        6. Dense-Punktwolke exportieren
        7. Metriken und Diagramme exportieren
        """
        run_start_time = time.time()

        selected_device, number_of_cuda_devices = select_pycolmap_device()

        print("[WSL] pycolmap version:", pycolmap.__version__)
        print("[WSL] CUDA devices:", "(nicht verfügbar)" if number_of_cuda_devices is None else number_of_cuda_devices)
        print("[WSL] Using device:", "cuda" if selected_device == pycolmap.Device.cuda else "cpu")

        work_directory.mkdir(parents=True, exist_ok=True)
        clean_previous_run_outputs(work_directory)

        database_path = work_directory / "database.db"
        sparse_root_directory = work_directory / "sparse"
        sparse_root_directory.mkdir(parents=True, exist_ok=True)

        print("[WSL] IMAGE_DIR:", image_directory)
        print("[WSL] WORKDIR  :", work_directory)

        # 1) Feature-Extraktion.
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(image_directory),
            camera_model=CAMERA_MODEL_NAME,
            camera_mode=pycolmap.CameraMode.SINGLE,
            device=selected_device,
        )

        # 2) Matching.
        run_image_matching(database_path=database_path, device=selected_device)

        # 3) Sparse Mapping.
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_directory),
            output_path=str(sparse_root_directory),
        )

        if not reconstructions:
            raise RuntimeError("No reconstruction created (incremental_mapping returned empty).")

        best_reconstruction_id = max(reconstructions.keys(), key=lambda reconstruction_id: reconstructions[reconstruction_id].num_images())
        best_reconstruction = reconstructions[best_reconstruction_id]

        print("[WSL] Reconstruction id    :", best_reconstruction_id)
        print("[WSL] Reconstruction images:", best_reconstruction.num_images())
        print("[WSL] Reconstruction reg   :", best_reconstruction.num_reg_images())
        print("[WSL] Reconstruction points:", best_reconstruction.num_points3D())
        print("[WSL] Mean reproj error px :", best_reconstruction.compute_mean_reprojection_error())
        print("[WSL] Mean track length    :", best_reconstruction.compute_mean_track_length())
        print("[WSL] Mean obs/reg image   :", best_reconstruction.compute_mean_observations_per_reg_image())

        sparse_model_directory = sparse_root_directory / str(best_reconstruction_id)

        for file_name in ["cameras.bin", "images.bin", "points3D.bin"]:
            model_file_path = sparse_model_directory / file_name
            print("[WSL] sparse model file:", model_file_path, "exists:", model_file_path.exists())

        if best_reconstruction.num_images() < 5 or best_reconstruction.num_points3D() < 100:
            raise RuntimeError("Sparse reconstruction too small/unstable; skip dense.")

        # 4) Sparse exportieren.
        sparse_ply_path = work_directory / "sparse.ply"
        best_reconstruction.export_PLY(str(sparse_ply_path))
        print("[WSL] Exported sparse (WSL):", sparse_ply_path)

        WSL_EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")

        sparse_ply_export_path_on_windows = WSL_EXPORT_DIRECTORY / f"{work_directory.name}_sparse_{timestamp_string}.ply"
        shutil.copy2(sparse_ply_path, sparse_ply_export_path_on_windows)
        print("[WSL] Copied sparse to Windows:", sparse_ply_export_path_on_windows)

        # 5) Dense-Rekonstruktion.
        dense_fused_ply_path = run_dense_reconstruction(
            image_directory=image_directory,
            work_directory=work_directory,
            sparse_model_directory=sparse_model_directory,
        )

        # 6) Dense-Punktwolke nach Windows kopieren.
        dense_ply_export_path_on_windows = WSL_EXPORT_DIRECTORY / f"{work_directory.name}_dense_fused_{timestamp_string}.ply"
        shutil.copy2(dense_fused_ply_path, dense_ply_export_path_on_windows)
        print("[WSL] Copied dense to Windows:", dense_ply_export_path_on_windows)

        # 7) Kernmetriken und Diagramme exportieren.
        total_runtime_seconds = time.time() - run_start_time

        export_reconstruction_metrics_and_plots(
            reconstruction=best_reconstruction,
            image_directory=image_directory,
            fused_dense_ply_path=dense_fused_ply_path,
            export_directory=WSL_EXPORT_DIRECTORY,
            export_prefix=f"{work_directory.name}_{timestamp_string}",
            total_runtime_seconds=total_runtime_seconds,
        )


    # ──────────────────────────────────────────────────────────────
    # WSL-HAUPTEINSTIEG
    # ──────────────────────────────────────────────────────────────

    def run_pipeline_from_wsl() -> None:
        """
        Einstiegspunkt für WSL/Linux.

        Erwartet einen Windows-Pfad zum Bildordner, wandelt ihn um,
        synchronisiert die Bilder in einen lokalen Cache und startet
        anschließend die vollständige Rekonstruktionspipeline.
        """
        argument_parser = argparse.ArgumentParser()
        argument_parser.add_argument(
            "--win_images",
            type=str,
            required=True,
            help="Windows images folder, e.g. C:/Users/.../images",
        )
        parsed_arguments = argument_parser.parse_args()

        source_image_directory_in_wsl = convert_windows_like_path_to_wsl_path(parsed_arguments.win_images)

        if not source_image_directory_in_wsl.exists():
            raise FileNotFoundError(
                f"[WSL] Windows images folder not found in WSL: {source_image_directory_in_wsl}"
            )

        dataset_name = source_image_directory_in_wsl.name
        cache_root_directory = Path.home() / "photogrammetry" / "datasets_cache" / dataset_name
        cache_image_directory = cache_root_directory / "images"
        run_work_directory = Path.home() / "photogrammetry" / "runs" / dataset_name

        sync_images_to_local_cache(source_image_directory_in_wsl, cache_image_directory)
        run_sparse_and_dense_reconstruction(cache_image_directory, run_work_directory)


    if __name__ == "__main__":
        run_pipeline_from_wsl()
