"""
vision.py — compatibility shim

All logic has been split into focused modules:
  vision_core.py   — screen capture, template matching, OCR engine, utilities
  army_scanner.py  — ArmyScanner class (grid detection, slot reading, scrolling)
  map_scanner.py   — MapScanner class  (full-map scan, coordinate conversion)

This file re-exports everything so any legacy import still works.
"""

from vision_core import (  # noqa: F401
    get_ocr_engine,
    imread_safe,
    find_hwnd,
    capture_game_ignore_ui,
    find_image_and_location,
    find_all_matches,
)
from army_scanner import ArmyScanner as _ArmyScanner

_scanner = None

def _get_scanner():
    global _scanner
    if _scanner is None:
        _scanner = _ArmyScanner()
    return _scanner


# ---------------------------------------------------------------------------
# Backward-compatible function aliases (delegate to ArmyScanner instance)
# ---------------------------------------------------------------------------

def debug_roi_view(screen):
    return _get_scanner().debug_roi_view(screen)




def get_army_grid_slots(screen, scan_zone="top"):
    return _get_scanner().get_slots(screen, scan_zone=scan_zone)

def read_army_count(screen):
    return _get_scanner().read_count(screen)

def perform_army_scroll(window_title, grid_info, drag_px):
    return _get_scanner().scroll(window_title, grid_info, drag_px)


def identify_army_content(cell_img, status_tpls=None, type_tpls=None, slot_id=0):
    """Backward-compat wrapper — status_tpls/type_tpls ignored (loaded by ArmyScanner)."""
    return _get_scanner().identify_content(cell_img, slot_id=slot_id)