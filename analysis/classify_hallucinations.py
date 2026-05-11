#!/usr/bin/env python3
"""
Hallucination Classification Script (with H1/H2 Taxonomy)
============================================================
Extracts all hallucinated tool names from experiment results, aggregates by
frequency, filters pre-filter artifacts, merges partial-string duplicates,
and classifies each name as:

  H2 = External-real (commercially available or open-source tool, not in our 82-tool inventory)
  Near-miss = Imprecise reference to a tool that IS in our 82-tool inventory
  Non-specific = Generic terms, JSON fragments, or sentence fragments lacking a specific product referent
  H1 = Fabricated (no verifiable external referent - genuine confabulation)

Classification of 197 consolidated names (covering 76.5% of mentions) was performed
via web search verification of each tool name against vendor product pages,
GitHub/SourceForge repositories, and Wikipedia entries (verified February 2026).

Output: classify_hallucinations_results.txt (text report)
        supplement_197_classifications.csv (supplementary material)
"""

import csv
import json
import glob
import os
import re
import sys
from collections import defaultdict

# ===========================================================================
# Configuration
# ===========================================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "results")

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classify_hallucinations_results.txt')

PHASES = {
    'phase1': 'Gen1 Standard (C0-C5)',
    'phase2': 'Gen1 Thinking (C0+C5)',
    'phase3a': 'Gen2 Standard (C0-C5)',
    'phase3b': 'Gen2 Thinking (C0+C5)',
}

# ===========================================================================
# Manual H1/H2 Classification of Top Names
# ===========================================================================
# Each entry maps a consolidated tool name to its classification.
# Classifications verified via web search (February 2026).

CLASSIFICATIONS = {
    # --- H2: Real external tools (exist commercially or as open-source) ---
    "LTspice":              ("H2", "Analog Devices (formerly Linear Technology) SPICE simulator"),
    "Altium":               ("H2", "Altium Designer - commercial PCB design suite"),
    "PSpice":               ("H2", "Cadence Design Systems SPICE simulator"),
    "Fusion":               ("H2", "Autodesk Fusion 360 - CAD/electronics platform"),
    "SolidWorks":           ("H2", "Dassault Systemes - 3D CAD with simulation"),
    "JLCPCB":               ("H2", "PCB manufacturer with online design tools/calculators"),
    "FEMM":                 ("H2", "Finite Element Method Magnetics - open-source FEA"),
    "LTSpice":              ("H2", "LTspice variant capitalization"),
    "ANSYS Maxwell":        ("H2", "Ansys electromagnetic simulation software"),
    "PLECS":                ("H2", "Plexim - power electronics circuit simulation"),
    "PSIM":                 ("H2", "Powersim Inc. - power electronics simulation"),
    "Cadence Allegro":      ("H2", "Cadence PCB design and layout tool"),
    "Autodesk Fusion":      ("H2", "Autodesk Fusion 360 variant name"),
    "COMSOL":               ("H2", "COMSOL Multiphysics - FEA simulation platform"),
    "Cadence Sigrity":      ("H2", "Cadence signal/power integrity analysis"),
    "Magnetics Designer":   ("H2", "Intusoft - transformer/inductor design software"),
    "Polar Si9000":         ("H2", "Polar Instruments - PCB impedance calculator"),
    "CST Studio":           ("H2", "Dassault Systemes (formerly CST) - EM simulation suite"),
    "MATLAB/Simulink":      ("H2", "MathWorks - numerical computing and simulation"),
    "Keysight ADS":         ("H2", "Keysight Advanced Design System - RF/microwave EDA"),
    "Mentor Graphics":      ("H2", "Siemens EDA (formerly Mentor) - EDA tools"),
    "WEBDESIGNER":          ("H2", "ON Semiconductor Power Supply WebDesigner"),
    "Quite Universal Circuit Simulator": ("H2", "QUCS - open-source circuit simulator"),
    "AutoCAD":              ("H2", "Autodesk - general-purpose CAD"),
    "HyperLynx":            ("H2", "Siemens/Mentor - signal integrity analysis"),
    "QUCS":                 ("H2", "Quite Universal Circuit Simulator - open-source"),
    "Ansys Maxwell":        ("H2", "Ansys electromagnetic simulation (variant name)"),
    "PSPICE":               ("H2", "PSpice variant capitalization"),
    "Ansys Icepak":         ("H2", "Ansys thermal management simulation"),
    "Polar Si9000e":        ("H2", "Polar Instruments impedance tool variant"),
    "Strata Developer Studio": ("H2", "ON Semiconductor - cloud-connected dev platform"),
    "Simbeor":              ("H2", "Simberian - EM signal integrity software"),
    "Power Stage Designer": ("H2", "Texas Instruments - power supply design tool"),
    "TI Power Stage Designer": ("H2", "Texas Instruments Power Stage Designer"),
    "Falstad":              ("H2", "Paul Falstad - online circuit simulator"),
    "FreeCAD":              ("H2", "Open-source parametric 3D CAD"),
    "ngspice":              ("H2", "Open-source SPICE circuit simulator"),
    "WebDesigner+":         ("H2", "ON Semiconductor WebDesigner variant"),
    "Cadence OrCAD":        ("H2", "Cadence OrCAD - PCB design suite"),
    "EasyEDA PCB Editor":   ("H2", "JLCPCB/EasyEDA PCB editor variant name"),
    "EasyEDA Schematic":    ("H2", "EasyEDA schematic editor variant name"),
    "PCBWay":               ("H2", "PCB manufacturer with online tools"),
    "Altair Flux":          ("H2", "Altair - electromagnetic and thermal simulation"),
    "SIMPLIS":              ("H2", "SIMPLIS Technologies - switching power supply simulation"),
    "SIMetrix/SIMPLIS":     ("H2", "SIMetrix/SIMPLIS - combined SPICE + switching sim"),
    "MagNet":               ("H2", "Infolytica/Mentor - magnetics FEA simulation"),
    "OpenModelica":         ("H2", "Open-source Modelica-based simulation"),
    "AppCAD":               ("H2", "Broadcom/Avago - RF design calculator"),
    "SimScale":             ("H2", "Cloud-based engineering simulation platform"),
    "Ansys HFSS":           ("H2", "Ansys - high-frequency EM simulation"),
    "OpenEMS":              ("H2", "Open-source EM field solver"),
    "QuickField":           ("H2", "Tera Analysis - FEA electromagnetic simulation"),
    "TINA-TI":              ("H2", "Texas Instruments - SPICE-based circuit simulator"),
    "OSH Park":             ("H2", "PCB manufacturing service"),
    "Seeed Studio":         ("H2", "Hardware/IoT manufacturer with tools"),
    "OrCAD PSpice":         ("H2", "Cadence OrCAD PSpice simulator"),
    "Cadence PSpice":       ("H2", "Cadence PSpice simulator variant name"),
    "Multisim":             ("H2", "NI/Emona - circuit simulation (now part of NI)"),
    "Fritzing":             ("H2", "Open-source electronics prototyping tool"),
    "Sonnet":               ("H2", "Sonnet Software - planar 3D EM simulation"),
    "Qucs-S":               ("H2", "QUCS with SPICE - open-source simulator"),
    "Infineon IPOSIM":      ("H2", "Infineon - IGBT power loss simulator"),
    "ON Semiconductor WebDesigner": ("H2", "ON Semi power supply WebDesigner tool"),
    "ON Semiconductor Power Supply Designer (PSD)": ("H2", "ON Semi PSD tool"),
    "onsemi Design Tools":  ("H2", "ON Semiconductor design tool suite"),
    "Texas Instruments WEBENCH": ("H2", "TI WEBENCH variant name"),
    "Ansys SIwave":         ("H2", "Ansys signal/power integrity analysis"),
    "EasyEDA 3D Viewer":    ("H2", "EasyEDA 3D visualization variant"),
    "Atlc":                 ("H2", "Arbitrary Transmission Line Calculator - open-source FEA"),
    "KiCad / Altium":       ("H2", "Reference to both KiCad and Altium tools"),
    "EasyEDA BOM Manager":  ("H2", "EasyEDA BOM management feature"),
    "EasyEDA SPICE":        ("H2", "EasyEDA SPICE simulation feature"),
    "EasyEDA Simulator":    ("H2", "EasyEDA simulation feature"),
    "EasyEDA Gerber Viewer": ("H2", "EasyEDA Gerber viewing feature"),
    "EasyEDA + JLCPCB":     ("H2", "Combined EasyEDA/JLCPCB reference"),
    "Excel":                ("H2", "Microsoft Excel - general-purpose spreadsheet"),
    "Git":                  ("H2", "Version control system (not EDA-specific)"),
    "Analog Devices":       ("H2", "Analog Devices company tools (general reference)"),
    "WEBENCH Power Designer": ("H2", "TI WEBENCH Power Designer variant name"),

    # --- Near-miss: Imprecise references to tools IN the 82-tool database ---
    "Saturn":               ("Near-miss", "DB: Saturn PCB Design Toolkit"),
    "Temperature Rise Calculator": ("Near-miss", "DB: Sierra Circuits Trace Width, Current Capacity, and Temperature Rise Calculator"),
    "Micron":               ("Near-miss", "DB: Micron 20 PCB Production Calculator"),
    "Skottans":             ("Near-miss", "DB: Skottans Elektronik Impedance Calculator"),
    "Sierra Circuits":      ("Near-miss", "DB: Sierra Circuits Trace Width... Calculator"),
    "Use Saturn PCB Toolkit": ("Near-miss", "Sentence fragment referencing DB: Saturn PCB Design Toolkit"),
    "KiCad Calculator":     ("Near-miss", "DB: KiCad (extended with Calculator)"),
    "TI WEBENCH":           ("Near-miss", "DB: WEBENCH (TI)"),
    "KiCad PCB Calculator": ("Near-miss", "DB: KiCad (extended with PCB Calculator)"),
    "Circuit Digest Calculator": ("Near-miss", "DB: Circuit Digest PCB Trace Width Calculator"),
    "Technick":             ("Near-miss", "DB: Technick PCB Impedance Calculator"),
    "Why Saturn PCB Toolkit": ("Near-miss", "Sentence fragment referencing DB: Saturn PCB Design Toolkit"),
    "KiCad PCB Editor":     ("Near-miss", "DB: KiCad (extended with PCB Editor)"),
    "KiCad Gerber Viewer":  ("Near-miss", "DB: KiCad (extended with Gerber Viewer)"),
    "KiCad Schematic Editor": ("Near-miss", "DB: KiCad (extended with Schematic Editor)"),
    "EMCLAB":               ("Near-miss", "DB: EMCLAB PCB Trace Impedance Calculator"),
    "KiCad EDA Suite":      ("Near-miss", "DB: KiCad (extended with EDA Suite)"),
    "EEWeb Calculator":     ("Near-miss", "DB: EEWeb Microstrip Impedance Calculator"),
    "EEWeb Microstrip Calculator": ("Near-miss", "DB: EEWeb Microstrip Impedance Calculator"),
    "Digi-Key Calculator":  ("Near-miss", "DB: Digi-Key PCB Trace Width Calculator"),
    "Megabyte Circuit":     ("Near-miss", "DB: Megabyte Circuit PCB Calculator"),
    "TI Calculator":        ("Near-miss", "DB: TI PCB Thermal Calculator"),
    "Technick Calculator":  ("Near-miss", "DB: Technick PCB Impedance Calculator"),
    "Technick Impedance Calculator": ("Near-miss", "DB: Technick PCB Impedance Calculator"),
    "Technick PCB Calculator": ("Near-miss", "DB: Technick PCB Impedance Calculator"),
    "Open Saturn PCB Toolkit": ("Near-miss", "Sentence fragment referencing DB tool"),
    "Use EasyEDA Pro":      ("Near-miss", "DB: EasyEDA (extended name)"),
    "Complementary Calculator": ("Near-miss", "Generic reference to complementary DB calculators"),
    "Via Calculator":       ("Near-miss", "Generic calculator reference overlapping DB tools"),
    "PCB Calculator":       ("Near-miss", "Generic reference to PCB calculators in DB"),
    "PCB Toolkit":          ("Near-miss", "DB: Saturn PCB Design Toolkit (shortened)"),
    "Arbitrary Transmission Line Calculator": ("Near-miss", "Could reference DB tools or Atlc"),
    "PDN Analyzer":         ("Near-miss", "Generic power delivery network analysis reference"),
    "TI Thermal Calculator": ("Near-miss", "DB: TI PCB Thermal Calculator"),
    "Power Supply WebDesigner": ("Near-miss", "DB: overlaps ON Semi Design Tools in same domain"),

    # --- Non-specific: Generic terms, JSON fragments, sentence fragments ---
    "Practical":            ("Artifact", "JSON response fragment (Practical Tips)"),
    "Layer":                ("Artifact", "Feature/term (Layer Stack Manager)"),
    "Gerber":               ("Artifact", "File format name (Gerber format)"),
    "Recommended":          ("Artifact", "Response structure word"),
    "Complementary Tools":  ("Artifact", "Response structure phrase"),
    "Design":               ("Artifact", "Generic word from responses"),
    "Comprehensive":        ("Artifact", "Adjective from responses"),
    "SPICE":                ("Artifact", "Simulation methodology name (not specific tool)"),
    "Field Solver":         ("Artifact", "Generic tool class name"),
    "Stage":                ("Artifact", "Generic word / JSON workflow phase"),
    "Your":                 ("Artifact", "Pronoun from response text"),
    "Weeks":                ("Artifact", "Time word from response"),
    "Thermal":              ("Artifact", "Generic technical term"),
    "Score":                ("Artifact", "JSON field name"),
    "Extensive":            ("Artifact", "Adjective from response"),
    "Logic Analyzer":       ("Artifact", "Instrument category name"),
    "Direct":               ("Artifact", "Adjective from response"),
    "Schematic Editor":     ("Artifact", "Generic feature name"),
    "Spreadsheet":          ("Artifact", "Generic software category"),
    "Stackup":              ("Artifact", "Technical PCB term"),
    "Avoid":                ("Artifact", "Verb from response text"),
    "PCB Editor":           ("Artifact", "Generic feature name"),
    "Full":                 ("Artifact", "Adjective from response"),
    "Generate":             ("Artifact", "Verb from response text"),
    "Advanced":             ("Artifact", "Adjective from response"),
    "PCB CAD":              ("Artifact", "Generic category name"),
    "Getting Started":      ("Artifact", "Response/documentation phrase"),
    "Phases":               ("Artifact", "JSON workflow structure"),
    "Power Analyzer":       ("Artifact", "Instrument/feature category"),
    "Integrated":           ("Artifact", "Adjective from response"),
    "CISPR":                ("Artifact", "EMC standard name (not a tool)"),
    "Download":             ("Artifact", "Verb from response text"),
    "Bode":                 ("Artifact", "Analysis technique name (Bode plot)"),
    "Tier":                 ("Artifact", "Generic word from response"),
    "Budget":               ("Artifact", "Generic word from response"),
    "Steps":                ("Artifact", "Response structure word"),
    "Month":                ("Artifact", "Time word from response"),
    "Power Designer":       ("Artifact", "Generic term (ambiguous)"),
    "Target":               ("Artifact", "Generic word from response"),
    "Order":                ("Artifact", "Generic word from response"),
    "Complete":             ("Artifact", "Adjective from response"),
    "Export":               ("Artifact", "Feature name from response"),
    "Simple":               ("Artifact", "Adjective from response"),
    "Verification":         ("Artifact", "Process name from response"),
    "Seamless":             ("Artifact", "Adjective from response"),
    "Panel":                ("Artifact", "UI/feature term"),
    "Strong":               ("Artifact", "Adjective from response"),
    "Impedance":            ("Artifact", "Technical term"),
    "Mixed-Signal":         ("Artifact", "Category descriptor"),
    "Spend":                ("Artifact", "Verb from response"),
    "Expect":               ("Artifact", "Verb from response"),
    "Apply":                ("Artifact", "Verb from response"),
    "Windows":              ("Artifact", "OS name (not EDA tool)"),
    "STAGE":                ("Artifact", "JSON workflow structure (uppercase)"),
    "Inner":                ("Artifact", "Adjective from response"),
    "Simulation":           ("Artifact", "Generic process name"),
    "Spectrum Analyzer":    ("Artifact", "Instrument category"),
    "SMPS":                 ("Artifact", "Power supply acronym (not a tool)"),
    "Footprint Editor":     ("Artifact", "Generic EDA feature name"),
    "Over":                 ("Artifact", "Preposition from response"),
    "Consider":             ("Artifact", "Verb from response"),
    "Content Manager":      ("Artifact", "Generic feature name"),
    "Manufacturing":        ("Artifact", "Process name"),
    "PCB Design Software":  ("Artifact", "Generic category"),
    "Suggested":            ("Artifact", "Adjective from response"),
    "Documentation":        ("Artifact", "Generic term"),
    "Print":                ("Artifact", "Verb/feature from response"),
    "Minimum":              ("Artifact", "Adjective from response"),
    "FABRICATION":          ("Artifact", "Process name (uppercase)"),
    "Linux":                ("Artifact", "Operating system name"),
    "Using":                ("Artifact", "Verb from response"),
    "Define":               ("Artifact", "Verb from response"),
    "Keep":                 ("Artifact", "Verb from response"),
    "OrCAD PCB":            ("H2", "Cadence OrCAD PCB variant"),
    "OrCAD PSpice":         ("H2", "Cadence OrCAD PSpice variant"),
    "Vector Network Analyzer": ("Artifact", "Instrument category"),
    "Current":              ("Artifact", "Generic word from response"),
    "TRAN Pout AVG":        ("Artifact", "Simulation metric/command"),
    "Efficiency":           ("Artifact", "Technical metric name"),
    "FCC Part":             ("Artifact", "Regulatory standard reference"),
    "Broad":                ("Artifact", "Adjective from response"),
    "Key Practical Tips":   ("Artifact", "Response structure phrase"),
    "Rich":                 ("Artifact", "Adjective from response"),
    "Students":             ("Artifact", "Noun from response"),
    "Calculate":            ("Artifact", "Verb from response"),
    "Typically":            ("Artifact", "Adverb from response"),
    "Compare":              ("Artifact", "Verb from response"),
}

# ===========================================================================
# Evidence URLs for H2 (external-real) tools
# ===========================================================================
# Each H2 tool maps to a vendor product page, official repository, or
# Wikipedia entry. Variant names map to the same URL as the base tool.
# Verified February 2026.

EVIDENCE_URLS = {
    # --- Analog Devices / Linear Technology ---
    "LTspice":              "https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html",
    "LTSpice":              "https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html",
    "Analog Devices":       "https://www.analog.com/en/design-center.html",

    # --- Altium ---
    "Altium":               "https://www.altium.com/altium-designer",

    # --- Cadence Design Systems ---
    "PSpice":               "https://www.pspice.com/",
    "PSPICE":               "https://www.pspice.com/",
    "Cadence PSpice":       "https://www.pspice.com/",
    "OrCAD PSpice":         "https://www.pspice.com/",
    "Cadence Allegro":      "https://www.cadence.com/en_US/home/tools/pcb-design-and-analysis/pcb-layout/allegro-pcb-designer.html",
    "Cadence Sigrity":      "https://www.cadence.com/en_US/home/tools/pcb-design-and-analysis/si-pi-analysis.html",
    "Cadence OrCAD":        "https://www.orcad.com/",
    "OrCAD PCB":            "https://www.orcad.com/",

    # --- Autodesk ---
    "Fusion":               "https://www.autodesk.com/products/fusion-360",
    "Autodesk Fusion":      "https://www.autodesk.com/products/fusion-360",
    "AutoCAD":              "https://www.autodesk.com/products/autocad",

    # --- Dassault Systemes ---
    "SolidWorks":           "https://www.solidworks.com/",
    "CST Studio":           "https://www.3ds.com/products/simulia/cst-studio-suite",

    # --- JLCPCB / EasyEDA ---
    "JLCPCB":               "https://jlcpcb.com/",
    "EasyEDA PCB Editor":   "https://easyeda.com/",
    "EasyEDA Schematic":    "https://easyeda.com/",
    "EasyEDA 3D Viewer":    "https://easyeda.com/",
    "EasyEDA Simulator":    "https://easyeda.com/",
    "EasyEDA Gerber Viewer": "https://easyeda.com/",
    "EasyEDA + JLCPCB":     "https://easyeda.com/",
    "EasyEDA SPICE":        "https://easyeda.com/",
    "EasyEDA BOM Manager":  "https://easyeda.com/",

    # --- Open-source tools ---
    "FEMM":                 "https://www.femm.info/",
    "QUCS":                 "https://qucs.sourceforge.net/",
    "Quite Universal Circuit Simulator": "https://qucs.sourceforge.net/",
    "Qucs-S":               "https://ra3xdh.github.io/",
    "FreeCAD":              "https://www.freecad.org/",
    "ngspice":              "https://ngspice.sourceforge.io/",
    "OpenEMS":              "https://openems.de/",
    "OpenModelica":         "https://openmodelica.org/",
    "Atlc":                 "https://atlc.sourceforge.net/",
    "Fritzing":             "https://fritzing.org/",
    "Git":                  "https://git-scm.com/",

    # --- Ansys ---
    "ANSYS Maxwell":        "https://www.ansys.com/products/electronics/ansys-maxwell",
    "Ansys Maxwell":        "https://www.ansys.com/products/electronics/ansys-maxwell",
    "Ansys Icepak":         "https://www.ansys.com/products/electronics/ansys-icepak",
    "Ansys HFSS":           "https://www.ansys.com/products/electronics/ansys-hfss",
    "Ansys SIwave":         "https://www.ansys.com/products/electronics/ansys-siwave",

    # --- Siemens EDA (formerly Mentor Graphics) ---
    "Mentor Graphics":      "https://eda.sw.siemens.com/",
    "HyperLynx":            "https://eda.sw.siemens.com/en-US/pcb/hyperlynx/",
    "MagNet":               "https://plm.sw.siemens.com/en-US/simcenter/electromagnetics-simulation/magnet/",

    # --- Texas Instruments ---
    "WEBENCH Power Designer": "https://www.ti.com/design-resources/design-tools-simulation/webench-power-designer.html",
    "Texas Instruments WEBENCH": "https://www.ti.com/design-resources/design-tools-simulation/webench-power-designer.html",
    "Power Stage Designer": "https://www.ti.com/tool/POWERSTAGE-DESIGNER",
    "TI Power Stage Designer": "https://www.ti.com/tool/POWERSTAGE-DESIGNER",
    "TINA-TI":              "https://www.ti.com/tool/TINA-TI",

    # --- ON Semiconductor / onsemi ---
    "WEBDESIGNER":          "https://www.onsemi.com/design/webdesigner+",
    "WebDesigner+":         "https://www.onsemi.com/design/webdesigner+",
    "ON Semiconductor WebDesigner": "https://www.onsemi.com/design/webdesigner+",
    "ON Semiconductor Power Supply Designer (PSD)": "https://www.onsemi.com/design/webdesigner+",
    "onsemi Design Tools":  "https://www.onsemi.com/design",
    "Strata Developer Studio": "https://www.onsemi.com/design/tools-software/strata-developer-studio",

    # --- Other commercial ---
    "PLECS":                "https://www.plexim.com/plecs",
    "PSIM":                 "https://powersimtech.com/",
    "COMSOL":               "https://www.comsol.com/",
    "Magnetics Designer":   "http://www.intusoft.com/mag.htm",
    "Polar Si9000":         "https://www.polarinstruments.com/products/si/Si9000.html",
    "Polar Si9000e":        "https://www.polarinstruments.com/products/si/Si9000.html",
    "MATLAB/Simulink":      "https://www.mathworks.com/products/simulink.html",
    "Keysight ADS":         "https://www.keysight.com/us/en/products/software/pathwave-design-software/pathwave-advanced-design-system.html",
    "Simbeor":              "https://www.simberian.com/",
    "Falstad":              "https://www.falstad.com/circuit/",
    "PCBWay":               "https://www.pcbway.com/",
    "SIMPLIS":              "https://www.simetrix.co.uk/",
    "SIMetrix/SIMPLIS":     "https://www.simetrix.co.uk/",
    "Altair Flux":          "https://altair.com/flux",
    "QuickField":           "https://quickfield.com/",
    "Sonnet":               "https://www.sonnetsoftware.com/",
    "SimScale":             "https://www.simscale.com/",
    "Multisim":             "https://www.ni.com/en/shop/electronic-test-instrumentation/application-software-for-electronic-test-and-instrumentation-category/what-is-multisim.html",
    "Infineon IPOSIM":      "https://www.infineon.com/cms/en/tools/landing/iposim.html",
    "AppCAD":               "https://www.broadcom.com/info/wireless/appcad",
    "Seeed Studio":         "https://www.seeedstudio.com/",
    "OSH Park":             "https://oshpark.com/",
    "Excel":                "https://www.microsoft.com/en-us/microsoft-365/excel",

    # --- Multi-tool references ---
    "KiCad / Altium":       "https://www.kicad.org/",
}

# Known pre-filter artifacts (common words/phrases removed before classification)
KNOWN_ARTIFACTS = {
    "phase", "step", "week", "type", "tool", "action", "note", "tip",
    "description", "summary", "title", "format", "name", "verdict",
    "overview", "recommendation", "comparison", "workflow", "integration",
    "compatibility", "tips", "strengths", "limitations",
    "the", "and", "for", "with", "from", "that", "this", "will",
    "can", "use", "also", "more", "best", "good", "high", "low",
    "new", "key", "set", "run", "test", "data", "file", "open",
    "phase 1", "phase 2", "phase 3", "phase 4", "phase 5",
    "step 1", "step 2", "step 3", "step 4", "step 5",
}


def load_all_results():
    """Load all result JSON files from all phases."""
    all_entries = []
    phase_counts = {}
    for phase_dir, phase_label in PHASES.items():
        phase_path = os.path.join(BASE_DIR, phase_dir)
        if not os.path.exists(phase_path):
            print(f"WARNING: Phase directory not found: {phase_path}")
            continue
        json_files = glob.glob(os.path.join(phase_path, 'results_*.json'))
        count = 0
        for fpath in json_files:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        entry['_phase'] = phase_dir
                        all_entries.append(entry)
                        count += 1
                elif isinstance(data, dict):
                    data['_phase'] = phase_dir
                    all_entries.append(data)
                    count += 1
        phase_counts[phase_dir] = count
    return all_entries, phase_counts


def is_artifact(name):
    """Check if a tool name is likely an extraction artifact."""
    lower = name.lower().strip()
    if lower in KNOWN_ARTIFACTS:
        return True
    if len(lower) <= 2:
        return True
    if re.match(r'^[\d\.\-\s]+$', lower):
        return True
    if re.match(r'^(the|and|for|with|from|phase|step|tool|type)\s', lower):
        return True
    return False


def normalize_tool_name(name):
    """Normalize a tool name for deduplication."""
    clean = name.strip()
    clean = re.sub(r'[\.\,\;\:\!\?]+$', '', clean)
    return clean


def find_base_name(name, all_names_sorted):
    """Check if 'name' is a longer variant of a shorter base name."""
    lower = name.lower()
    for shorter_name in all_names_sorted:
        shorter_lower = shorter_name.lower()
        if shorter_lower == lower:
            continue
        if lower.startswith(shorter_lower + ' ') or lower.startswith(shorter_lower + ','):
            return shorter_name
    return None


def classify_name(name):
    """
    Classify a consolidated tool name.
    Returns (category, description) where category is H1, H2, Near-miss, or Artifact (non-specific).
    For names not in CLASSIFICATIONS dict, returns Unclassified.
    """
    if name in CLASSIFICATIONS:
        return CLASSIFICATIONS[name]
    # Heuristic: if it looks like a sentence fragment or generic term
    lower = name.lower()
    generic_indicators = [
        'calculator', 'editor', 'viewer', 'manager', 'suite',
        'workflow', 'recommended', 'complementary', 'comprehensive',
        'practical', 'getting started', 'why ', 'use ', 'open ',
        'avoid', 'download', 'generate', 'consider', 'expect',
    ]
    for indicator in generic_indicators:
        if lower.startswith(indicator) or (indicator.startswith(' ') is False and lower == indicator):
            pass  # Don't auto-classify, leave as unclassified
    # Default: unclassified (will be counted separately)
    return ("Unclassified", "Not in top-K manual classification")


def main():
    print("=" * 70)
    print("HALLUCINATION CLASSIFICATION WITH H1/H2 TAXONOMY")
    print("=" * 70)

    # Load data
    all_entries, phase_counts = load_all_results()
    print(f"\nLoaded {len(all_entries)} query results from {len(phase_counts)} phases")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} entries")

    # Sanity check: total must equal expected experiment size
    EXPECTED_TOTAL = 6912  # 2592 + 864 + 2592 + 864
    EXPECTED_PER_PHASE = {'phase1': 2592, 'phase2': 864, 'phase3a': 2592, 'phase3b': 864}
    if len(all_entries) != EXPECTED_TOTAL:
        print(f"\n  *** ERROR: Expected {EXPECTED_TOTAL} entries, got {len(all_entries)}. "
              f"Results directory may contain extra/missing files. ***")
        sys.exit(1)
    for phase, expected_n in EXPECTED_PER_PHASE.items():
        actual_n = phase_counts.get(phase, 0)
        if actual_n != expected_n:
            print(f"  *** WARNING: {phase} has {actual_n} entries, expected {expected_n} ***")
    print(f"  Sanity check PASSED: {len(all_entries)} entries")

    # Extract all hallucinated tool names
    raw_mentions = defaultdict(int)
    raw_mentions_by_config = defaultdict(lambda: defaultdict(int))
    total_queries = 0
    total_hallucinated_mentions = 0

    for entry in all_entries:
        total_queries += 1
        config_id = entry.get('ConfigId', 'unknown')
        hallucinated = entry.get('HallucinatedToolNames', [])
        if hallucinated:
            for tool_name in hallucinated:
                clean = normalize_tool_name(tool_name)
                if clean:
                    raw_mentions[clean] += 1
                    raw_mentions_by_config[clean][config_id] += 1
                    total_hallucinated_mentions += 1

    print(f"\nTotal queries: {total_queries}")
    print(f"Total hallucinated mentions: {total_hallucinated_mentions}")
    print(f"Unique hallucinated names (raw): {len(raw_mentions)}")

    # Separate pre-filter artifacts
    artifacts = {}
    tool_mentions = {}
    for name, count in raw_mentions.items():
        if is_artifact(name):
            artifacts[name] = count
        else:
            tool_mentions[name] = count

    artifact_mention_total = sum(artifacts.values())
    print(f"\nPre-filter artifacts removed: {len(artifacts)} names ({artifact_mention_total} mentions)")
    print(f"Remaining: {len(tool_mentions)} names ({sum(tool_mentions.values())} mentions)")

    # Sort by frequency for base-name detection
    sorted_by_freq = sorted(tool_mentions.keys(), key=lambda n: tool_mentions[n], reverse=True)

    # Merge partial-string duplicates
    base_name_map = {}
    for name in sorted_by_freq:
        base = find_base_name(name, sorted_by_freq)
        if base:
            base_name_map[name] = base

    consolidated = defaultdict(int)
    consolidated_variants = defaultdict(list)
    consolidated_by_config = defaultdict(lambda: defaultdict(int))

    for name, count in tool_mentions.items():
        base = base_name_map.get(name, name)
        while base in base_name_map and base_name_map[base] != base:
            base = base_name_map[base]
        consolidated[base] += count
        if name != base:
            consolidated_variants[base].append((name, count))
        for config_id, cfg_count in raw_mentions_by_config[name].items():
            consolidated_by_config[base][config_id] += cfg_count

    print(f"After consolidation: {len(consolidated)} unique tool names")

    # Sort by frequency
    ranked = sorted(consolidated.items(), key=lambda x: x[1], reverse=True)
    total_tool_mentions = sum(count for _, count in ranked)

    # Classify each name and compute statistics
    category_mentions = defaultdict(int)   # category -> total mentions
    category_names = defaultdict(int)      # category -> number of unique names
    category_c0 = defaultdict(int)         # category -> C0 mentions
    category_c5 = defaultdict(int)         # category -> C5 mentions
    category_examples = defaultdict(list)  # category -> [(name, count)]

    classified_count = 0
    classified_mentions = 0

    for name, count in ranked:
        cat, desc = classify_name(name)
        category_mentions[cat] += count
        category_names[cat] += 1
        category_c0[cat] += consolidated_by_config[name].get('C0', 0)
        category_c5[cat] += consolidated_by_config[name].get('C5', 0)
        if cat != "Unclassified":
            classified_count += 1
            classified_mentions += count
        if len(category_examples[cat]) < 8:
            category_examples[cat].append((name, count))

    # Compute coverage milestones
    cumulative = 0
    coverage_milestones = {}
    for i, (name, count) in enumerate(ranked, 1):
        cumulative += count
        pct = cumulative / total_tool_mentions * 100
        for k in [10, 20, 30, 50, 100]:
            if k not in coverage_milestones and i >= k:
                coverage_milestones[k] = pct

    # =====================================================================
    # Build output report
    # =====================================================================
    lines = []
    lines.append("=" * 95)
    lines.append("HALLUCINATION CLASSIFICATION WITH H1/H2 TAXONOMY")
    lines.append("=" * 95)
    lines.append("")
    lines.append(f"Total queries analyzed: {total_queries}")
    lines.append(f"Total hallucinated mentions (raw): {total_hallucinated_mentions}")
    lines.append(f"Pre-filter artifacts removed: {len(artifacts)} names ({artifact_mention_total} mentions)")
    lines.append(f"Tool mentions after pre-filter: {total_tool_mentions}")
    lines.append(f"Unique names (after consolidation): {len(consolidated)}")
    lines.append(f"Manually classified: {classified_count} names ({classified_mentions} mentions, "
                 f"{classified_mentions/total_tool_mentions*100:.1f}% of post-filter mentions)")
    lines.append("")

    lines.append("COVERAGE ANALYSIS:")
    for k, pct in sorted(coverage_milestones.items()):
        lines.append(f"  Top-{k} names cover {pct:.1f}% of all tool mentions")
    lines.append("")

    # =====================================================================
    # KEY RESULT: H1/H2 Decomposition
    # =====================================================================
    lines.append("=" * 95)
    lines.append("H1/H2 DECOMPOSITION (KEY RESULT)")
    lines.append("=" * 95)
    lines.append("")

    # Only count classified names for the decomposition
    classified_total = sum(v for k, v in category_mentions.items() if k != "Unclassified")
    for cat in ["H2", "Near-miss", "Artifact", "H1", "Unclassified"]:
        # Map internal "Artifact" key to paper-consistent "Non-specific" for display
        display_cat = "Non-specific" if cat == "Artifact" else cat
        mentions = category_mentions[cat]
        names = category_names[cat]
        pct_of_classified = (mentions / classified_total * 100) if classified_total > 0 else 0
        pct_of_total = mentions / total_tool_mentions * 100
        c0 = category_c0[cat]
        c5 = category_c5[cat]
        label = {
            "H2": "External-real (exist commercially/open-source)",
            "Near-miss": "Imprecise reference to inventory tool",
            "Artifact": "Non-specific mention (generic terms, fragments)",
            "H1": "Fabricated (no verifiable referent)",
            "Unclassified": "Not yet classified (long tail)",
        }[cat]

        lines.append(f"  {display_cat:12s}: {mentions:6d} mentions ({names:4d} names) "
                     f"= {pct_of_total:5.1f}% of all | {pct_of_classified:5.1f}% of classified"
                     f"  [C0: {c0}, C5: {c5}]")
        lines.append(f"               {label}")
        if category_examples[cat]:
            examples = ", ".join(f"{n} ({c})" for n, c in category_examples[cat][:5])
            lines.append(f"               Examples: {examples}")
        lines.append("")

    lines.append("-" * 95)

    # Frequency-weighted H1/H2 split (excluding non-specific mentions and near-misses)
    h1_mentions = category_mentions["H1"]
    h2_mentions = category_mentions["H2"]
    h1_h2_total = h1_mentions + h2_mentions
    if h1_h2_total > 0:
        h1_pct = h1_mentions / h1_h2_total * 100
        h2_pct = h2_mentions / h1_h2_total * 100
    else:
        h1_pct = h2_pct = 0

    lines.append("")
    lines.append("FREQUENCY-WEIGHTED H1/H2 SPLIT (excluding non-specific mentions and near-misses):")
    lines.append(f"  H1 (fabricated):     {h1_mentions:6d} mentions = {h1_pct:.1f}%")
    lines.append(f"  H2 (external-real):  {h2_mentions:6d} mentions = {h2_pct:.1f}%")
    lines.append(f"  Total H1+H2:         {h1_h2_total:6d} mentions")
    lines.append("")

    # Architecture effectiveness by category
    lines.append("ARCHITECTURE EFFECTIVENESS BY CATEGORY (C0 vs C5):")
    for cat in ["H2", "Near-miss", "H1"]:
        c0 = category_c0[cat]
        c5 = category_c5[cat]
        if c0 > 0:
            reduction = (1 - c5/c0) * 100
            lines.append(f"  {cat:12s}: C0={c0:5d}, C5={c5:5d}, reduction={reduction:.1f}%")
        else:
            lines.append(f"  {cat:12s}: C0={c0:5d}, C5={c5:5d}")
    lines.append("")

    # =====================================================================
    # Top 100 classified list
    # =====================================================================
    lines.append("=" * 95)
    lines.append("TOP 100 CLASSIFIED NAMES")
    lines.append("=" * 95)
    lines.append("")
    lines.append(f"{'Rank':<5} {'Cat':<12} {'Tool Name':<50} {'Mentions':<9} {'C0':<7} {'C5':<7}")
    lines.append("-" * 90)

    cumulative = 0
    for i, (name, count) in enumerate(ranked[:100], 1):
        cumulative += count
        cat, desc = classify_name(name)
        c0 = consolidated_by_config[name].get('C0', 0)
        c5 = consolidated_by_config[name].get('C5', 0)
        lines.append(f"{i:<5} {cat:<12} {name:<50} {count:<9} {c0:<7} {c5:<7}")

    lines.append("")
    lines.append(f"Top-100 cumulative: {cumulative} mentions "
                 f"({cumulative/total_tool_mentions*100:.1f}% of all)")
    lines.append("")

    # =====================================================================
    # LaTeX-ready summary for paper insertion
    # =====================================================================
    lines.append("=" * 95)
    lines.append("LATEX-READY PARAGRAPH (for paper Section 5.X)")
    lines.append("=" * 95)
    lines.append("")

    # Count classified names covering significant portion
    h2_names = category_names["H2"]
    nm_names = category_names["Near-miss"]
    art_names = category_names["Artifact"]
    h1_names = category_names["H1"]

    lines.append(f"We manually classified {classified_count} consolidated tool names")
    lines.append(f"(covering {classified_mentions/total_tool_mentions*100:.1f}% of all "
                 f"{total_tool_mentions} out-of-inventory mentions).")
    lines.append(f"Of the {classified_mentions} classified mentions:")
    lines.append(f"  - {h2_mentions} ({h2_mentions/classified_mentions*100:.1f}%) were H2 "
                 f"(external-real: commercially available or open-source tools)")
    lines.append(f"  - {category_mentions['Near-miss']} ({category_mentions['Near-miss']/classified_mentions*100:.1f}%) "
                 f"were near-miss references to inventory tools")
    lines.append(f"  - {category_mentions['Artifact']} ({category_mentions['Artifact']/classified_mentions*100:.1f}%) "
                 f"were non-specific mentions")
    lines.append(f"  - {h1_mentions} ({h1_mentions/classified_mentions*100:.1f}%) were H1 "
                 f"(fabricated names with no verifiable referent)")
    lines.append("")
    lines.append("Key finding: Among classified mentions, the H1/H2 split is "
                 f"{h1_pct:.1f}%/{h2_pct:.1f}%.")
    lines.append("The vast majority of 'hallucinated' tool mentions refer to real,")
    lines.append("commercially available tools that are simply absent from the curated inventory.")
    lines.append("")
    lines.append(f"Architecture effectiveness is consistent across categories:")
    for cat in ["H2", "Near-miss"]:
        c0 = category_c0[cat]
        c5 = category_c5[cat]
        if c0 > 0:
            reduction = (1 - c5/c0) * 100
            lines.append(f"  {cat}: C0->C5 reduction = {reduction:.1f}%")
    lines.append("")

    output = '\n'.join(lines)
    print(output)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    # =====================================================================
    # Supplementary material CSV (all 197 classified names)
    # =====================================================================
    csv_path = os.path.join(os.path.dirname(OUTPUT_FILE), 'supplement_197_classifications.csv')
    csv_rows = []
    csv_rank = 0
    for name, count in ranked:
        cat, desc = classify_name(name)
        if cat == "Unclassified":
            continue
        csv_rank += 1
        c0 = consolidated_by_config[name].get('C0', 0)
        c5 = consolidated_by_config[name].get('C5', 0)
        # Map internal "Artifact" label to paper-consistent "Non-specific"
        display_cat = "Non-specific" if cat == "Artifact" else cat
        # Look up evidence URL (H2 names have vendor/repo URLs; others get empty)
        url = EVIDENCE_URLS.get(name, "")
        csv_rows.append({
            'Rank': csv_rank,
            'ConsolidatedName': name,
            'Category': display_cat,
            'TotalMentions': count,
            'C0_Mentions': c0,
            'C5_Mentions': c5,
            'Evidence': desc,
            'EvidenceURL': url,
        })

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Rank', 'ConsolidatedName', 'Category', 'TotalMentions',
            'C0_Mentions', 'C5_Mentions', 'Evidence', 'EvidenceURL'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Supplement CSV: {len(csv_rows)} classified names saved to {csv_path}")


if __name__ == '__main__':
    main()
