import sys
import os
import subprocess
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import multiprocessing
from multiprocessing import Pool, cpu_count, current_process
import time
import re
from rapidfuzz import process, fuzz

# GUI dependencies
from PySide6.QtCore import QObject, Signal, QThread, Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, QLabel,
                               QLineEdit, QPushButton, QTextEdit, QFileDialog,
                               QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
                               QGroupBox, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, 
                               QComboBox, QListWidget, QListWidgetItem)

# --- Core Utility & Multiprocessing Functions (Unchanged) ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower().replace('_', ' ').replace('#', ' ').replace('-', ' ')
    text = re.sub(r'\s\d{4,}', '', text)
    company_suffixes_end_of_string = ['inc', 'corp', 'co', 'ltd', 'p.c.', 'pc', 'llc', 'l.l.c.', 'l.c.', 'lc', 'ltd. co.', 'ltd co', 'lp', 'l.p.', 'llp', 'l.l.l.p.', 'pllc', 'b.v.', 'gmbh', 'enterprises', 'industries', 'solutions', 'ventures', 'holdings', 'group', 'services', 'technology', 'systems', 'inc.', 'corp.', 'co.', 'ltd.', 'pc.', 'llc.', 'l.l.c.', 'l.c.', 'lc.', 'lp.', 'l.p.', 'llp.', 'p.l.l.c.', '& co.']
    suffixes_end_pattern = r'\s*(' + '|'.join(re.escape(s) for s in company_suffixes_end_of_string) + r')\s*$'
    text = re.sub(suffixes_end_pattern, '', text, flags=re.IGNORECASE).strip()
    abbreviations = { r'\btech\b': 'technology', r'\bsvcs?\b': 'services', r'\bsolns?\b': 'solutions', r'\bmktg\b': 'marketing', r'\bmgmt\b': 'management', r'\bdev\b': 'development', r'\bintl\b': 'international', r'\bcorp\b': 'corporation', r'\binc\b': 'incorporated'}
    for abbr, full in abbreviations.items(): text = re.sub(abbr, full, text)
    common_tlds = ['com', 'net', 'org', 'io', 'co', 'store', 'shop', 'online', 'website']
    tld_pattern = r'\.(' + '|'.join(re.escape(tld) for tld in common_tlds) + r')\b\s*$'
    text = re.sub(tld_pattern, '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.split()), set(text2.split())
    if not set1 and not set2: return 1.0 
    if not set1 or not set2: return 0.0 
    return len(set1.intersection(set2)) / len(set1.union(set2))
def ultimate_validation(text1, text2, jaccard_threshold):
    clean1, clean2 = clean_text(text1), clean_text(text2)
    if not clean1 or not clean2: return False
    if jaccard_similarity(clean1, clean2) >= jaccard_threshold: return True
    if fuzz.WRatio(clean1, clean2) >= 95: return True
    words1, words2 = set(clean1.split()), set(clean2.split())
    if min(len(words1), len(words2)) > 1 and (words1.issubset(words2) or words2.issubset(words1)):
        if fuzz.WRatio(clean1, clean2) >= 88: return True
    short_str, long_str = (clean1, clean2) if len(clean1) < len(clean2) else (clean2, clean1)
    if long_str.startswith(short_str) and fuzz.WRatio(short_str, long_str) >= 90: return True
    return False
def process_fuzzy_market_chunk(args):
    market, source_chunk, master_names, threshold = args
    matches = {}
    JACCARD_THRESHOLD = 0.67 
    for source_name in source_chunk:
        best_match, score, _ = process.extractOne(source_name, master_names, scorer=fuzz.WRatio)
        if score >= threshold and jaccard_similarity(clean_text(source_name), clean_text(best_match)) >= JACCARD_THRESHOLD:
            matches[source_name] = {'master': best_match, 'method': 'Fuzzy'}
    return market, matches
RETRIEVER_MODEL = None
CROSS_ENCODER = None
def pool_initializer_semantic(model_name, cross_encoder_model_name):
    global RETRIEVER_MODEL, CROSS_ENCODER
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if RETRIEVER_MODEL is None: RETRIEVER_MODEL = SentenceTransformer(model_name, device=device)
    if CROSS_ENCODER is None: CROSS_ENCODER = CrossEncoder(cross_encoder_model_name, device=device)
def process_semantic_market_chunk(args):
    global RETRIEVER_MODEL, CROSS_ENCODER
    market, source_items, master_items, threshold, k_candidates, semantic_weight, fuzzy_weight, jaccard_threshold = args
    matches, market_logs = {}, []
    source_items = [item for item in source_items if isinstance(item, str) and item]
    master_items = [item for item in master_items if isinstance(item, str) and item]
    if not source_items or not master_items: return market, matches, []
    original_source_items = np.array(source_items)
    original_master_items = np.array(master_items)
    source_items_cleaned = [clean_text(s) for s in original_source_items]
    master_items_cleaned = [clean_text(m) for m in original_master_items]
    device_type = RETRIEVER_MODEL.device.type
    with torch.autocast(device_type=device_type):
        master_vectors = RETRIEVER_MODEL.encode(master_items_cleaned, batch_size=256, convert_to_tensor=False).astype('float32')
        source_vectors = RETRIEVER_MODEL.encode(source_items_cleaned, batch_size=256, convert_to_tensor=False).astype('float32')
    if master_vectors.shape[0] == 0 or source_vectors.shape[0] == 0: return market, matches, market_logs
    faiss.normalize_L2(master_vectors); faiss.normalize_L2(source_vectors)
    index = faiss.IndexFlatIP(master_vectors.shape[1]); index.add(master_vectors)
    K = min(k_candidates, len(master_items))
    _, indices = index.search(source_vectors, K)
    all_pairs_to_rerank, rerank_info = [], []
    for i, source_item_cleaned in enumerate(source_items_cleaned):
        original_candidate_names = [original_master_items[idx] for idx in indices[i] if idx != -1]
        if not original_candidate_names: continue
        rerank_info.append({'original_index': i, 'candidate_indices': indices[i]})
        for name in [clean_text(name) for name in original_candidate_names]:
            all_pairs_to_rerank.append([source_item_cleaned, name])
    if all_pairs_to_rerank:
        with torch.autocast(device_type=device_type):
            all_scores = CROSS_ENCODER.predict(all_pairs_to_rerank, show_progress_bar=False)
        score_offset = 0
        for info in rerank_info:
            candidate_indices = [idx for idx in info['candidate_indices'] if idx != -1]
            num_candidates = len(candidate_indices)
            if num_candidates == 0: continue
            semantic_scores = all_scores[score_offset : score_offset + num_candidates]
            original_source_name = original_source_items[info['original_index']]
            original_candidate_names = [original_master_items[idx] for idx in candidate_indices]
            fuzzy_scores = np.array([fuzz.WRatio(original_source_name, c) / 100.0 for c in original_candidate_names])
            hybrid_scores = (semantic_weight * semantic_scores) + (fuzzy_weight * fuzzy_scores)
            valid_mask = np.array([ultimate_validation(original_source_name, c, jaccard_threshold) for c in original_candidate_names])
            if np.any(valid_mask):
                valid_hybrid_scores = hybrid_scores[valid_mask]
                if np.max(valid_hybrid_scores) > threshold:
                    best_score_index = np.where(valid_mask)[0][np.argmax(valid_hybrid_scores)]
                    final_match_index = candidate_indices[best_score_index]
                    matches[original_source_name] = {'master': original_master_items[final_match_index], 'method': 'Semantic'}
            score_offset += num_candidates
    return market, matches, market_logs


# --- GUI Components (Unchanged) ---
class SavePromptDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Processing Complete!")
        self.setModal(True); self.result = None; self.format = None
        main_label = QLabel("Processing Complete. Select files to save.")
        main_label.setAlignment(Qt.AlignCenter)
        self.save_excel_group = QGroupBox("Save as Excel (.xlsx)")
        excel_layout = QVBoxLayout()
        self.save_main_excel_btn = QPushButton("Save Matched Source Data")
        self.save_summary_excel_btn = QPushButton("Save Summary Report")
        excel_layout.addWidget(self.save_main_excel_btn); excel_layout.addWidget(self.save_summary_excel_btn)
        self.save_excel_group.setLayout(excel_layout)
        self.save_csv_group = QGroupBox("Save as CSV (.csv)")
        csv_layout = QVBoxLayout()
        self.save_main_csv_btn = QPushButton("Save Matched Source Data")
        self.save_summary_csv_btn = QPushButton("Save Summary Report")
        csv_layout.addWidget(self.save_main_csv_btn); csv_layout.addWidget(self.save_summary_csv_btn)
        self.save_csv_group.setLayout(csv_layout)
        self.save_all_group = QGroupBox("Save All")
        save_all_layout = QHBoxLayout()
        self.save_all_excel_btn = QPushButton("Save All as Excel")
        self.save_all_csv_btn = QPushButton("Save All as CSV")
        save_all_layout.addWidget(self.save_all_excel_btn); save_all_layout.addWidget(self.save_all_csv_btn)
        self.save_all_group.setLayout(save_all_layout)
        self.open_folder_btn = QPushButton("Open Output Folder")
        self.run_again_btn = QPushButton("Run Again")
        self.quit_btn = QPushButton("Quit Application")
        self.save_main_excel_btn.clicked.connect(lambda: self.set_result("main", "xlsx"))
        self.save_summary_excel_btn.clicked.connect(lambda: self.set_result("summary_report", "xlsx"))
        self.save_main_csv_btn.clicked.connect(lambda: self.set_result("main", "csv"))
        self.save_summary_csv_btn.clicked.connect(lambda: self.set_result("summary_report", "csv"))
        self.save_all_excel_btn.clicked.connect(lambda: self.set_result("save_all", "xlsx"))
        self.save_all_csv_btn.clicked.connect(lambda: self.set_result("save_all", "csv"))
        self.open_folder_btn.clicked.connect(lambda: self.set_result("open_folder"))
        self.run_again_btn.clicked.connect(lambda: self.set_result("run_again")); self.run_again_btn.clicked.connect(self.accept)
        self.quit_btn.clicked.connect(lambda: self.set_result("quit")); self.quit_btn.clicked.connect(self.accept)
        format_layout = QHBoxLayout(); format_layout.addWidget(self.save_excel_group); format_layout.addWidget(self.save_csv_group)
        action_layout = QHBoxLayout(); action_layout.addWidget(self.open_folder_btn); action_layout.addWidget(self.run_again_btn); action_layout.addWidget(self.quit_btn)
        main_layout = QVBoxLayout(self); main_layout.addWidget(main_label); main_layout.addLayout(format_layout); main_layout.addWidget(self.save_all_group); main_layout.addLayout(action_layout)
        self.setLayout(main_layout)
    def set_result(self, res, file_format=None):
        self.result = res; self.format = file_format; self.done(QDialog.Accepted)
class LogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Process Log")
        self.setGeometry(100, 100, 750, 550)
        self.setModal(False) 
        layout = QVBoxLayout(self)
        self.log_text_edit = QTextEdit(readOnly=True)
        self.log_text_edit.setStyleSheet("background-color:#3c3c3c; color:#f0f0f0; font-family:Consolas,monospace; font-size:9pt;")
        layout.addWidget(self.log_text_edit)
        self.setLayout(layout)
    def add_log_message(self, message):
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())
    def clear_log(self):
        self.log_text_edit.clear()


class Worker(QObject):
    log = Signal(str)
    match_found = Signal(str, str, str) 
    prompt_for_saving = Signal()
    finished_saving = Signal(str, str)
    error = Signal(str)
    current_task = Signal(str, int, int)

    def __init__(self, source_path, master_path, output_dir, 
                 fuzz_threshold, semantic_threshold, 
                 model_name, cross_encoder_model_name, k_candidates,
                 semantic_weight, fuzzy_weight, jaccard_threshold,
                 process_unmatched_only, source_name_col, master_name_col, group_by_col,
                 merge_cols):
        super().__init__()
        self.source_path = source_path
        self.master_path = master_path
        self.output_dir = output_dir
        self.fuzz_threshold = fuzz_threshold * 100 
        self.semantic_threshold = semantic_threshold 
        self.model_name = model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.k_candidates = k_candidates
        self.semantic_weight = semantic_weight
        self.fuzzy_weight = fuzzy_weight
        self.jaccard_threshold = jaccard_threshold 
        self.process_unmatched_only = process_unmatched_only 
        self.source_name_col = source_name_col
        self.master_name_col = master_name_col
        self.group_by_col = group_by_col
        self.merge_cols = merge_cols
        self._is_cancelled = False
        self.df_to_update = None
        self.df_master_source = None
        self.summary_df = None

    def cancel(self):
        self._is_cancelled = True
        
    # ---vvv--- THIS IS THE CORRECTED METHOD ---vvv---
    def _create_summary_report(self, final_source_df, all_matches):
        """
        Generates a simple, source-centric summary of match results.
        This version uses iterrows() to be safe with column names containing spaces.
        """
        self.log.emit("Creating simplified summary report...")
        
        match_method_lookup = {key: details['method'] for key, details in all_matches.items()}
        
        results = []
        # Use iterrows() which provides a Series for each row.
        # This allows accessing columns by their original string name, even with spaces.
        for _, row in final_source_df.iterrows():
            source_name = row[self.source_name_col]
            group_val = row[self.group_by_col]
            matched_name = row.get('Matched Name') # .get() is safer than direct access

            if pd.notna(matched_name) and matched_name != '':
                status = "Matched"
                method = match_method_lookup.get((source_name, group_val), 'Pre-Matched')
            else:
                status = "Unmatched"
                method = ''
                matched_name = ''

            results.append({
                'Match Status': status,
                'Source Name': source_name,
                'Matched Name': matched_name,
                'Match Method': method
            })

        self.summary_df = pd.DataFrame(results)
        self.log.emit("Summary report created successfully.")
    # ---^^^--- END OF CORRECTED METHOD ---^^^---

    def run(self):
        try:
            start_total_time = time.time()
            matched_name_col = 'Matched Name'
            num_cpus = cpu_count()
            self.log.emit("--- Starting Hybrid Match Process ---")
            
            self.log.emit("\nLoading and preparing data...")
            self.df_to_update = pd.read_csv(self.source_path, low_memory=False) if self.source_path.lower().endswith('.csv') else pd.read_excel(self.source_path)
            self.df_master_source = pd.read_csv(self.master_path, low_memory=False) if self.master_path.lower().endswith('.csv') else pd.read_excel(self.master_path)
            if self._is_cancelled: return

            required_source_cols = [self.source_name_col, self.group_by_col]
            if not all(col in self.df_to_update.columns for col in required_source_cols): raise ValueError(f"Missing required columns in Source file: {', '.join([c for c in required_source_cols if c not in self.df_to_update.columns])}")
            required_master_cols = [self.master_name_col, self.group_by_col]
            if not all(col in self.df_master_source.columns for col in required_master_cols): raise ValueError(f"Missing required columns in Master file: {', '.join([c for c in required_master_cols if c not in self.df_master_source.columns])}")
            df_to_process = self.df_to_update
            df_already_matched = pd.DataFrame()
            if self.process_unmatched_only and matched_name_col in self.df_to_update.columns:
                self.df_to_update[matched_name_col] = self.df_to_update[matched_name_col].fillna('')
                is_unmatched_mask = self.df_to_update[matched_name_col] == ''
                df_to_process = self.df_to_update[is_unmatched_mask].copy()
                df_already_matched = self.df_to_update[~is_unmatched_mask].copy()
                self.log.emit(f"Found {len(df_already_matched)} pre-matched rows. Processing {len(df_to_process)} unmatched rows.")
            df_to_process[self.source_name_col] = df_to_process[self.source_name_col].astype(str)
            df_to_process[self.group_by_col] = df_to_process[self.group_by_col].astype(str).str.strip().str.upper()
            self.df_master_source[self.master_name_col] = self.df_master_source[self.master_name_col].astype(str)
            self.df_master_source[self.group_by_col] = self.df_master_source[self.group_by_col].astype(str).str.strip().str.upper()
            
            source_grouped = df_to_process.groupby(self.group_by_col)
            master_grouped = self.df_master_source.groupby(self.group_by_col)
            all_matches = {}

            fuzzy_tasks = []
            for market, group in source_grouped:
                if market in master_grouped.groups:
                    master_names = master_grouped.get_group(market)[self.master_name_col].dropna().unique().tolist()
                    source_names = group[self.source_name_col].dropna().unique().tolist()
                    if source_names and master_names:
                        chunk_size = max(100, len(source_names) // (num_cpus * 2))
                        for i in range(0, len(source_names), chunk_size):
                             fuzzy_tasks.append((market, source_names[i:i + chunk_size], master_names, self.fuzz_threshold))

            if fuzzy_tasks and not self._is_cancelled:
                with Pool(processes=num_cpus) as pool:
                    for i, (market, market_matches) in enumerate(pool.imap_unordered(process_fuzzy_market_chunk, fuzzy_tasks)):
                        if self._is_cancelled: break
                        self.current_task.emit(f"Fuzzy: {market}", i + 1, len(fuzzy_tasks))
                        for source, match_details in market_matches.items():
                             all_matches[(source, market)] = match_details
                             self.match_found.emit(source, match_details['master'], match_details['method'])

            self.log.emit("\n--- Pass 2: Running Semantic AI on Unmatched Names ---")
            df_to_process['temp_key'] = list(zip(df_to_process[self.source_name_col], df_to_process[self.group_by_col]))
            unmatched_df = df_to_process[~df_to_process['temp_key'].isin(all_matches.keys())]
            semantic_tasks = []
            if not unmatched_df.empty:
                for market, group in unmatched_df.groupby(self.group_by_col):
                     if market in master_grouped.groups:
                        master_names = master_grouped.get_group(market)[self.master_name_col].dropna().unique().tolist()
                        source_names = group[self.source_name_col].dropna().unique().tolist()
                        if source_names and master_names:
                             semantic_tasks.append((market, source_names, master_names, self.semantic_threshold, self.k_candidates, self.semantic_weight, self.fuzzy_weight, self.jaccard_threshold))
            if semantic_tasks and not self._is_cancelled:
                with Pool(processes=num_cpus, initializer=pool_initializer_semantic, initargs=(self.model_name, self.cross_encoder_model_name,)) as pool:
                    for i, (market, market_matches, _) in enumerate(pool.imap_unordered(process_semantic_market_chunk, semantic_tasks)):
                        if self._is_cancelled: break
                        self.current_task.emit(f"AI: {market}", i + 1, len(semantic_tasks))
                        for source, match_details in market_matches.items():
                             all_matches[(source, market)] = match_details
                             self.match_found.emit(source, match_details['master'], match_details['method'])
            
            if self._is_cancelled: return

            self.log.emit("\nFinalizing results and performing join...")
            df_to_process[matched_name_col] = df_to_process['temp_key'].map(lambda k: all_matches.get(k, {}).get('master'))
            df_to_process.drop(columns=['temp_key'], inplace=True)
            if not df_already_matched.empty: final_source_df = pd.concat([df_already_matched, df_to_process], ignore_index=True)
            else: final_source_df = df_to_process

            master_for_join = self.df_master_source.drop_duplicates(subset=[self.master_name_col, self.group_by_col])
            cols_to_keep_in_master = list(set([self.master_name_col, self.group_by_col] + self.merge_cols))
            master_for_join = master_for_join[cols_to_keep_in_master]
            self.log.emit(f"Merging selected master columns: {', '.join(self.merge_cols)}")

            self.df_to_update = pd.merge(
                final_source_df,
                master_for_join,
                how='left',
                left_on=[matched_name_col, self.group_by_col],
                right_on=[self.master_name_col, self.group_by_col]
            )
            self.log.emit("Join complete. Final output file created.")
            
            self._create_summary_report(final_source_df, all_matches)
            
            null_string_pattern = re.compile(r'^\s*(nan|none|null|nat)\s*$', re.IGNORECASE)
            for df in [self.df_to_update, self.summary_df]:
                if df is not None and not df.empty:
                    df.fillna('', inplace=True)
                    df.replace(to_replace=null_string_pattern, value='', regex=True, inplace=True)
            
            end_total_time = time.time()
            self.log.emit(f"\nTotal Processing Time: {end_total_time - start_total_time:.2f} seconds.")
            self.prompt_for_saving.emit()
            
        except (ValueError, KeyError) as e:
            self.error.emit(f"Column Error: Please check that column names are correct and exist in your files. Details: {e}")
        except Exception as e:
            import traceback
            self.error.emit(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
            
    def save_files(self, save_option, file_format):
        try:
            base_name = os.path.splitext(os.path.basename(self.source_path))[0]
            if save_option == 'quit': self.finished_saving.emit(None, 'quit'); return
            df_to_save, suffix = None, ""
            if save_option == 'main': df_to_save, suffix = self.df_to_update, "_matched"
            elif save_option == 'summary_report': df_to_save, suffix = self.summary_df, "_summary_report"
            if df_to_save is not None and not df_to_save.empty:
                output_path = os.path.join(self.output_dir, f"{base_name}{suffix}.{file_format}")
                self.log.emit(f"\nSaving '{os.path.basename(output_path)}'...")
                if file_format == "xlsx": df_to_save.to_excel(output_path, index=False, na_rep='')
                else: df_to_save.to_csv(output_path, index=False, na_rep='')
                self.log.emit("Save successful!")
            elif df_to_save is not None: self.log.emit(f"\nSkipping save for '{save_option}' because it is empty.")
            self.finished_saving.emit(self.output_dir, save_option)
        except Exception as e: self.error.emit(f"An error occurred during saving: {e}")


# --- MatcherApp GUI Class (Unchanged) ---
class MatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pipelined Hybrid Matcher (Fuzzy + AI)")
        self.setGeometry(0, 0, 850, 900); self.center_window()
        self.thread, self.worker, self.save_dialog = None, None, None
        self.log_dialog = None 
        self.source_columns = []
        self.master_columns = []
        self.setStyleSheet("""
            QWidget{ background-color:#2b2b2b; color:#f0f0f0; font-family:Segoe UI,sans-serif; font-size:10pt }
            QGroupBox{ font-weight:bold; border:1px solid #555; border-radius:4px; margin-top:10pt }
            QGroupBox::title{ subcontrol-origin:margin; subcontrol-position:top left; padding:0 5px }
            QPushButton{ background-color:#0078d7; border:none; padding:8px 16px; border-radius:4px; font-weight:bold; }
            QPushButton:hover{ background-color:#1088e7 } QPushButton:disabled{ background-color:#444; color:#888 }
            QLineEdit,QTextEdit,QSpinBox,QDoubleSpinBox,QComboBox,QListWidget { background-color:#3c3c3c; border:1px solid #555; border-radius:4px; padding:4px }
            QLabel#title_label, QLabel#header_label { font-weight:bold; }
            QSlider::groove:horizontal{ height:8px; background:#3c3c3c; margin:2px 0; border-radius:4px }
            QSlider::handle:horizontal{ background:#0078d7; width:18px; margin:-5px 0; border-radius:9px }
        """)
        
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget); main_layout.setContentsMargins(15, 15, 15, 15); main_layout.setSpacing(10)
        top_h_layout = QHBoxLayout()
        left_panel_layout = QVBoxLayout()
        right_panel_layout = QVBoxLayout()
        top_h_layout.addLayout(left_panel_layout, stretch=1)
        top_h_layout.addLayout(right_panel_layout, stretch=1)
        main_layout.addLayout(top_h_layout)
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout(file_group)
        self.source_file_entry = QLineEdit(readOnly=True); self.browse_source_btn = QPushButton("Browse...")
        self.master_file_entry = QLineEdit(readOnly=True); self.browse_master_btn = QPushButton("Browse...")
        self.output_dir_entry = QLineEdit(); self.browse_output_dir_btn = QPushButton("Select Folder...")
        file_layout.addWidget(QLabel("Source File:"), 0, 0); file_layout.addWidget(self.source_file_entry, 0, 1); file_layout.addWidget(self.browse_source_btn, 0, 2)
        file_layout.addWidget(QLabel("Master File:"), 1, 0); file_layout.addWidget(self.master_file_entry, 1, 1); file_layout.addWidget(self.browse_master_btn, 1, 2)
        file_layout.addWidget(QLabel("Output Folder:"), 2, 0); file_layout.addWidget(self.output_dir_entry, 2, 1); file_layout.addWidget(self.browse_output_dir_btn, 2, 2)
        left_panel_layout.addWidget(file_group)
        col_group = QGroupBox("Column Mapping")
        col_layout = QGridLayout(col_group)
        self.source_col_combobox = QComboBox(); self.source_col_combobox.setPlaceholderText("Select Source File")
        self.master_col_combobox = QComboBox(); self.master_col_combobox.setPlaceholderText("Select Master File")
        self.group_by_col_combobox = QComboBox(); self.group_by_col_combobox.setPlaceholderText("Select Both Files")
        col_layout.addWidget(QLabel("Source Name Column:"), 0, 0); col_layout.addWidget(self.source_col_combobox, 0, 1)
        col_layout.addWidget(QLabel("Master Name Column:"), 1, 0); col_layout.addWidget(self.master_col_combobox, 1, 1)
        col_layout.addWidget(QLabel("Grouping Column (common):"), 2, 0); col_layout.addWidget(self.group_by_col_combobox, 2, 1)
        left_panel_layout.addWidget(col_group)
        left_panel_layout.addStretch()
        merge_group = QGroupBox("Master Columns to Merge")
        merge_layout = QVBoxLayout(merge_group)
        self.merge_cols_listwidget = QListWidget()
        self.merge_cols_listwidget.setAlternatingRowColors(True)
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(deselect_all_btn)
        merge_layout.addLayout(btn_layout)
        merge_layout.addWidget(self.merge_cols_listwidget)
        right_panel_layout.addWidget(merge_group)
        bottom_grid_layout = QGridLayout()
        main_layout.addLayout(bottom_grid_layout)
        options_group = QGroupBox("Matching Strategy")
        options_layout = QGridLayout(options_group)
        self.fuzz_slider = QSlider(Qt.Horizontal); self.fuzz_slider.setRange(85, 100); self.fuzz_slider.setValue(95)
        self.fuzz_threshold_spinbox = QDoubleSpinBox(); self.fuzz_threshold_spinbox.setRange(0.85, 1.0); self.fuzz_threshold_spinbox.setDecimals(2); self.fuzz_threshold_spinbox.setSingleStep(0.01); self.fuzz_threshold_spinbox.setValue(0.95)
        options_layout.addWidget(QLabel("<b>Fuzzy Thresh:</b>"), 0, 0); options_layout.addWidget(self.fuzz_slider, 0, 1); options_layout.addWidget(self.fuzz_threshold_spinbox, 0, 2)
        self.semantic_slider = QSlider(Qt.Horizontal); self.semantic_slider.setRange(0, 100); self.semantic_slider.setValue(90)
        self.semantic_threshold_spinbox = QDoubleSpinBox(); self.semantic_threshold_spinbox.setRange(0.0, 1.0); self.semantic_threshold_spinbox.setDecimals(2); self.semantic_threshold_spinbox.setSingleStep(0.01); self.semantic_threshold_spinbox.setValue(0.90)
        options_layout.addWidget(QLabel("<b>AI Thresh:</b>"), 1, 0); options_layout.addWidget(self.semantic_slider, 1, 1); options_layout.addWidget(self.semantic_threshold_spinbox, 1, 2)
        bottom_grid_layout.addWidget(options_group, 0, 0)
        hybrid_group = QGroupBox("Hybrid Scoring Weights (AI Pass)")
        hybrid_layout = QGridLayout(hybrid_group)
        self.semantic_weight_slider = QSlider(Qt.Horizontal); self.semantic_weight_slider.setRange(0, 100); self.semantic_weight_slider.setValue(30)
        self.semantic_weight_spinbox = QDoubleSpinBox(); self.semantic_weight_spinbox.setRange(0.0, 1.0); self.semantic_weight_spinbox.setDecimals(2); self.semantic_weight_spinbox.setSingleStep(0.05); self.semantic_weight_spinbox.setValue(0.30)
        hybrid_layout.addWidget(QLabel("<b>Semantic:</b>"), 0, 0); hybrid_layout.addWidget(self.semantic_weight_slider, 0, 1); hybrid_layout.addWidget(self.semantic_weight_spinbox, 0, 2)
        self.fuzzy_weight_slider = QSlider(Qt.Horizontal); self.fuzzy_weight_slider.setRange(0, 100); self.fuzzy_weight_slider.setValue(70)
        self.fuzzy_weight_spinbox = QDoubleSpinBox(); self.fuzzy_weight_spinbox.setRange(0.0, 1.0); self.fuzzy_weight_spinbox.setDecimals(2); self.fuzzy_weight_spinbox.setSingleStep(0.05); self.fuzzy_weight_spinbox.setValue(0.70)
        hybrid_layout.addWidget(QLabel("<b>Fuzzy:</b>"), 1, 0); hybrid_layout.addWidget(self.fuzzy_weight_slider, 1, 1); hybrid_layout.addWidget(self.fuzzy_weight_spinbox, 1, 2)
        self.jaccard_thresh_slider = QSlider(Qt.Horizontal); self.jaccard_thresh_slider.setRange(0, 100); self.jaccard_thresh_slider.setValue(60)
        self.jaccard_thresh_spinbox = QDoubleSpinBox(); self.jaccard_thresh_spinbox.setRange(0.0, 1.0); self.jaccard_thresh_spinbox.setDecimals(2); self.jaccard_thresh_spinbox.setSingleStep(0.05); self.jaccard_thresh_spinbox.setValue(0.60)
        hybrid_layout.addWidget(QLabel("<b>Jaccard:</b>"), 2, 0); hybrid_layout.addWidget(self.jaccard_thresh_slider, 2, 1); hybrid_layout.addWidget(self.jaccard_thresh_spinbox, 2, 2)
        bottom_grid_layout.addWidget(hybrid_group, 0, 1)
        perf_group = QGroupBox("Performance (AI)"); perf_layout = QGridLayout(perf_group)
        self.k_spinbox = QSpinBox(); self.k_spinbox.setRange(3, 20); self.k_spinbox.setValue(10)
        perf_layout.addWidget(QLabel("Candidates (K):"), 0, 0, Qt.AlignLeft); perf_layout.addWidget(self.k_spinbox, 0, 1, Qt.AlignLeft); perf_layout.setColumnStretch(2, 1)
        bottom_grid_layout.addWidget(perf_group, 0, 2)
        action_widget = QWidget()
        run_control_layout = QVBoxLayout(action_widget); run_control_layout.setContentsMargins(0,10,0,0)
        self.unmatched_only_checkbox = QCheckBox("Process unmatched merchants only")
        run_control_layout.addWidget(self.unmatched_only_checkbox, 0, Qt.AlignCenter)
        action_button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Matching"); self.run_button.setDisabled(True)
        self.view_log_btn = QPushButton("View Live Log")
        self.show_save_dialog_btn = QPushButton("Save Options"); self.show_save_dialog_btn.hide()
        action_button_layout.addWidget(self.run_button); action_button_layout.addWidget(self.view_log_btn); action_button_layout.addWidget(self.show_save_dialog_btn)
        run_control_layout.addLayout(action_button_layout)
        main_layout.addWidget(action_widget)
        self.progress_label = QLabel("Ready to run."); self.progress_label.setStyleSheet("font-weight:bold; color:#00aaff;")
        main_layout.addWidget(self.progress_label)
        self.log_textbox = QTextEdit(readOnly=True); self.log_textbox.setMaximumHeight(80)
        main_layout.addWidget(self.log_textbox)
        self.browse_source_btn.clicked.connect(self.select_source_file); self.browse_master_btn.clicked.connect(self.select_master_file)
        self.browse_output_dir_btn.clicked.connect(self.select_output_dir); self.run_button.clicked.connect(self.start_processing)
        self.show_save_dialog_btn.clicked.connect(self.on_prompt_for_saving); self.view_log_btn.clicked.connect(self.show_log_window)
        select_all_btn.clicked.connect(self.select_all_merge_cols)
        deselect_all_btn.clicked.connect(self.deselect_all_merge_cols)
        for w in [self.source_file_entry, self.master_file_entry, self.output_dir_entry]: w.textChanged.connect(self.check_inputs)
        for cb in [self.source_col_combobox, self.master_col_combobox, self.group_by_col_combobox]: cb.currentIndexChanged.connect(self.check_inputs)
        self.fuzz_slider.valueChanged.connect(lambda v: self.update_spinbox_from_slider(v, self.fuzz_threshold_spinbox)); self.fuzz_threshold_spinbox.valueChanged.connect(lambda v: self.update_slider_from_spinbox(v, self.fuzz_slider))
        self.semantic_slider.valueChanged.connect(lambda v: self.update_spinbox_from_slider(v, self.semantic_threshold_spinbox)); self.semantic_threshold_spinbox.valueChanged.connect(lambda v: self.update_slider_from_spinbox(v, self.semantic_slider))
        self.semantic_weight_slider.valueChanged.connect(lambda v: self.update_spinbox_from_slider(v, self.semantic_weight_spinbox)); self.semantic_weight_spinbox.valueChanged.connect(lambda v: self.update_slider_from_spinbox(v, self.semantic_weight_slider))
        self.fuzzy_weight_slider.valueChanged.connect(lambda v: self.update_spinbox_from_slider(v, self.fuzzy_weight_spinbox)); self.fuzzy_weight_spinbox.valueChanged.connect(lambda v: self.update_slider_from_spinbox(v, self.fuzzy_weight_slider))
        self.jaccard_thresh_slider.valueChanged.connect(lambda v: self.update_spinbox_from_slider(v, self.jaccard_thresh_spinbox)); self.jaccard_thresh_spinbox.valueChanged.connect(lambda v: self.update_slider_from_spinbox(v, self.jaccard_thresh_slider))
        self.semantic_weight_spinbox.valueChanged.connect(self.sync_weights); self.fuzzy_weight_spinbox.valueChanged.connect(self.sync_weights)
    def _get_columns_from_file(self, file_path):
        if not file_path or not os.path.exists(file_path): return []
        try:
            if file_path.lower().endswith('.csv'): df_header = pd.read_csv(file_path, nrows=0, low_memory=False)
            else: df_header = pd.read_excel(file_path, nrows=0)
            return df_header.columns.tolist()
        except Exception as e:
            QMessageBox.warning(self, "File Read Error", f"Could not read columns from {os.path.basename(file_path)}.\n\nError: {e}")
            return []
    def _update_group_by_combobox(self):
        self.group_by_col_combobox.clear()
        if self.source_columns and self.master_columns:
            common_cols = sorted(list(set(self.source_columns) & set(self.master_columns)))
            self.group_by_col_combobox.addItems(common_cols)
            if not common_cols: self.log_textbox.append("Warning: No common columns found for grouping.")
            else: self.log_textbox.append("Found common columns for grouping.")
        self.check_inputs()
    def _populate_merge_cols_list(self):
        self.merge_cols_listwidget.clear()
        for col_name in self.master_columns:
            item = QListWidgetItem(col_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.merge_cols_listwidget.addItem(item)
    def select_all_merge_cols(self):
        for i in range(self.merge_cols_listwidget.count()):
            self.merge_cols_listwidget.item(i).setCheckState(Qt.Checked)
    def deselect_all_merge_cols(self):
        for i in range(self.merge_cols_listwidget.count()):
            self.merge_cols_listwidget.item(i).setCheckState(Qt.Unchecked)
    def update_spinbox_from_slider(self, value, spinbox): spinbox.blockSignals(True); spinbox.setValue(value / 100.0); spinbox.blockSignals(False)
    def update_slider_from_spinbox(self, value, slider): slider.blockSignals(True); slider.setValue(int(value * 100)); slider.blockSignals(False)
    def sync_weights(self):
        sender = self.sender()
        sem_val, fuzz_val = self.semantic_weight_spinbox.value(), self.fuzzy_weight_spinbox.value()
        if abs(sem_val + fuzz_val - 1.0) > 0.001:
            if sender == self.semantic_weight_spinbox: self.fuzzy_weight_spinbox.setValue(round(1.0 - sem_val, 2))
            elif sender == self.fuzzy_weight_spinbox: self.semantic_weight_spinbox.setValue(round(1.0 - fuzz_val, 2))
            self.update_slider_from_spinbox(self.fuzzy_weight_spinbox.value(), self.fuzzy_weight_slider)
            self.update_slider_from_spinbox(self.semantic_weight_spinbox.value(), self.semantic_weight_slider)
    def center_window(self):
        try: self.move(self.screen().geometry().center() - self.rect().center())
        except: pass
    def check_inputs(self):
        paths_ok = all([self.source_file_entry.text(), self.master_file_entry.text(), self.output_dir_entry.text()])
        cols_ok = all([self.source_col_combobox.currentIndex() > -1, self.master_col_combobox.currentIndex() > -1, self.group_by_col_combobox.currentIndex() > -1])
        self.run_button.setEnabled(paths_ok and cols_ok)
    def select_source_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Source Data File", "", "Excel/CSV Files (*.xlsx *.xls *.csv)");
        if path: 
            self.source_file_entry.setText(path)
            self.source_columns = self._get_columns_from_file(path)
            self.source_col_combobox.clear(); self.source_col_combobox.addItems(self.source_columns)
            self._update_group_by_combobox()
    def select_master_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Master Reference File", "", "Excel/CSV Files (*.xlsx *.xls *.csv)");
        if path: 
            self.master_file_entry.setText(path)
            self.master_columns = self._get_columns_from_file(path)
            self.master_col_combobox.clear(); self.master_col_combobox.addItems(self.master_columns)
            self._update_group_by_combobox()
            self._populate_merge_cols_list()
    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder");
        if path: self.output_dir_entry.setText(path)
    def start_processing(self):
        self.log_textbox.clear()
        if self.log_dialog: self.log_dialog.clear_log()
        selected_merge_cols = []
        for i in range(self.merge_cols_listwidget.count()):
            item = self.merge_cols_listwidget.item(i)
            if item.checkState() == Qt.Checked:
                selected_merge_cols.append(item.text())
        self.run_button.setText("Cancel"); self.run_button.clicked.disconnect(); self.run_button.clicked.connect(self.cancel_processing)
        self.show_save_dialog_btn.hide()
        self.thread = QThread()
        self.worker = Worker(self.source_file_entry.text(), self.master_file_entry.text(), self.output_dir_entry.text(),
                             self.fuzz_threshold_spinbox.value(), self.semantic_threshold_spinbox.value(),
                             'Snowflake/snowflake-arctic-embed-l', 'cross-encoder/ms-marco-MiniLM-L-6-v2', self.k_spinbox.value(),
                             self.semantic_weight_spinbox.value(), self.fuzzy_weight_spinbox.value(), self.jaccard_thresh_spinbox.value(),
                             self.unmatched_only_checkbox.isChecked(),
                             source_name_col=self.source_col_combobox.currentText(),
                             master_name_col=self.master_col_combobox.currentText(),
                             group_by_col=self.group_by_col_combobox.currentText(),
                             merge_cols=selected_merge_cols)
        self.worker.moveToThread(self.thread)
        self.worker.prompt_for_saving.connect(self.on_prompt_for_saving); self.worker.finished_saving.connect(self.on_saving_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.log.connect(self.on_worker_log)
        self.worker.match_found.connect(self.on_match_found)
        self.worker.current_task.connect(self.update_progress_label)
        self.thread.started.connect(self.worker.run); self.thread.finished.connect(self.reset_ui_state); self.thread.start()
        self.progress_label.setText("Starting processing...")
    def show_log_window(self):
        if not self.log_dialog: self.log_dialog = LogDialog(self)
        self.log_dialog.show(); self.log_dialog.raise_(); self.log_dialog.activateWindow()
    def on_worker_log(self, message):
        self.log_textbox.append(message)
        if self.log_dialog: self.log_dialog.add_log_message(message)
    def on_match_found(self, source, master, method):
        if self.log_dialog:
            color_map = {"Fuzzy": "#87ceeb", "Semantic": "#98fb98"}
            color = color_map.get(method, "#ffffff")
            log_line = f'<font color="{color}"><b>[MATCH]</b></font> "{source}" -> "{master}" <font color="#aaaaaa">(Method: {method})</font>'
            self.log_dialog.add_log_message(log_line)
    def update_progress_label(self, task_name, completed, total): self.progress_label.setText(f"Processing '{task_name}'... (step {completed}/{total})")
    def on_prompt_for_saving(self):
        self.run_button.setText("Run Again"); self.run_button.clicked.disconnect(); self.run_button.clicked.connect(self.reset_ui_state)
        self.check_inputs()
        if self.save_dialog is None: self.save_dialog = SavePromptDialog(self)
        self.progress_label.setText("Processing complete. Ready to save files."); self.show_save_dialog_btn.show()
        while self.save_dialog.exec() == QDialog.Accepted:
            self.handle_save_dialog_result()
            if self.save_dialog.result in ['quit', 'run_again']: break
        self.show_save_dialog_btn.show()
    def handle_save_dialog_result(self):
        if not self.save_dialog: return
        save_option, file_format = self.save_dialog.result, self.save_dialog.format
        if save_option == 'quit': self.worker.save_files('quit', None)
        elif save_option == 'open_folder': self.open_output_folder(self.output_dir_entry.text())
        elif save_option == 'save_all':
            self.worker.save_files('main', file_format)
            self.worker.save_files('summary_report', file_format)
        elif save_option in ['main', 'summary_report']:
            self.worker.save_files(save_option, file_format)
    def on_saving_finished(self, output_dir, save_option):
        if save_option == 'quit': self.close()
    def cancel_processing(self):
        if self.worker: self.worker.cancel()
        self.run_button.setText("Cancelling..."); self.run_button.setDisabled(True); self.progress_label.setText("Cancelling...")
    def on_processing_error(self, error_msg):
        QMessageBox.critical(self, "An Unexpected Error Occurred", error_msg); self.reset_ui_state()
    def reset_ui_state(self):
        if self.thread and self.thread.isRunning(): self.thread.quit(); self.thread.wait()
        self.thread, self.worker = None, None
        self.run_button.setText("Run Matching")
        try: self.run_button.clicked.disconnect()
        except RuntimeError: pass
        self.run_button.clicked.connect(self.start_processing); self.check_inputs()
        self.show_save_dialog_btn.hide(); self.save_dialog = None 
        self.progress_label.setText("Ready to run.")
    def open_output_folder(self, folder_path):
        try:
            if sys.platform == "win32": os.startfile(folder_path)
            elif sys.platform == "darwin": subprocess.Popen(["open", folder_path])
            else: subprocess.Popen(["xdg-open", folder_path])
        except Exception as e: QMessageBox.critical(self, "Error", f"Could not open path: {e}")
    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(self, 'Confirm Quit', "A matching process is running. Are you sure you want to quit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.log_textbox.append("\n--- QUIT SIGNALLED: Attempting graceful shutdown... ---")
                self.cancel_processing()
                if not self.thread.wait(5000): 
                    self.log_textbox.append("Warning: Thread did not stop in time. Forcing termination.")
                    self.thread.terminate()
                event.accept()
            else: event.ignore()
        else: event.accept()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) 
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = MatcherApp()
    window.resize(1000, 650)
    window.show()
    sys.exit(app.exec())