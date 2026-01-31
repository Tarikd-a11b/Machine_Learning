import sys
import fitz  # PyMuPDF
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QFileDialog,
    QLabel, QComboBox, QHBoxLayout, QSplitter, QGroupBox, QMessageBox,
    QProgressBar, QCheckBox, QSpinBox, QTabWidget, QToolBar, QAction,
    QMainWindow, QStatusBar, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QRect
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QColor, QTextCharFormat, QTextCursor, QPen
from googletrans import Translator
import json
import os
import pickle
import time # Hata durumunda bekleme süresi için

LANGUAGES = {
    "Türkçe": "tr", "İngilizce": "en", "İspanyolca": "es", "Fransızca": "fr",
    "Almanca": "de", "İtalyanca": "it", "Rusça": "ru", "Japonca": "ja",
    "Korece": "ko", "Çince (Basit)": "zh-cn", "Arapça": "ar",
    "Portekizce": "pt", "Hollandaca": "nl",
}

class TranslationThread(QThread):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    
    def __init__(self, text, target_lang):
        super().__init__()
        self.text = text
        self.target_lang = target_lang
        self.translator = Translator()

    def run(self):
        # googletrans, ağ hatalarına veya API değişikliklerine karşı hassastır.
        # Tekrar deneme mantığı (retry logic) eklemek stabiliteyi artırır.
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                translation = self.translator.translate(self.text, dest=self.target_lang)
                self.finished.emit(self.text, translation.text)
                return
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt) # Üstel geri çekilme (Exponential backoff)
                else:
                    self.error.emit(str(e))
                    return

class PDFViewer(QLabel):
    textSelected = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.setStyleSheet("background: white; padding: 20px;")
        self.pdf_doc = None
        self.zoom = 1.5
        self.page_images = []
        self.page_texts = []
        self.page_layout_info = [] # {index, y_start, y_end, x_offset, width, height}

        # Selection logic
        self.start_pos = None
        self.end_pos = None
        self.selection_rect = None 
        self.is_selecting = False
        
    def load_pdf(self, pdf_doc, progress_callback=None):
        self.pdf_doc = pdf_doc
        self.page_images = []
        self.page_texts = []
        
        # PyMuPDF'u kullanarak tüm sayfaları render et
        for i, page in enumerate(pdf_doc):
            self.page_texts.append(page.get_text("text"))
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.samples
            qimg = QImage(img_data, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.page_images.append(pixmap)
            if progress_callback:
                progress_callback(i + 1)
                
        self.render_all_pages()

    def render_all_pages(self):
        if not self.page_images:
            self.setPixmap(QPixmap()) # Görüntüyü temizle
            return

        # Toplam yükseklik ve genişliği hesapla
        total_height = sum(img.height() for img in self.page_images)
        total_height += 30 * len(self.page_images) # Sayfa başlıkları ve boşluklar için
        max_width = max(img.width() for img in self.page_images)
        
        combined = QPixmap(max_width, total_height)
        combined.fill(Qt.white)
        painter = QPainter(combined)
        
        y_offset = 0
        self.page_layout_info = [] # Sayfa koordinatlarını kaydetmek için
        
        for i, img in enumerate(self.page_images):
            # Sayfa başlığını çiz
            painter.setPen(Qt.gray)
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, y_offset + 15, f"Sayfa {i + 1}")
            y_offset += 25
            
            # Sayfayı ortala ve çiz
            x_offset = (max_width - img.width()) // 2
            painter.drawPixmap(x_offset, y_offset, img)
            
            # Sayfa koordinatlarını kaydet
            page_height = img.height()
            self.page_layout_info.append({
                'index': i,
                'y_start': y_offset,
                'y_end': y_offset + page_height,
                'x_offset': x_offset,
                'width': img.width(),
                'height': page_height
            })
            y_offset += page_height + 5
            
        painter.end()
        self.setPixmap(combined)
        
    def mousePressEvent(self, event):
        """Seçimi başlatır"""
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.is_selecting = True
            self.selection_rect = None
            self.update() # Ekranı yeniden çiz

    def mouseMoveEvent(self, event):
        """Seçim dikdörtgenini günceller"""
        if self.is_selecting and self.start_pos:
            self.end_pos = event.pos()
            # Başlangıç ve bitiş noktalarından normalleştirilmiş dikdörtgeni hesapla
            self.selection_rect = QRect(self.start_pos, self.end_pos).normalized()
            self.update() # paintEvent'i tetikle

    def mouseReleaseEvent(self, event):
        """Seçimi bitirir ve metni ayıklar"""
        if self.is_selecting and self.start_pos and self.end_pos:
            self.is_selecting = False
            
            selection_rect = QRect(self.start_pos, self.end_pos).normalized()
            selected_text = self.extract_text_from_selection(selection_rect)
            
            if selected_text:
                self.textSelected.emit(selected_text)

            # Seçimi temizle
            self.selection_rect = None
            self.update()
            
        self.start_pos = None
        self.end_pos = None
    
    def paintEvent(self, event):
        # Temel QLabel çizimini çağır
        super().paintEvent(event) 

        # Seçim dikdörtgenini çiz
        if self.selection_rect and self.start_pos: # Sadece seçim yapılırken çiz
            painter = QPainter(self)
            painter.setPen(QPen(QColor(0, 0, 255, 180), 1, Qt.DotLine))
            painter.setBrush(QColor(0, 0, 255, 30)) # Yarı şeffaf mavi dolgu
            painter.drawRect(self.selection_rect)
            painter.end()

    def extract_text_from_selection(self, rect):
        """Ekran koordinatlarından metin ayıklar (PyMuPDF'e tersine mühendislik)"""
        if not self.pdf_doc:
            return ""

        selected_pages_text = []
        zoom = self.zoom
        
        # Seçimin çakıştığı sayfaları bul
        for info in self.page_layout_info:
            page_index = info['index']

            # Seçimin dikey olarak sayfa sınırları içinde olup olmadığını kontrol et
            if rect.bottom() >= info['y_start'] and rect.top() <= info['y_end']:
                page = self.pdf_doc[page_index]
                
                # 1. Seçim dikdörtgenini sayfa görseline göre sınırla
                page_rect_x0 = max(rect.left(), info['x_offset']) 
                page_rect_y0 = max(rect.top(), info['y_start'])
                page_rect_x1 = min(rect.right(), info['x_offset'] + info['width'])
                page_rect_y1 = min(rect.bottom(), info['y_end'])

                # 2. Sınırlandırılmış Qt koordinatlarını sayfa görseline göre göreli hale getir
                x0_pix = page_rect_x0 - info['x_offset']
                y0_pix = page_rect_y0 - info['y_start']
                x1_pix = page_rect_x1 - info['x_offset']
                y1_pix = page_rect_y1 - info['y_start']
                
                # Geçersiz bir seçim alanı olup olmadığını kontrol et
                if x1_pix <= x0_pix or y1_pix <= y0_pix:
                    continue

                # 3. Piksel koordinatlarını PyMuPDF'un PDF birimi koordinatlarına dönüştür (Zoom'u tersine çevir)
                x0_fitz = x0_pix / zoom
                y0_fitz = y0_pix / zoom
                x1_fitz = x1_pix / zoom
                y1_fitz = y1_pix / zoom
                
                fitz_rect = fitz.Rect(x0_fitz, y0_fitz, x1_fitz, y1_fitz)
                
                # 4. PyMuPDF kullanarak metni ayıkla
                page_text = page.get_text("text", clip=fitz_rect).strip()
                
                if page_text:
                    selected_pages_text.append(page_text)

        return " ".join(selected_pages_text)


class PDFTranslator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📚 PDF Çeviri Asistanı Pro")
        self.resize(1600, 950)
        self.settings_file = "pdf_translator_settings.json"
        self.highlights_file = "pdf_highlights.pkl"
        self.load_settings()
        self.translator = Translator()
        self.pdf_doc = None
        self.last_translated_text = ""
        self.translation_thread = None
        self.current_file_path = ""
        self.highlights = {}  # {file_path: [(start, end, color), ...]}
        self.load_highlights()
        self.init_ui()
        self.apply_clean_style()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # TOOLBAR - KOYU MAVİ
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e3c72, stop:1 #2a5298);
                spacing: 15px;
                padding: 12px 20px;
                border-bottom: 2px solid #1a2f5c;
            }
            QToolButton {
                background: rgba(255, 255, 255, 0.15);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 13px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 0.25);
                border-color: rgba(255, 255, 255, 0.5);
            }
        """)
        
        open_action = QAction("📁 PDF Aç", self)
        open_action.triggered.connect(self.open_pdf)
        toolbar.addAction(open_action)
        toolbar.addSeparator()
        
        lang_label = QLabel("🌍 Çeviri Dili:")
        lang_label.setStyleSheet("color: white; font-weight: 600; background: transparent;")
        toolbar.addWidget(lang_label)
        
        self.target_lang_combo = QComboBox()
        for lang_name in LANGUAGES.keys():
            self.target_lang_combo.addItem(lang_name)
        self.target_lang_combo.setCurrentText(self.settings.get("target_lang", "Türkçe"))
        self.target_lang_combo.currentTextChanged.connect(self.save_settings)
        self.target_lang_combo.setStyleSheet("""
            QComboBox {
                background: white;
                border: 2px solid #1e3c72;
                border-radius: 5px;
                padding: 6px 12px;
                min-width: 150px;
                font-weight: 600;
                color: #1e3c72;
            }
            QComboBox:hover {
                border-color: #2a5298;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        toolbar.addWidget(self.target_lang_combo)
        toolbar.addSeparator()
        
        # Zoom kontrolleri
        zoom_label = QLabel("🔍")
        zoom_label.setStyleSheet("color: white; font-size: 16px; background: transparent;")
        toolbar.addWidget(zoom_label)
        
        zoom_out_action = QAction("🔍−", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        self.zoom_display = QLabel(f"{self.settings.get('zoom', 150)}%")
        self.zoom_display.setStyleSheet("color: white; font-weight: 600; background: transparent; padding: 0 10px;")
        toolbar.addWidget(self.zoom_display)
        
        zoom_in_action = QAction("🔍+", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        toolbar.addSeparator()
        
        # Altını çizme butonu
        highlight_action = QAction("🖍️ Altını Çiz", self)
        highlight_action.triggered.connect(self.highlight_text)
        toolbar.addAction(highlight_action)
        
        clear_highlight_action = QAction("🗑️ Çizgileri Temizle", self)
        clear_highlight_action.triggered.connect(self.clear_highlights)
        toolbar.addAction(clear_highlight_action)
        
        toolbar.addSeparator()
        
        copy_action = QAction("📋 Kopyala", self)
        copy_action.triggered.connect(self.copy_translation)
        toolbar.addAction(copy_action)
        
        self.addToolBar(toolbar)

        # TAB WIDGET
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                background: #fafafa;
            }
            QTabBar::tab {
                background: #f5f5f5;
                color: #555;
                border: 1px solid #e0e0e0;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 25px;
                margin-right: 3px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: white;
                color: #1e3c72;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background: #eeeeee;
            }
        """)
        
        # TAB 1: Okuma
        reading_tab = QWidget()
        reading_layout = QVBoxLayout()
        reading_tab.setLayout(reading_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # PDF Container
        pdf_scroll = QScrollArea()
        pdf_scroll.setWidgetResizable(True)
        pdf_scroll.setStyleSheet("QScrollArea { border: none; background: #f8f8f8; }")
        
        # PDFViewer artık fare ile seçim yapabilir
        self.pdf_viewer = PDFViewer()
        self.pdf_viewer.textSelected.connect(self.on_text_selected)
        
        pdf_scroll.setWidget(self.pdf_viewer)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.selectionChanged.connect(self.translate_selected_text)
        self.text_area.setFont(QFont("Georgia", 11))
        self.text_area.setStyleSheet("""
            QTextEdit {
                background: white;
                border: 1px solid #e0e0e0;
                padding: 20px;
                line-height: 1.8;
                color: #333;
            }
        """)
        
        pdf_container = QTabWidget()
        pdf_container.addTab(pdf_scroll, "📄 Görsel Görünüm (Seçim Aktif)")
        pdf_container.addTab(self.text_area, "📝 Metin Görünümü (Çizim Aktif)")
        pdf_container.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #e0e0e0; }
            QTabBar::tab {
                background: #f5f5f5;
                padding: 8px 15px;
                border: 1px solid #e0e0e0;
                border-bottom: none;
                font-size: 12px;
            }
            QTabBar::tab:selected { background: white; color: #1e3c72; font-weight: 600; }
        """)
        splitter.addWidget(pdf_container)

        # Çeviri Paneli - GRADIENT ARKA PLAN
        translation_container = QWidget()
        translation_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-left: 2px solid #5a67d8;
            }
        """)
        translation_layout = QVBoxLayout()
        translation_container.setLayout(translation_layout)
        
        title_label = QLabel("✨ Anlık Çeviri")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: 700;
            color: white;
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            margin: 10px;
        """)
        translation_layout.addWidget(title_label)
        
        src_label = QLabel("📝 Seçilen Metin:")
        src_label.setStyleSheet("color: white; font-weight: 600; padding: 10px 15px 5px 15px; background: transparent;")
        translation_layout.addWidget(src_label)
        
        self.source_text_display = QTextEdit()
        self.source_text_display.setPlaceholderText("Metni seçin...")
        self.source_text_display.setReadOnly(True)
        self.source_text_display.setMaximumHeight(120)
        self.source_text_display.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 8px;
                padding: 12px;
                font-size: 11px;
                color: #333;
                margin: 0 10px;
            }
        """)
        translation_layout.addWidget(self.source_text_display)
        
        trans_label = QLabel("🌐 Çeviri:")
        trans_label.setStyleSheet("color: white; font-weight: 600; padding: 15px 15px 5px 15px; background: transparent;")
        translation_layout.addWidget(trans_label)
        
        self.translation_display = QTextEdit()
        self.translation_display.setPlaceholderText("Çeviri burada görünecek...")
        self.translation_display.setReadOnly(True)
        self.translation_display.setFont(QFont("Segoe UI", 12))
        self.translation_display.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 0.98);
                border: 2px solid rgba(255, 255, 255, 0.6);
                border-radius: 8px;
                padding: 15px;
                color: #222;
                font-weight: 600;
                margin: 0 10px 10px 10px;
            }
        """)
        translation_layout.addWidget(self.translation_display)
        
        splitter.addWidget(translation_container)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)
        reading_layout.addWidget(splitter)
        
        # TAB 2: Ayarlar
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        settings_tab.setLayout(settings_layout)
        
        settings_group = QGroupBox("⚙️ Çeviri Ayarları")
        settings_group.setStyleSheet("""
            QGroupBox {
                background: white;
                border: 2px solid #1e3c72;
                border-radius: 8px;
                padding: 20px;
                margin: 10px;
                font-weight: 600;
                color: #1e3c72;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
        """)
        sg_layout = QVBoxLayout()
        
        auto_layout = QHBoxLayout()
        auto_layout.addWidget(QLabel("⚡ Otomatik Çeviri:"))
        self.auto_translate_check = QCheckBox("Etkin")
        self.auto_translate_check.setChecked(self.settings.get("auto_translate", True))
        self.auto_translate_check.stateChanged.connect(self.save_settings)
        auto_layout.addWidget(self.auto_translate_check)
        auto_layout.addStretch()
        sg_layout.addLayout(auto_layout)
        
        char_layout = QHBoxLayout()
        char_layout.addWidget(QLabel("📏 Maksimum Karakter:"))
        self.char_limit_spin = QSpinBox()
        self.char_limit_spin.setRange(100, 2000)
        self.char_limit_spin.setValue(self.settings.get("char_limit", 500))
        self.char_limit_spin.setSuffix(" karakter")
        self.char_limit_spin.valueChanged.connect(self.save_settings)
        char_layout.addWidget(self.char_limit_spin)
        char_layout.addStretch()
        sg_layout.addLayout(char_layout)
        
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("🔍 PDF Yakınlaştırma:"))
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(100, 300)
        self.zoom_spin.setValue(int(self.settings.get("zoom", 150)))
        self.zoom_spin.setSuffix("%")
        self.zoom_spin.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_spin)
        zoom_layout.addStretch()
        sg_layout.addLayout(zoom_layout)
        
        settings_group.setLayout(sg_layout)
        settings_layout.addWidget(settings_group)
        settings_layout.addStretch()
        
        self.tab_widget.addTab(reading_tab, "📖 Okuma")
        self.tab_widget.addTab(settings_tab, "⚙️ Ayarlar")
        main_layout.addWidget(self.tab_widget)

        # STATUS BAR
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: #1e3c72;
                color: white;
                font-weight: 600;
                padding: 6px 10px;
                border-top: 2px solid #1a2f5c;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("📂 Hazır! PDF dosyanızı açın.")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid white;
                border-radius: 4px;
                text-align: center;
                background: rgba(255,255,255,0.2);
                color: white;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 3px;
            }
        """)
        self.status_bar.addPermanentWidget(self.progress_bar, 1)

    def apply_clean_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #fafafa; }
            QLabel { color: #555; }
            QCheckBox {
                font-weight: 500;
                color: #444;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #1e3c72;
                border-radius: 4px;
                background: white;
            }
            QCheckBox::indicator:checked {
                background: #1e3c72;
            }
            QSpinBox {
                background: white;
                border: 2px solid #1e3c72;
                border-radius: 5px;
                padding: 5px 10px;
                min-width: 150px;
                font-weight: 600;
            }
        """)

    def get_target_lang_code(self):
        return LANGUAGES.get(self.target_lang_combo.currentText(), 'tr')

    def zoom_in(self):
        current = self.zoom_spin.value()
        if current < 300:
            self.zoom_spin.setValue(current + 10)
    
    def zoom_out(self):
        current = self.zoom_spin.value()
        if current > 100:
            self.zoom_spin.setValue(current - 10)

    def highlight_text(self):
        """Sadece QTextEdit'te (Metin Görünümü) seçili metni altını çiz"""
        cursor = self.text_area.textCursor()
        if not cursor.hasSelection():
            QMessageBox.information(self, "Bilgi", "Lütfen önce Metin Görünümü sekmesinde altını çizmek istediğiniz metni seçin.")
            return
        
        # Format uygula (Sarı highlight)
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(255, 255, 0, 100)) 
        cursor.mergeCharFormat(fmt)
        
        # Kaydet
        if self.current_file_path:
            if self.current_file_path not in self.highlights:
                self.highlights[self.current_file_path] = []
            
            # Geçerli seçimi kaydet (altını çizme işlemi geri alınırsa baştan sona kaydedilmiş olur)
            self.highlights[self.current_file_path].append((cursor.selectionStart(), cursor.selectionEnd()))
            self.save_highlights()
            self.status_bar.showMessage("✅ Altı çizildi ve kaydedildi!", 2000)

    def clear_highlights(self):
        """Tüm altı çizilmiş yerleri temizle"""
        if self.current_file_path in self.highlights:
            del self.highlights[self.current_file_path]
            self.save_highlights()
        
        # QTextEdit'teki formatları temizle
        cursor = self.text_area.textCursor()
        cursor.select(QTextCursor.Document)
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(255, 255, 255)) # Beyaz arka plan ile formatı sıfırla
        cursor.mergeCharFormat(fmt)
        
        self.status_bar.showMessage("🗑️ Tüm çizgiler temizlendi", 2000)

    def apply_saved_highlights(self):
        """Kaydedilmiş altı çizilmiş yerleri uygula"""
        if self.current_file_path not in self.highlights:
            return
        
        # Önce tüm document'ı sıfırla
        doc_cursor = self.text_area.textCursor()
        doc_cursor.select(QTextCursor.Document)
        fmt_clear = QTextCharFormat()
        fmt_clear.setBackground(QColor(255, 255, 255))
        doc_cursor.mergeCharFormat(fmt_clear)
        
        # Sonra kayıtlı highlight'ları uygula
        fmt_highlight = QTextCharFormat()
        fmt_highlight.setBackground(QColor(255, 255, 0, 100))
        
        for start, end in self.highlights[self.current_file_path]:
            cursor = self.text_area.textCursor()
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt_highlight)
            
        self.text_area.setTextCursor(QTextCursor(self.text_area.document()))

    def load_highlights(self):
        """Altı çizilmiş yerleri yükle"""
        try:
            if os.path.exists(self.highlights_file):
                with open(self.highlights_file, 'rb') as f:
                    self.highlights = pickle.load(f)
        except:
            self.highlights = {}

    def save_highlights(self):
        """Altı çizilmiş yerleri kaydet"""
        try:
            with open(self.highlights_file, 'wb') as f:
                pickle.dump(self.highlights, f)
        except:
            pass

    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "PDF Dosyası Seç", "", "PDF Files (*.pdf)")
        if not file_path:
            return
        try:
            self.pdf_doc = fitz.open(file_path)
            self.current_file_path = file_path
            self.load_pdf()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"PDF açılamadı:\n{e}")

    def load_pdf(self):
        if not self.pdf_doc:
            return
        
        # Tüm metni QTextEdit'e yükle
        all_text = ""
        for i, page in enumerate(self.pdf_doc):
            all_text += f"\n\n{'─'*80}\n📄 SAYFA {i+1}\n{'─'*80}\n\n"
            all_text += page.get_text("text")
        self.text_area.setText(all_text)
        self.apply_saved_highlights()
        
        # PDF görselini yükle
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.pdf_doc))
        
        # STABİLİTE DÜZELTMESİ: QApplication.processEvents() kaldırıldı
        def progress_callback(value):
            self.progress_bar.setValue(value)
        
        zoom_level = self.zoom_spin.value() / 100.0
        self.pdf_viewer.zoom = zoom_level
        self.pdf_viewer.load_pdf(self.pdf_doc, progress_callback)
        self.status_bar.showMessage(f"✅ {os.path.basename(self.current_file_path)} ({len(self.pdf_doc)} sayfa)")
        self.progress_bar.setVisible(False)

    def on_zoom_changed(self, value):
        self.zoom_display.setText(f"{value}%")
        if self.pdf_doc:
            zoom_level = value / 100.0
            self.pdf_viewer.zoom = zoom_level
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(self.pdf_doc))
            
            # STABİLİTE DÜZELTMESİ: QApplication.processEvents() kaldırıldı
            def progress_callback(val):
                self.progress_bar.setValue(val)
            
            self.pdf_viewer.load_pdf(self.pdf_doc, progress_callback)
            self.progress_bar.setVisible(False)
        self.save_settings()

    def on_text_selected(self, text):
        """PDFViewer'dan gelen seçili metni çeviriye gönderir"""
        if text and self.auto_translate_check.isChecked():
            self.start_translation(text)

    def translate_selected_text(self):
        """QTextEdit'ten gelen seçili metni çeviriye gönderir"""
        if not self.auto_translate_check.isChecked():
            return
        cursor = self.text_area.textCursor()
        selected_text = cursor.selectedText().strip()
        
        # Paragraf sınırlarını ve gereksiz boşlukları temizle
        if '\n' in selected_text:
            selected_text = ' '.join(selected_text.split())
        
        if not selected_text:
            return
        
        self.start_translation(selected_text)

    def start_translation(self, selected_text):
        # Devam eden bir çeviri varsa durdur
        if self.translation_thread and self.translation_thread.isRunning():
            self.translation_thread.terminate()
            self.translation_thread.wait()

        max_chars = self.char_limit_spin.value()
        display_text = selected_text
        
        if len(selected_text) > max_chars:
            display_text = selected_text[:max_chars] + "..."
            selected_text = selected_text[:max_chars]
            self.status_bar.showMessage(f"⚠️ Metin kısaltıldı ({max_chars} karakter)")
            
        if selected_text == self.last_translated_text:
            return
        
        self.last_translated_text = selected_text
        self.source_text_display.setPlainText(display_text)
        target_lang = self.get_target_lang_code()
        self.translation_display.setPlaceholderText("⏳ Çevriliyor...")
        
        self.translation_thread = TranslationThread(selected_text, target_lang)
        self.translation_thread.finished.connect(self.on_translation_finished)
        self.translation_thread.error.connect(self.on_translation_error)
        self.translation_thread.start()

    def on_translation_finished(self, original, translated):
        self.translation_display.setPlainText(translated)
        self.status_bar.showMessage(f"✅ Çeviri tamamlandı ({len(original)} karakter)")
        self.translation_thread = None

    def on_translation_error(self, error):
        self.translation_display.setPlainText(f"❌ Hata: {error}\n\n• İnternet bağlantınızı kontrol edin\n• Daha kısa metin deneyin\n• (Google Translate API geçici olarak erişilemiyor olabilir)")
        self.status_bar.showMessage("⚠️ Çeviri başarısız")
        self.translation_thread = None

    def copy_translation(self):
        text = self.translation_display.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_bar.showMessage("📋 Kopyalandı!", 2000)

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings = json.load(f)
            else:
                self.settings = {}
        except:
            self.settings = {}

    def save_settings(self):
        self.settings = {
            "target_lang": self.target_lang_combo.currentText(),
            "auto_translate": self.auto_translate_check.isChecked(),
            "char_limit": self.char_limit_spin.value(),
            "zoom": self.zoom_spin.value(),
        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PDFTranslator()
    window.show()
    sys.exit(app.exec_())