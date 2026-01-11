import sys
import os
import threading
import numpy as np
import hashlib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QSplitter, QMessageBox, QGroupBox, QScrollArea,
    QCheckBox, QSizePolicy, QGridLayout, QSlider
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import the existing conversion functionality
import torch
import rawpy
import imageio.v3 as iio
import colour

# Add the existing modules to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from convert_raw_torch import (
    FLog2Pipeline, LUT3DApplier, crop_raw_with_flips, device
)

class ConversionWorker(QThread):
    """Worker thread for converting raw images with LUTs"""
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(str, np.ndarray)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    # Class-level caches to prevent recompilation
    _compiled_processor_cache = None  # Cache for compiled FLog2Pipeline
    _compiled_lut_appliers_cache = {}  # Cache for compiled LUT appliers: {lut_hash: compiled_applier}
    
    def __init__(self, image_path, lut_folder, manual_exposure=0.0):
        super().__init__()
        self.image_path = image_path
        self.lut_folder = lut_folder
        self.manual_exposure = manual_exposure
        self.is_running = True
    
    def run(self):
        try:
            # Load raw image
            with rawpy.imread(self.image_path) as raw:
                xyz_image = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    bright=1.0,
                    output_color=rawpy.ColorSpace.XYZ,
                    gamma=(1, 1), 
                    output_bps=16
                )
                xyz_cropped = crop_raw_with_flips(xyz_image, raw.sizes)
            
            # Convert to tensor
            raw_tensor = torch.from_numpy(xyz_cropped.astype(np.float32)).to(device) / 65535.0
            
            # Create and compile processor - only for CUDA, use cache if available
            if ConversionWorker._compiled_processor_cache is None:
                processor = FLog2Pipeline()
                if device.type == 'cuda':
                    # Compile only for CUDA device
                    processor = torch.compile(processor, mode="reduce-overhead")
                # Cache the compiled/non-compiled processor
                ConversionWorker._compiled_processor_cache = processor.to(device)
            else:
                # Use cached processor
                processor = ConversionWorker._compiled_processor_cache
            
            # Process to F-Log2
            with torch.inference_mode():
                flog2_img, gain_applied = processor(raw_tensor, ev_offset=self.manual_exposure)
            
            # Get LUT files
            lut_files = [f for f in os.listdir(self.lut_folder) if f.endswith('.cube')]
            if not lut_files:
                self.error_signal.emit("No .cube files found in the selected folder.")
                return
            
            # Process each LUT
            for i, lut_file in enumerate(lut_files):
                if not self.is_running:
                    break
                
                # Update progress
                progress = int((i + 1) / len(lut_files) * 100)
                self.progress_signal.emit(progress)
                
                lut_path = os.path.join(self.lut_folder, lut_file)
                lut_name = os.path.splitext(lut_file)[0]
                
                # Load LUT
                lut_obj = colour.read_LUT(lut_path)
                lut_np = lut_obj.table.astype(np.float32)
                
                # Create a unique hash for the LUT to use as cache key
                lut_hash = hashlib.md5(lut_np.tobytes()).hexdigest()
                
                # Check if LUT applier is already in cache
                if lut_hash not in ConversionWorker._compiled_lut_appliers_cache:
                    lut_applier = LUT3DApplier(lut_np)
                    if device.type == 'cuda':
                        # Compile only for CUDA device
                        lut_applier = torch.compile(lut_applier)
                    # Cache the compiled/non-compiled applier
                    ConversionWorker._compiled_lut_appliers_cache[lut_hash] = lut_applier.to(device)
                else:
                    # Use cached applier
                    lut_applier = ConversionWorker._compiled_lut_appliers_cache[lut_hash]
                
                # Apply LUT
                with torch.inference_mode():
                    batch_img = flog2_img.unsqueeze(0)
                    final_img = lut_applier(batch_img).squeeze(0)
                
                # Convert to numpy array
                final_np = (final_img * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
                
                # Ensure image has correct dimensions (height, width, 3)
                if final_np.ndim == 3 and final_np.shape[2] == 3:
                    # Emit result
                    self.result_signal.emit(lut_name, final_np)
                elif final_np.ndim == 4 and final_np.shape[0] == 1:
                    # Remove batch dimension if present
                    final_np = final_np.squeeze(0)
                    if final_np.shape[2] == 3:
                        self.result_signal.emit(lut_name, final_np)
                    else:
                        print(f"Warning: LUT {lut_name} produced image with incorrect channels: {final_np.shape[2]}")
                else:
                    print(f"Warning: LUT {lut_name} produced image with invalid shape: {final_np.shape}")
            
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(f"Error during conversion: {str(e)}")
    
    def stop(self):
        self.is_running = False

class RawConverterApp(QMainWindow):
    """Main GUI application for raw image conversion"""
    
    def __init__(self):
        super().__init__()
        
        # Calculate scaling factor based on screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()
        
        # Base resolution (1080p)
        base_width = 1920
        base_height = 1080
        
        # Calculate scaling factor (minimum 1.0, maximum 2.0)
        self.scale_factor = min(max(self.screen_width / base_width, self.screen_height / base_height), 2.0)
        
        # Calculate adaptive thumbnail size
        self.thumbnail_size = int(100 * self.scale_factor)  # Base 100px * scale factor
        
        # Initialize UI
        self.init_ui()
        
        # Initialize other attributes
        self.current_raw_path = None
        self.lut_folder = None
        self.converted_images = {}
        self.selected_images = set()
        self.conversion_worker = None
        self.thumbnails = {}  # Store thumbnails: {lut_name: QLabel}
    
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("RAW to Fujifilm LUT Converter")
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply adaptive font sizes for 4K resolution
        font_size = int(14 * self.scale_factor)
        
        # Apply adaptive font sizes to all UI elements
        font = self.font()
        font.setPointSize(font_size)
        self.setFont(font)
        
        # Apply a clean stylesheet
        self.setStyleSheet("""
            /* Main window */
            QMainWindow {
                background-color: #2c3e50;
            }
            
            /* Widgets */
            QWidget {
                color: #ecf0f1;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            /* Buttons */
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #2980b9;
            }
            
            QPushButton:pressed {
                background-color: #1c638e;
            }
            
            /* Group boxes */
            QGroupBox {
                background-color: #34495e;
                border: 1px solid #455a64;
                border-radius: 8px;
            }
            
            /* List widget */
            QListWidget {
                background-color: #34495e;
                border: 1px solid #455a64;
                border-radius: 6px;
            }
            
            /* Checkbox */
            QCheckBox::indicator {
                border: 2px solid #3498db;
                border-radius: 3px;
                background-color: #2c3e50;
            }
            
            QCheckBox::indicator:checked {
                background-color: #3498db;
            }
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Top control bar
        self.control_bar = QHBoxLayout()
        
        # Open raw button
        self.btn_open_raw = QPushButton("Open RAW Photo")
        self.btn_open_raw.clicked.connect(self.open_raw_file)
        self.control_bar.addWidget(self.btn_open_raw)
        
        # Select LUT folder button
        self.btn_select_luts = QPushButton("Select LUT Folder")
        self.btn_select_luts.clicked.connect(self.select_lut_folder)
        self.control_bar.addWidget(self.btn_select_luts)
        
        # Convert button
        self.btn_convert = QPushButton("Convert")
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setEnabled(False)
        self.control_bar.addWidget(self.btn_convert)
        
        # Export selected button
        self.btn_export = QPushButton("Export Selected")
        self.btn_export.clicked.connect(self.export_selected)
        self.btn_export.setEnabled(False)
        self.control_bar.addWidget(self.btn_export)
        
        main_layout.addLayout(self.control_bar)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Status label (small, non-intrusive)
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(20)
        self.status_label.setMaximumHeight(30)
        main_layout.addWidget(self.status_label)
        
        # Thumbnail container (top panel)
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn) 
        self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Horizontal scrolling only

        # Create a container widget for thumbnails
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QHBoxLayout(self.thumbnail_container)
        self.thumbnail_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.thumbnail_layout.setSpacing(int(15 * self.scale_factor))  # Adaptive spacing
        # Add adaptive padding to prevent content overlap
        padding = int(15 * self.scale_factor)
        self.thumbnail_layout.setContentsMargins(padding, padding, padding, padding)

        # Calculate dynamic panel height by accounting for actual UI elements
        # 1. Start with a base height (30% of screen height)
        # 2. Adjust for control bar and status label heights
        
        # Estimate control bar height based on button heights
        button_height = self.btn_open_raw.sizeHint().height()
        control_bar_height = button_height + int(10 * self.scale_factor)  # Buttons + spacing
        
        # Use status label's maximum height
        status_label_height = self.status_label.maximumHeight()
        
        # Calculate top panel height by accounting for control bar and status label
        base_panel_height = int(self.screen_height * 0.3)
        top_panel_height = base_panel_height
        
        self.thumbnail_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.thumbnail_scroll.setMaximumHeight(top_panel_height)
        
        # Set default message when no thumbnails
        self.default_thumbnail_label = QLabel("No converted images yet")
        self.default_thumbnail_label.setAlignment(Qt.AlignCenter)
        self.default_thumbnail_label.setMinimumSize(int(400 * self.scale_factor), int(100 * self.scale_factor))
        
        # Apply adaptive font to default label
        font = self.default_thumbnail_label.font()
        font.setPointSize(int(font.pointSize() * self.scale_factor))
        self.default_thumbnail_label.setFont(font)
        self.thumbnail_layout.addWidget(self.default_thumbnail_label)
        
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        main_layout.addWidget(self.thumbnail_scroll, 3)  # Stretch factor 3 for top panel
        
        # Main splitter (preview and list)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Image preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Add padding that's 3% of the panel size to all 4 corners
        padding_percentage = 0.03  # 3%
        
        # Apply initial padding based on estimated panel size
        # This will be updated when the window is shown and resized
        estimated_width = self.screen_width * 0.5  # Approximately half the screen width
        estimated_height = self.screen_height * 0.5  # Approximately half the screen height
        
        # Base padding calculations
        pad_left = int(estimated_width * padding_percentage)
        pad_right = int(estimated_width * padding_percentage)
        pad_bottom = int(estimated_height * padding_percentage)
        
        # For top padding, account for the group box title height
        # Add extra padding for the title (estimated 25px * scale factor)
        title_padding = int(15 * self.scale_factor)
        pad_top = int(estimated_height * padding_percentage) + title_padding
        
        preview_layout.setContentsMargins(pad_left, pad_top, pad_right, pad_bottom)

        self.preview_label = QLabel("No image selected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        preview_layout.addWidget(self.preview_label)
        
        main_splitter.addWidget(preview_group)
        
        # Right side: LUT list and exposure controls
        list_group = QGroupBox("Converted Images")
        
        # Create vertical splitter for list and exposure controls
        list_splitter = QSplitter(Qt.Vertical)
        list_splitter.setHandleWidth(int(5 * self.scale_factor))
        
        # Add padding that's 3% of the panel size to all 4 corners
        padding_percentage = 0.03  # 3%
        
        # Apply initial padding based on estimated panel size
        # This will be updated when the window is shown and resized
        estimated_width = self.screen_width * 0.5  # Approximately half the screen width
        estimated_height = self.screen_height * 0.5  # Approximately half the screen height
        
        # Base padding calculations
        pad_left = int(estimated_width * padding_percentage)
        pad_right = int(estimated_width * padding_percentage)
        pad_bottom = int(estimated_height * padding_percentage)
        
        # For top padding, account for the group box title height
        # Add extra padding for the title (estimated 25px * scale factor)
        title_padding = int(15 * self.scale_factor)
        pad_top = int(estimated_height * padding_percentage) + title_padding
        
        # LUT List Section
        lut_list_widget = QWidget()
        lut_list_layout = QVBoxLayout(lut_list_widget)
        lut_list_layout.setContentsMargins(pad_left, pad_top, pad_right, pad_bottom)

        self.lut_list = QListWidget()
        self.lut_list.itemClicked.connect(self.show_preview)
        self.lut_list.itemChanged.connect(self.handle_item_changed)
        # Disable default selection mode
        self.lut_list.setSelectionMode(QListWidget.NoSelection)
        lut_list_layout.addWidget(self.lut_list)

        # Select all checkbox
        self.select_all_cb = QCheckBox("Select All")
        self.select_all_cb.stateChanged.connect(self.toggle_select_all)
        lut_list_layout.addWidget(self.select_all_cb)
        
        # Exposure Controls Section
        exposure_widget = QWidget()
        exposure_layout = QVBoxLayout(exposure_widget)
        exposure_layout.setContentsMargins(pad_left, pad_top, pad_right, pad_bottom)
        exposure_layout.setSpacing(int(10 * self.scale_factor))
        
        # Initialize manual exposure value (0.0 = no adjustment)
        self.manual_exposure = 0.0
        
        # Exposure title label
        exposure_label = QLabel("Manual Exposure")
        exposure_label.setAlignment(Qt.AlignCenter)
        # Use slightly smaller font for exposure controls
        exposure_font = exposure_label.font()
        exposure_font.setPointSize(int(exposure_font.pointSize() * self.scale_factor)) 
        exposure_label.setFont(exposure_font)
        exposure_layout.addWidget(exposure_label)
        
        # Current exposure value label
        self.exposure_value_label = QLabel(f"Value: {self.manual_exposure:.1f} EV")
        self.exposure_value_label.setAlignment(Qt.AlignCenter)
        self.exposure_value_label.setFont(exposure_font)  # Use same reduced font
        exposure_layout.addWidget(self.exposure_value_label)
        
        # Exposure slider (-3 to 3 range, with 0.1 step)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(-30)  # -3.0 EV
        self.exposure_slider.setMaximum(30)   # 3.0 EV
        self.exposure_slider.setValue(0)      # Initial value (0.0 EV)
        self.exposure_slider.setTickInterval(5)  # 0.5 EV ticks
        self.exposure_slider.setTickPosition(QSlider.TicksBelow)
        self.exposure_slider.valueChanged.connect(self.on_exposure_slider_changed)
        exposure_layout.addWidget(self.exposure_slider)
        
        # Apply exposure button
        self.apply_exposure_btn = QPushButton("Apply Exposure")
        self.apply_exposure_btn.clicked.connect(self.apply_manual_exposure)
        self.apply_exposure_btn.setFont(exposure_font)  # Use same reduced font for consistency
        # Disable until images are converted
        self.apply_exposure_btn.setEnabled(False)
        exposure_layout.addWidget(self.apply_exposure_btn)
        
        # Add widgets to splitter
        list_splitter.addWidget(lut_list_widget)
        list_splitter.addWidget(exposure_widget)
        
        # Set stretch factors to maintain 70/30 ratio during resizing
        list_splitter.setStretchFactor(0, 7)  # 70% for list
        list_splitter.setStretchFactor(1, 3)  # 30% for exposure controls
        
        # Set initial sizes (70% for list, 30% for exposure controls)
        list_splitter.setSizes([int(estimated_height * 0.7), int(estimated_height * 0.3)])
        
        # Main layout for list group
        list_layout = QVBoxLayout(list_group)
        list_layout.addWidget(list_splitter)
        
        main_splitter.addWidget(list_group)
        
        # Set splitter sizes
        main_splitter.setSizes([600, 600])
        
        # Set stretch factor 7 for main splitter (70% of height)
        main_layout.addWidget(main_splitter, 7)  
        
        # Set main window size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Store references to the layouts that need dynamic padding
        self.preview_layout = preview_layout
        self.list_layout = list_layout

        self.setCentralWidget(main_widget)
        
        # Connect resize event to update padding dynamically
        self.resizeEvent = self.update_dynamic_padding
    
    def open_raw_file(self):
        """Open a RAW image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open RAW Photo", "",
            "RAW Files (*.arw *.nef *.cr2 *.cr3 *.dng *.orf *.rw2);;All Files (*.*)"
        )
        
        if file_path:
            self.current_raw_path = file_path
            self.status_label.setText(f"Opened: {os.path.basename(file_path)}")
            self.check_convert_button()
    
    def select_lut_folder(self):
        """Select folder containing LUT files"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select LUT Folder"
        )
        
        if folder_path:
            self.lut_folder = folder_path
            lut_count = len([f for f in os.listdir(folder_path) if f.endswith('.cube')])
            self.status_label.setText(f"LUT Folder: {os.path.basename(folder_path)} ({lut_count} LUTs)")
            self.check_convert_button()
    
    def check_convert_button(self):
        """Enable convert button if both raw file and LUT folder are selected"""
        self.btn_convert.setEnabled(
            self.current_raw_path is not None and self.lut_folder is not None
        )
    
    def show_preview_by_name(self, lut_name):
        """Show preview of an image by its LUT name"""
        # Find the corresponding item in the list
        for i in range(self.lut_list.count()):
            item = self.lut_list.item(i)
            if item.text() == lut_name:
                self.show_preview(item)
                break
    
    def start_conversion(self):
        """Start the conversion process"""
        if not self.current_raw_path or not self.lut_folder:
            return
        
        # Clear previous results
        self.lut_list.clear()
        self.converted_images.clear()
        self.selected_images.clear()
        self.preview_label.setText("Converting...")
        
        # Clear thumbnails
        for thumbnail in self.thumbnails.values():
            thumbnail.hide()
            thumbnail.setParent(None)  # Remove from layout
        self.thumbnails.clear()
        
        # Restore default message
        if not hasattr(self, 'default_thumbnail_label') or not self.default_thumbnail_label.parent():
            self.default_thumbnail_label = QLabel("Converting...")
            self.default_thumbnail_label.setAlignment(Qt.AlignCenter)
            self.default_thumbnail_label.setMinimumSize(400, 100)
            self.thumbnail_layout.addWidget(self.default_thumbnail_label)
        
        # Disable buttons during conversion
        self.btn_convert.setEnabled(False)
        self.btn_open_raw.setEnabled(False)
        self.btn_select_luts.setEnabled(False)
        self.btn_export.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.conversion_worker = ConversionWorker(
            self.current_raw_path, self.lut_folder
        )
        self.conversion_worker.progress_signal.connect(self.update_progress)
        self.conversion_worker.result_signal.connect(self.add_converted_image)
        self.conversion_worker.error_signal.connect(self.show_error)
        self.conversion_worker.finished_signal.connect(self.conversion_finished)
        self.conversion_worker.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def create_thumbnail(self, lut_name, image_data):
        """Create and display a thumbnail for the converted image with LUT name label overlaid"""
        # Convert numpy array to QImage
        try:
            if not image_data.flags.contiguous:
                image_data = np.ascontiguousarray(image_data)
            
            height, width, channel = image_data.shape
            bytes_per_line = channel * width
            q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Calculate adaptive dimensions based on top panel height
            top_panel_height = self.thumbnail_scroll.maximumHeight()
            
            # Calculate available height for thumbnails by accounting for all elements:
            # 1. Get the actual available height within the thumbnail scroll area
            # 2. Subtract scroll bar height
            # 3. Subtract minimal margins
            
            # Estimate scroll bar height (actual height when rendered)
            scrollbar_height = self.thumbnail_scroll.horizontalScrollBar().sizeHint().height()
            
            # Get control bar height (based on button sizes)
            button_height = self.btn_open_raw.sizeHint().height()
            control_bar_height = button_height + int(10 * self.scale_factor)  # Buttons + spacing
            
            # Get status label height
            status_label_height = self.status_label.maximumHeight()

            # Calculate available height by subtracting scroll bar
            available_height = int(0.9*top_panel_height) - int(1.5*scrollbar_height) - control_bar_height - status_label_height
            
            # Use fixed small margins for proper spacing
            side_margins = int(5 * self.scale_factor)  # Small margins on sides
            top_bottom_margins = int(3 * self.scale_factor)  # Minimal margins on top/bottom
            
            # Calculate maximum image height after accounting for all UI elements
            image_max_height = available_height - (2 * top_bottom_margins)
            
            # Ensure minimum height is maintained for usability
            image_max_height = max(image_max_height, int(100 * self.scale_factor))
            
            # Ensure we don't exceed screen height constraints
            screen_height_limit = int(self.screen_height * 0.25)  # Limit to 25% of screen height
            image_max_height = min(image_max_height, screen_height_limit)
            
            # Calculate aspect ratio
            aspect_ratio = width / height  # width / height
            
            # Determine thumbnail image dimensions based on aspect ratio
            if aspect_ratio > 1:  # Landscape
                image_width = int(min(image_max_height * aspect_ratio, self.screen_width * 0.15))  # Limit width to 15% of screen
                image_height = image_max_height
            else:  # Portrait or square
                image_height = image_max_height
                image_width = int(image_height * aspect_ratio)
            
            # Apply minimum size constraints
            image_width = max(image_width, int(100 * self.scale_factor))
            image_height = max(image_height, int(100 * self.scale_factor))
            
            # Ensure image height doesn't exceed calculated maximum
            image_height = min(image_height, image_max_height)
            
            # Scale image to thumbnail size
            scaled_q_image = q_image.scaled(
                image_width, image_height,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # Create container widget for image + overlay label
            thumbnail_container = QWidget()
            thumbnail_container.setFixedWidth(image_width + (2 * side_margins))
            thumbnail_container.setFixedHeight(image_height + (2 * top_bottom_margins))
            
            # Create grid layout for container (allows overlay positioning)
            thumbnail_layout = QGridLayout(thumbnail_container)
            thumbnail_layout.setAlignment(Qt.AlignTop)
            thumbnail_layout.setSpacing(0)
            thumbnail_layout.setContentsMargins(side_margins, top_bottom_margins, side_margins, top_bottom_margins)
            
            # Create image label
            image_label = QLabel()
            image_label.setPixmap(QPixmap.fromImage(scaled_q_image))
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(image_width, image_height)
            
            # Create LUT name label with improved overlay styling
            name_label = QLabel(lut_name)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setWordWrap(True)
            name_label.setToolTip(lut_name)
            name_label.setFixedSize(image_width, int(40 * self.scale_factor))  # Increased height for better visibility
            
            # Use a more visible font - smaller but bold
            font = QFont()
            font.setPointSize(int(3 * self.scale_factor))
            font.setBold(True)
            name_label.setFont(font)
            
            # Add image to grid layout
            thumbnail_layout.addWidget(image_label, 0, 0)
            
            # Add label to grid layout at the same position (overlay), align to bottom
            thumbnail_layout.addWidget(name_label, 0, 0, Qt.AlignBottom)
            
            # Set adaptive stylesheet for container - remove extra margins
            thumbnail_container.setStyleSheet(f'''
                QWidget {{
                    background-color: #34495e;
                    border: 1px solid #455a64;
                    border-radius: {int(6 * self.scale_factor)}px;
                    margin: 0;
                    padding: 0;
                }}
                QWidget:hover {{
                    background-color: #455a64;
                }}
            ''')
            
            # Set stylesheet for the overlay label - improve visibility
            name_label.setStyleSheet(f'''
                QLabel {{
                    background-color: rgba(0, 0, 0, 0.85);  /* Darker semi-transparent background for better visibility */
                    color: white;
                    padding: {int(8 * self.scale_factor)}px;
                    border-bottom-left-radius: {int(6 * self.scale_factor)}px;
                    border-bottom-right-radius: {int(6 * self.scale_factor)}px;
                    font-weight: bold;
                }}
            ''')
            
            # Connect thumbnail click to show preview
            thumbnail_container.mousePressEvent = lambda event, name=lut_name: self.show_preview_by_name(name)
            
            return thumbnail_container
            
        except Exception as e:
            print(f"Error creating thumbnail for {lut_name}: {e}")
            return None
    
    def add_converted_image(self, lut_name, image_data):
        """Add a converted image to the list and create thumbnail"""
        self.converted_images[lut_name] = image_data
        
        # Remove default message if it's the first image
        if len(self.converted_images) == 1:
            self.default_thumbnail_label.hide()
            self.default_thumbnail_label.setParent(None)  # Remove from layout
        
        # Create thumbnail
        thumbnail = self.create_thumbnail(lut_name, image_data)
        if thumbnail:
            self.thumbnails[lut_name] = thumbnail
            self.thumbnail_layout.addWidget(thumbnail)
        
        # Create list item with checkbox
        item = QListWidgetItem(lut_name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)  # Default to unchecked
        self.lut_list.addItem(item)
        
        # Show preview of the first item automatically
        if len(self.converted_images) == 1:
            self.show_preview(item)
    
    def show_preview(self, item):
        """Show preview of the selected image"""
        lut_name = item.text()
        if lut_name in self.converted_images:
            image_data = self.converted_images[lut_name]
            
            # Convert numpy array to QPixmap
            try:
                # Ensure array is contiguous and in correct format
                if not image_data.flags.contiguous:
                    image_data = np.ascontiguousarray(image_data)
                
                height, width, channel = image_data.shape
                bytes_per_line = channel * width
                
                q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
            except Exception as e:
                print(f"Error creating QImage: {e}")
                print(f"Image data shape: {image_data.shape}, dtype: {image_data.dtype}")
                return
            
            # Scale pixmap to fit preview area
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)
    
    def conversion_finished(self):
        """Handle conversion completion"""
        self.progress_bar.setVisible(False)
        
        # Re-enable buttons
        self.btn_convert.setEnabled(True)
        self.btn_open_raw.setEnabled(True)
        self.btn_select_luts.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.apply_exposure_btn.setEnabled(True)  # Enable exposure apply button
        
        self.status_label.setText(f"Conversion completed: {len(self.converted_images)} images")
    
    def on_exposure_slider_changed(self, value):
        """Handle exposure slider value changes"""
        # Convert slider value (-30 to 30) to EV (-3.0 to 3.0)
        self.manual_exposure = value / 10.0
        # Update value label
        self.exposure_value_label.setText(f"Value: {self.manual_exposure:.1f} EV")
    
    def apply_manual_exposure(self):
        """Re-convert images with manual exposure applied"""
        if not self.current_raw_path or not self.lut_folder:
            return
        
        # Clear previous results
        self.lut_list.clear()
        self.converted_images.clear()
        self.selected_images.clear()
        self.preview_label.setText("Converting with exposure adjustment...")
        
        # Clear thumbnails
        for thumbnail in self.thumbnails.values():
            thumbnail.hide()
            thumbnail.setParent(None)  # Remove from layout
        self.thumbnails.clear()
        
        # Restore default message
        if not hasattr(self, 'default_thumbnail_label') or not self.default_thumbnail_label.parent():
            self.default_thumbnail_label = QLabel("Converting with exposure adjustment...")
            self.default_thumbnail_label.setAlignment(Qt.AlignCenter)
            self.default_thumbnail_label.setMinimumSize(400, 100)
            self.thumbnail_layout.addWidget(self.default_thumbnail_label)
        
        # Disable buttons during conversion
        self.btn_convert.setEnabled(False)
        self.btn_open_raw.setEnabled(False)
        self.btn_select_luts.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.apply_exposure_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start worker thread with manual exposure
        self.conversion_worker = ConversionWorker(
            self.current_raw_path, self.lut_folder, self.manual_exposure
        )
        self.conversion_worker.progress_signal.connect(self.update_progress)
        self.conversion_worker.result_signal.connect(self.add_converted_image)
        self.conversion_worker.error_signal.connect(self.show_error)
        self.conversion_worker.finished_signal.connect(self.conversion_finished)
        self.conversion_worker.start()
    
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.conversion_finished()
    
    def handle_item_changed(self, item):
        """Handle item check state changes"""
        pass  # No need to track, we'll check states on demand

    def toggle_select_all(self, state):
        """Toggle selection (check/uncheck) of all items"""
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        for i in range(self.lut_list.count()):
            item = self.lut_list.item(i)
            item.setCheckState(check_state)
    
    def export_selected(self):
        """Export checked images"""
        # Get all checked items
        checked_items = []
        for i in range(self.lut_list.count()):
            item = self.lut_list.item(i)
            if item.checkState() == Qt.Checked:
                checked_items.append(item)

        if not checked_items:
            QMessageBox.warning(self, "Warning", "No images checked for export")
            return

        # Ask for export folder
        export_folder = QFileDialog.getExistingDirectory(
            self, "Select Export Folder"
        )

        if export_folder:
            base_name = os.path.splitext(os.path.basename(self.current_raw_path))[0]

            # Export each checked image
            for item in checked_items:
                lut_name = item.text()
                if lut_name in self.converted_images:
                    image_data = self.converted_images[lut_name]
                    export_path = os.path.join(
                        export_folder,
                        f"{base_name}_{lut_name}.jpg"
                    )

                    try:
                        iio.imwrite(export_path, image_data, quality=90)
                    except Exception as e:
                        QMessageBox.critical(
                            self, "Export Error",
                            f"Failed to export {lut_name}: {str(e)}"
                        )

            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {len(checked_items)} images"
            )
    
    def update_dynamic_padding(self, event):
        """Update padding dynamically when window is resized"""
        # Call the default resize event
        super().resizeEvent(event)
        
        # Calculate padding as 3% of the actual panel sizes
        padding_percentage = 0.03  # 3%
        
        # Get actual sizes of the panels
        try:
            # For Preview panel
            if hasattr(self, 'preview_layout') and self.preview_layout.parentWidget():
                preview_widget = self.preview_layout.parentWidget()
                if preview_widget and preview_widget.isVisible():
                    preview_size = preview_widget.size()
                    
                    # Base padding calculations
                    pad_left = int(preview_size.width() * padding_percentage)
                    pad_right = int(preview_size.width() * padding_percentage)
                    pad_bottom = int(preview_size.height() * padding_percentage)
                    
                    # For top padding, account for the group box title height
                    title_padding = int(25 * self.scale_factor)
                    pad_top = int(preview_size.height() * padding_percentage) + title_padding
                    
                    self.preview_layout.setContentsMargins(pad_left, pad_top, pad_right, pad_bottom)
            
            # For Converted Images panel
            if hasattr(self, 'list_layout') and self.list_layout.parentWidget():
                list_widget = self.list_layout.parentWidget()
                if list_widget and list_widget.isVisible():
                    list_size = list_widget.size()
                    
                    # Base padding calculations
                    pad_left = int(list_size.width() * padding_percentage)
                    pad_right = int(list_size.width() * padding_percentage)
                    pad_bottom = int(list_size.height() * padding_percentage)
                    
                    # For top padding, account for the group box title height
                    title_padding = int(25 * self.scale_factor)
                    pad_top = int(list_size.height() * padding_percentage) + title_padding
                    
                    self.list_layout.setContentsMargins(pad_left, pad_top, pad_right, pad_bottom)
        except Exception as e:
            # If there's an error (e.g., widgets not ready yet), just ignore it
            pass

    def closeEvent(self, event):
        """Handle window close event"""
        if self.conversion_worker and self.conversion_worker.isRunning():
            self.conversion_worker.stop()
            self.conversion_worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RawConverterApp()
    
    # Open in maximized mode by default
    window.showMaximized()
    
    sys.exit(app.exec_())
