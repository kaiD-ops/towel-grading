
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
import threading
from evaluate import ModelEvaluator
from config import MODEL_SAVE_PATH, LABEL_TO_GRADE


class TowelGradingGUI:
    """GUI for towel quality grading predictions."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Towel Quality Grading System")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Set style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Initialize model
        self.evaluator = None
        self.current_image_path = None
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the user interface."""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ============ Header ============
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame,
            text="🎯 Towel Quality Grading System",
            font=("Arial", 18, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=10)
        
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(side=tk.RIGHT, padx=10)
        
        self.status_label = ttk.Label(
            status_frame,
            text="Loading model...",
            font=("Arial", 10),
            foreground="orange"
        )
        self.status_label.pack()
        
        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5
        )
        
        # ============ Content Area ============
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Left panel - Image display
        left_panel = ttk.LabelFrame(content_frame, text="Image Preview", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_panel.rowconfigure(0, weight=1)
        left_panel.columnconfigure(0, weight=1)
        
        # Image canvas
        self.image_canvas = tk.Canvas(
            left_panel,
            bg="#f0f0f0",
            width=400,
            height=400,
            relief=tk.RIDGE,
            borderwidth=2
        )
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Placeholder text
        self.image_canvas.create_text(
            200, 200,
            text="No image selected\nClick 'Upload Image' to start",
            font=("Arial", 12),
            fill="#999999"
        )
        
        # Right panel - Results and controls
        right_panel = ttk.Frame(content_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_panel.columnconfigure(0, weight=1)
        
        # Upload button
        button_frame = ttk.Frame(right_panel)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        button_frame.columnconfigure(0, weight=1)
        
        self.upload_btn = ttk.Button(
            button_frame,
            text="📁 Upload Image",
            command=self.upload_image
        )
        self.upload_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        self.predict_btn = ttk.Button(
            button_frame,
            text="🔍 Predict Grade",
            command=self.predict_grade,
            state=tk.DISABLED
        )
        self.predict_btn.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Prediction result frame
        result_frame = ttk.LabelFrame(right_panel, text="Prediction Result", padding="15")
        result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        result_frame.columnconfigure(0, weight=1)
        
        # Grade display
        grade_display_frame = ttk.Frame(result_frame)
        grade_display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(grade_display_frame, text="Predicted Grade:", font=("Arial", 11, "bold")).pack()
        
        self.grade_label = ttk.Label(
            grade_display_frame,
            text="---",
            font=("Arial", 32, "bold"),
            foreground="#3498db"
        )
        self.grade_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(
            grade_display_frame,
            text="Confidence: ---",
            font=("Arial", 11)
        )
        self.confidence_label.pack()
        
        # Separator
        ttk.Separator(result_frame, orient=tk.HORIZONTAL).grid(
            row=1, column=0, sticky=(tk.W, tk.E), pady=10
        )
        
        # Score bars
        scores_frame = ttk.LabelFrame(result_frame, text="Confidence Scores", padding="10")
        scores_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        scores_frame.columnconfigure(1, weight=1)
        
        self.score_bars = {}
        colors = {'Grade A': '#2ecc71', 'Grade B': '#f39c12', 'Grade C': '#e74c3c'}
        
        for idx, grade_name in enumerate(['Grade A', 'Grade B', 'Grade C']):
            # Label
            ttk.Label(scores_frame, text=f"{grade_name}:").grid(row=idx, column=0, sticky=tk.W, pady=5)
            
            # Progress bar
            progress = ttk.Progressbar(
                scores_frame,
                length=150,
                mode='determinate',
                value=0,
                maximum=100
            )
            progress.grid(row=idx, column=1, sticky=(tk.W, tk.E), padx=5)
            self.score_bars[grade_name] = progress
            
            # Percentage label
            percent_label = ttk.Label(scores_frame, text="0%", width=5)
            percent_label.grid(row=idx, column=2, sticky=tk.E, padx=5)
            self.score_bars[f"{grade_name}_percent"] = percent_label
        
        # File info
        info_frame = ttk.LabelFrame(result_frame, text="File Information", padding="10")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(info_frame, text="File:").grid(row=0, column=0, sticky=tk.W)
        self.file_label = ttk.Label(info_frame, text="None", foreground="#666")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(info_frame, text="Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.size_label = ttk.Label(info_frame, text="---", foreground="#666")
        self.size_label.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Footer with instructions
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5)
        )
        
        instructions = ttk.Label(
            footer_frame,
            text="💡 Instructions: 1) Upload a towel image  2) Click 'Predict Grade' to classify  3) View results and confidence scores",
            font=("Arial", 9),
            foreground="#666"
        )
        instructions.pack(side=tk.LEFT, padx=10)
    
    def load_model(self):
        """Load the trained model in a separate thread."""
        def _load():
            try:
                # Check if model exists
                model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.h5')
                if not os.path.exists(model_path):
                    model_path = os.path.join(MODEL_SAVE_PATH, 'towel_grading_model.h5')
                
                if not os.path.exists(model_path):
                    self.status_label.config(
                        text="❌ Model not found. Please train the model first.",
                        foreground="red"
                    )
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error",
                        "Trained model not found!\n\nPlease train the model first using:\npython train.py"
                    ))
                    return
                
                self.evaluator = ModelEvaluator(model_path)
                self.status_label.config(
                    text="✓ Model loaded successfully",
                    foreground="green"
                )
                self.upload_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                self.status_label.config(
                    text=f"❌ Error loading model: {str(e)[:30]}...",
                    foreground="red"
                )
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Failed to load model:\n{str(e)}"
                ))
        
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
    
    def upload_image(self):
        """Open file dialog to select an image."""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select a towel image",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.update_file_info(file_path)
    
    def display_image(self, image_path):
        """Display the selected image on the canvas."""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Resize to fit canvas (max 400x400)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Clear and update canvas
            self.image_canvas.delete("all")
            self.image_canvas.create_image(200, 200, image=photo)
            self.image_canvas.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def update_file_info(self, file_path):
        """Update file information display."""
        try:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            
            self.file_label.config(text=file_name)
            self.size_label.config(text=f"{file_size:.2f} MB")
        except Exception as e:
            self.file_label.config(text="Error reading file")
    
    def predict_grade(self):
        """Make prediction on the selected image."""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if not self.evaluator:
            messagebox.showerror("Error", "Model not loaded. Please wait.")
            return
        
        # Disable button during prediction
        self.predict_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Predicting...", foreground="blue")
        self.root.update()
        
        def _predict():
            try:
                # Make prediction
                grade, confidence_scores = self.evaluator.predict_single_image(
                    self.current_image_path
                )
                
                # Update UI in main thread
                self.root.after(0, lambda: self.display_results(grade, confidence_scores))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Prediction Error",
                    f"Failed to make prediction:\n{str(e)}"
                ))
            finally:
                self.predict_btn.config(state=tk.NORMAL)
                self.status_label.config(text="✓ Ready", foreground="green")
        
        # Run prediction in separate thread to avoid freezing UI
        thread = threading.Thread(target=_predict, daemon=True)
        thread.start()
    
    def display_results(self, grade, confidence_scores):
        """Display prediction results."""
        # Update grade label
        self.grade_label.config(text=grade)
        
        # Get confidence for predicted grade
        grade_idx = int(grade.split()[-1]) - ord('A')
        confidence = confidence_scores[grade_idx]
        self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
        
        # Update score bars
        grades = ['Grade A', 'Grade B', 'Grade C']
        for idx, grade_name in enumerate(grades):
            score_value = confidence_scores[idx] * 100
            self.score_bars[grade_name].config(value=score_value)
            self.score_bars[f"{grade_name}_percent"].config(text=f"{score_value:.1f}%")


def main():
    """Main entry point for the GUI."""
    root = tk.Tk()
    app = TowelGradingGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
