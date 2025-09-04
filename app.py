import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import traceback

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Define your actual model architecture
# ----------------------------
class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate the size after convolutional layers
        self._to_linear = None
        self._calculate_conv_output_size((1, 1, 180))

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _calculate_conv_output_size(self, shape):
        batch_size = 1
        x = torch.rand(batch_size, *shape[1:])
        x = self.features(x)
        self._to_linear = x.shape[1] * x.shape[2]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# ----------------------------
# Model loading with detailed diagnostics
# ----------------------------
def load_model_with_diagnostics():
    """Load model with detailed error reporting"""
    MODEL_PATH = "ecg_cnn_model.pth"
    
    print(f"Attempting to load model from: {MODEL_PATH}")
    
    # Check if file exists
    if not os.path.exists(MODEL_PATH):
        error_msg = f"Model file not found at: {MODEL_PATH}"
        print(error_msg)
        return None, error_msg
    
    print("✓ Model file found")
    
    try:
        # Load the model file
        loaded_data = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print(f"✓ File loaded successfully. Object type: {type(loaded_data)}")
        
        # Create model instance
        model = ECGCNN(num_classes=5)
        
        # Handle different .pth file formats
        if isinstance(loaded_data, dict):
            print("✓ Loaded object is a dictionary")
            print(f"  Dictionary keys: {list(loaded_data.keys())}")
            
            # Try to find the state dict
            state_dict = None
            possible_keys = ['model_state_dict', 'state_dict', 'model']
            
            for key in possible_keys:
                if key in loaded_data:
                    state_dict = loaded_data[key]
                    print(f"  Found state dict in key: '{key}'")
                    break
            
            if state_dict is None:
                print("  Using entire dictionary as state dict")
                state_dict = loaded_data
        else:
            print("✓ Using loaded object as state dict")
            state_dict = loaded_data
        
        # Load state dict
        model.load_state_dict(state_dict)
        print("✓ State dict loaded successfully")
        
        model.to(device)
        model.eval()
        print("✓ Model loaded and ready for inference")
        return model, None
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(error_msg)
        print("Full traceback:")
        traceback.print_exc()
        return None, error_msg

# ----------------------------
# Initialize model and components
# ----------------------------
model, load_error = load_model_with_diagnostics()

# Create label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['N', 'L', 'R', 'V', 'A'])

# Annotation mapping
annotation_map = {
    'N': 'Normal Beat',
    'L': 'Left Bundle Branch Block', 
    'R': 'Right Bundle Branch Block',
    'V': 'Premature Ventricular Contraction',
    'A': 'Atrial Premature Beat'
}

# ----------------------------
# ECG preprocessing function
# ----------------------------
def preprocess_ecg(ecg_signal):
    """Preprocess ECG signal for model input"""
    # Ensure we have exactly 180 samples (model requirement)
    if len(ecg_signal) != 180:
        ecg_signal = np.interp(
            np.linspace(0, len(ecg_signal) - 1, 180),
            np.arange(len(ecg_signal)),
            ecg_signal
        )
    
    # Normalize the signal
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    return torch.FloatTensor(ecg_signal).unsqueeze(0).unsqueeze(0).to(device)

# ----------------------------
# Main analysis function - NO DUMMY DATA
# ----------------------------
def analyze_ecg(ecg_input):
    """Main function to analyze ECG input"""
    if model is None:
        error_text = f"##Model Loading Failed\n\n"
        error_text += f"**Error:** {load_error}\n\n"
        error_text += "**Please check:**\n"
        error_text += "1. File exists at: `ecg_cnn_model.pth`\n"
        error_text += "2. File is a valid PyTorch model\n"
        error_text += "3. Model architecture matches expected format\n"
        return error_text, None
    
    try:
        # Parse and validate input
        if not ecg_input.strip():
            return "**Oloche AI Assistant:** Please enter ECG values", None
        
        # Parse ECG values
        ecg_values = []
        for val in ecg_input.split(','):
            val = val.strip()
            if val:
                try:
                    ecg_values.append(float(val))
                except ValueError:
                    return f"**Oloche AI Assistant:** Invalid value '{val}'. Please enter only numbers separated by commas.", None
        
        if len(ecg_values) < 10:
            return "**Oloche AI Assistant:** Please enter at least 10 ECG values", None
        
        ecg_signal = np.array(ecg_values)
        
        # Preprocess and make prediction
        processed = preprocess_ecg(ecg_signal)
        
        with torch.no_grad():
            output = model(processed)
            probabilities = F.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1)
            
            class_idx = pred_class.item()
            class_name = label_encoder.inverse_transform([class_idx])[0]
            confidence = probabilities[0, class_idx].item()
        
        # Create visualization with white background and black text
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(ecg_signal, color='#3498db', linewidth=2)
        ax.set_title(f"ECG Signal - {annotation_map.get(class_name, class_name)}", 
                    fontsize=16, fontweight='bold', pad=20, color='black')
        ax.set_xlabel("Time (samples)", fontsize=14, color='black')
        ax.set_ylabel("Amplitude", fontsize=14, color='black')
        ax.tick_params(colors='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Add prediction info
        prediction_text = f"Prediction: {annotation_map.get(class_name, class_name)}\nConfidence: {confidence:.2%}"
        ax.text(0.02, 0.98, prediction_text, transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#3498db'), 
               fontsize=12, color='black')
        
        plt.tight_layout()
        
        # Prepare results
        result = f"## ECG Analysis Results\n\n"
        result += f"**Prediction:** {annotation_map.get(class_name, class_name)}\n"
        result += f"**Confidence:** {confidence:.2%}\n\n"
        result += "**Probability Distribution:**\n"
        
        for i, cls in enumerate(label_encoder.classes_):
            prob = probabilities[0, i].item()
            result += f"- {annotation_map.get(cls, cls)}: {prob:.2%}\n"
        
        result += f"\n**Signal Characteristics:**\n"
        result += f"- Length: {len(ecg_signal)} samples\n"
        result += f"- Mean: {np.mean(ecg_signal):.6f}\n"
        result += f"- Standard Deviation: {np.std(ecg_signal):.6f}\n"
        result += f"- Range: {np.ptp(ecg_signal):.6f}"
        
        result += f"\n\n---\n*Deep Learning Project by Eije, Oloche Celestine*"
        
        return result, fig
        
    except Exception as e:
        return f"**Oloche AI Assistant encountered an error during analysis:** {str(e)}", None

# ----------------------------
# Gradio UI with Enhanced Layout
# ----------------------------
custom_css = """
.gradio-container {
    max-width: 100% !important;
    margin: 0 auto !important;
    padding: 20px !important;
    background: #000000 !important; /* Dark gray background */
    color: #ffffff !important; /* White text */
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    margin-bottom: 15px;
    padding: 25px;
    background: #2d2d2d !important; /* Medium gray background */
    border-radius: 10px;
    border: 1px solid #444 !important;
}
.header h1 {
    font-size: 2.8em !important;
    font-weight: 900 !important;
    margin-bottom: 5px !important;
    color: #ffffff !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}
.clean-input textarea {
    background-color: #2d2d2d !important;
    border: 2px solid #555 !important;
    color: #ffffff !important;
    font-size: 14px !important;
    padding: 12px !important;
    border-radius: 8px !important;
    font-family: 'Consolas', 'Monaco', monospace !important;
    transition: border-color 0.3s ease;
}
.clean-input textarea:focus {
    border-color: #3498db !important;
    outline: none;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
}
.clean-input label {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    margin-bottom: 8px !important;
}
.button-primary {
    background: #3498db !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(52, 152, 219, 0.3);
}
.button-primary:hover {
    background: #2980b9 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(52, 152, 219, 0.4);
}
.button-primary:active {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(52, 152, 219, 0.3);
}
.button-secondary {
    background: #7f8c8d !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(127, 140, 141, 0.3);
}
.button-secondary:hover {
    background: #636e72 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(127, 140, 141, 0.4);
}
.button-secondary:active {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(127, 140, 141, 0.3);
}
.column {
    background: #2d2d2d !important;
    padding: 20px !important;
    margin: 10px !important;
    border-radius: 10px;
    border: 1px solid #444 !important;
}
.label {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin-bottom: 15px !important;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
}
.markdown-text {
    background: #2d2d2d !important;
    padding: 20px !important;
    border: 1px solid #444 !important;
    border-radius: 10px;
    color: #ffffff !important;
    font-size: 16px !important;
}
.plot-container {
    background: white !important;
    padding: 20px !important;
    border: 1px solid #444 !important;
    border-radius: 10px;
    width: 100% !important;
}
.gr-button {
    margin: 8px !important;
}
.gr-markdown {
    color: #ffffff !important;
}
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4 {
    color: #ffffff !important;
}
.gr-markdown p, .gr-markdown li {
    color: #e0e0e0 !important;
}
.footer {
    text-align: left;
    margin-top: 20px;
    padding: 20px;
    border-top: 2px solid #444;
    font-size: 14px;
    color: #e0e0e0 !important;
    font-weight: 400;
    background: #2d2d2d !important;
    border-radius: 10px;
    border: 1px solid #444 !important;
}
.footer ul {
    margin: 0;
    padding-left: 20px;
}
.footer li {
    margin-bottom: 8px;
}
.error-box {
    background-color: #4a1c1c !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border: 1px solid #742424 !important;
    color: #e57373 !important;
}
.success-box {
    background-color: #1c3a2c !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border: 1px solid #2d5e45 !important;
    color: #68d391 !important;
}
.dark-container {
    background: #2d2d2d !important;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #444 !important;
}
.dark-container ul {
    margin: 10px 0;
    padding-left: 20px;
}
.dark-container li {
    margin-bottom: 8px;
    color: #e0e0e0;
}
"""

with gr.Blocks(css=custom_css, title="ECG Classification System") as demo:
    
    gr.Markdown("""
    <div class="header" style="text-align: center;">
    <h1 style="text-align: center; font-weight: 1,000 !important;">Oloche'S AI Cardiologist: ECG Arrhythmia Classifier</h1>
    </div>
    """)
    
    # Show model status
    if model is None:
        gr.Markdown(f"""
        <div class="error-box">
            <h3>Model Loading Failed</h3>
            <p>{load_error}</p>
            <p>Please check the console for detailed error messages.</p>
        </div>
        """)
    else:
        gr.Markdown("""
        <div class="success-box">
              <p>This system uses convolutional neural networks (CNNs) trained on datasets from the MIT-BIH Arrhythmia Database to detect cardiac arrhythmias through real-time ECG signal analysis. Kindly input 180 comma-separated numerical values representing an ECG signal, and the system will classify it into one of five distinct arrhythmia categories: Normal (N), Left Bundle Branch Block (L), Right Bundle Branch Block (R), Premature Ventricular Contraction (V), or Atrial Premature Beat (A).
              The model is reliable and achieves high performance in detecting abnormal cardiac rhythms and underlying heart diseases, serving as a quick-diagnostic tool for cardiologists, biomedical researchers, industry experts, and medical professionals.</p>
            <h5>Ready for ECG signal analysis</h5>
        </div>
        """)
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### Input ECG Data", elem_classes=["label"])
            
            ecg_input = gr.Textbox(
                label="ECG Signal Values",
                value="",
                lines=6,
                placeholder="Enter 180 comma-separated numerical values (e.g, -0.115, -0.712, 0.55)...",
                info="Provide your actual ECG data for analysis",
                interactive=model is not None,
                elem_classes=["clean-input"]
            )
            
            with gr.Row():
                analyze_btn = gr.Button("Analyze ECG", variant="primary", interactive=model is not None, elem_classes=["button-primary"])
                clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["button-secondary"])
                
            gr.Markdown("""
            <div class="dark-container">
                <p><strong>Input Requirements:</strong></p>
                <ul>
                    <li>180 comma-separated numerical values</li>
                    <li>Represents a single ECG heartbeat</li>
                    <li>Raw voltage readings (will be normalized)</li>
                </ul>
            </div>
            """)
            
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### ECG Data Output", elem_classes=["label"])
            output_text = gr.Markdown(
                label="Results",
                value="Enter your ECG data and click 'Analyze ECG' for real predictions." if model else "Model failed to load. Please check the error message above.",
                elem_classes=["markdown-text"]
            )
    
    # ECG Visualization - Full width at the bottom
    gr.Markdown("### ECG Signal Visualization", elem_classes=["label"])
    output_plot = gr.Plot(label="ECG Signal", elem_classes=["plot-container"])
    
    gr.Markdown("""
    <div class="footer">
        <p><strong>Classification Categories:</strong></p>
        <ul>
            <li><strong>N</strong>: Normal Beat</li>
            <li><strong>L</strong>: Left Bundle Branch Block</li>
            <li><strong>R</strong>: Right Bundle Branch Block</li>
            <li><strong>V</strong>: Premature Ventricular Contraction</li>
            <li><strong>A</strong>: Atrial Premature Beat</li>
        </ul>
        <br>
        <p>Researcher: Eije, Oloche Celestine</p>
    </div>
    """)
    
    # Button actions
    analyze_btn.click(
        fn=analyze_ecg,
        inputs=ecg_input,
        outputs=[output_text, output_plot]
    )
    
    clear_btn.click(
        fn=lambda: ("", None),
        inputs=[],
        outputs=[ecg_input, output_plot]
    )
    
    # Enter key support
    ecg_input.submit(
        fn=analyze_ecg,
        inputs=ecg_input,
        outputs=[output_text, output_plot]
    )

# ----------------------------
# Launch application
# ----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ECG Classification App - Using Oloche's Trained Model")
    print("=" * 60)
    
    if model is not None:
        print("✅ Model loaded successfully!")
        print(f"   Model: ECGCNN")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")
        print("   No dummy data - using your actual trained model")
    else:
        print("❌ Model failed to load")
        print(f"   Error: {load_error}")
    
    print("=" * 60)
    demo.launch(share=False)
