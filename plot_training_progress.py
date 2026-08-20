import os
import re
import matplotlib.pyplot as plt
from pypdf import PdfReader

def extract_and_plot():
    # PDF paths to check
    pdf_path = "/Users/bogiekissiov/Desktop/Training progression 2.pdf"
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find PDF at {pdf_path}")
    
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Match lines like:
    # "Loss: 0.29905 (BC: 0.1286, Clean: 0.1061)"
    # Note: text formatting / newlines can break lines across pages, so clean whitespace
    pattern = r"Loss:\s*([0-9.]+)\s*\(\s*BC:\s*([0-9.]+)\s*,\s*Clean:\s*([0-9.]+)\s*\)"
    
    matches = re.findall(pattern, full_text)
    
    if not matches:
        print("No loss values matched. Checking cleaned text...")
        cleaned_text = re.sub(r'\s+', ' ', full_text)
        matches = re.findall(r"Loss:\s*([0-9.]+)\s*\(\s*BC:\s*([0-9.]+)\s*,\s*Clean:\s*([0-9.]+)\s*\)", cleaned_text)

    total_loss = [float(m[0]) for m in matches]
    bc_loss = [float(m[1]) for m in matches]
    clean_loss = [float(m[2]) for m in matches]
    steps = list(range(1, len(total_loss) + 1))

    print(f"Total data points extracted: {len(total_loss)}")

    # Plotting
    plt.figure(figsize=(14, 7), dpi=300)
    plt.plot(steps, total_loss, label="Total Loss", color="#2563eb", linewidth=1.5, alpha=0.9)
    plt.plot(steps, bc_loss, label="Baseline Correction (BC) Loss", color="#ea580c", linewidth=1.5, alpha=0.9)
    plt.plot(steps, clean_loss, label="Clean Loss", color="#16a34a", linewidth=1.5, alpha=0.9)

    plt.title("Training Progression Losses per Step", fontsize=15, fontweight="bold", pad=12)
    plt.xlabel("Step", fontsize=12, fontweight="medium")
    plt.ylabel("Loss", fontsize=12, fontweight="medium")
    plt.yscale("log")  # Using log scale can often be useful, but standard grid is great
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=12, loc="upper right", frameon=True)
    plt.tight_layout()

    output_path = "/Users/bogiekissiov/Desktop/trainingprogress.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    extract_and_plot()
