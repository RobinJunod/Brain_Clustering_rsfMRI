#%%
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import main

def select_file(label, file_type):
    """Opens a file dialog and updates the label with the selected file path."""
    file_path = filedialog.askopenfilename(
        title=f"Select {file_type}",
        filetypes=[("NIfTI files", "*.nii"), ("NIfTI GZ files", "*.nii.gz"), ("All files", "*.*")]
    )
    if file_path:
        label.config(text=file_path)

def select_output_folder(label):
    """Opens a folder dialog and updates the label with the selected folder path."""
    folder_path = filedialog.askdirectory(
        title="Select Output Folder"
    )
    if folder_path:
        label.config(text=folder_path)

def validate_selections():
    """Validates that all file and folder selections are made."""
    if not fmri_label.cget("text") or fmri_label.cget("text") == "No file selected":
        messagebox.showerror("Error", "Please select the resting state fMRI data.")
        return False
    if not roi_label.cget("text") or roi_label.cget("text") == "No file selected":
        messagebox.showerror("Error", "Please select the ROI file.")
        return False
    if not mask_label.cget("text") or mask_label.cget("text") == "No file selected":
        messagebox.showerror("Error", "Please select the brain mask file.")
        return False
    if not output_label.cget("text") or output_label.cget("text") == "No folder selected":
        messagebox.showerror("Error", "Please select the output folder.")
        return False
    return True

def start_computation():
    """Starts the computation after validating file and folder selections."""
    if not validate_selections():
        return
    
    log_message("Starting computation...")
    fmri_file = fmri_label.cget("text")
    roi_file = roi_label.cget("text")
    mask_file = mask_label.cget("text")
    output_folder = output_label.cget("text")
    
    log_message(f"fMRI File: {fmri_file}")
    log_message(f"ROI File: {roi_file}")
    log_message(f"Mask File: {mask_file}")
    log_message(f"Output Folder: {output_folder}")
    
    try:
        main.main(fmri_file, roi_file, mask_file, output_folder)
        log_message("Computation completed successfully!")
    except Exception as e:
        log_message(f"An error occurred: {e}")

def log_message(message):
    """Logs a message in the text box."""
    log_textbox.insert(tk.END, f"{message}\n")
    log_textbox.see(tk.END)  # Auto-scroll to the bottom


if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    root.title("Advanced File Selector with Computation")
    root.geometry("500x500")

    # Labels and buttons for selecting files
    fmri_label = tk.Label(root, text="No file selected", wraplength=450)
    fmri_button = tk.Button(root, text="Select Resting State fMRI Data", command=lambda: select_file(fmri_label, "Resting State fMRI Data"))

    roi_label = tk.Label(root, text="No file selected", wraplength=450)
    roi_button = tk.Button(root, text="Select ROI File", command=lambda: select_file(roi_label, "ROI File"))

    mask_label = tk.Label(root, text="No file selected", wraplength=450)
    mask_button = tk.Button(root, text="Select Brain Mask", command=lambda: select_file(mask_label, "Brain Mask"))

    output_label = tk.Label(root, text="No folder selected", wraplength=450)
    output_button = tk.Button(root, text="Select Output Folder", command=lambda: select_output_folder(output_label))

    # Button to start the computation
    go_button = tk.Button(root, text="Start Parcellation", command=start_computation)

    # Textbox for logging messages
    log_textbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10)
    log_textbox.config(state=tk.NORMAL)  # Make the textbox editable so we can log messages

    # Layout: Arrange buttons and labels in the window
    fmri_button.pack(pady=5)
    fmri_label.pack(pady=5)

    roi_button.pack(pady=5)
    roi_label.pack(pady=5)

    mask_button.pack(pady=5)
    mask_label.pack(pady=5)

    output_button.pack(pady=5)
    output_label.pack(pady=5)

    go_button.pack(pady=10)

    log_textbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Start the GUI event loop
    root.mainloop()
# %%
