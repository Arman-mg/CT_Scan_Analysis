# 🫁 CT Scan Analysis & Lung Infection Detection

This project provides a **machine learning-driven pipeline** to analyze **CT scan images** for lung segmentation and detection of **Ground Glass Opacities (GGO)** — a critical indicator of infections like **COVID-19**. The code integrates clustering algorithms and medical image processing techniques to **quantify infection severity** in each lung.

---

## 📌 Features

* **CT Scan Preprocessing**: Reads `.nii` files (Neuroimaging Informatics Technology Initiative format) and visualizes CT slices.
* **K-Means Clustering**: Segments CT images into regions of interest for lung and tissue detection.
* **DBSCAN Clustering**: Identifies and isolates **lungs** using density-based spatial clustering.
* **Ground Glass Opacities (GGO) Detection**: Highlights infection-prone regions in the lungs.
* **Infection Measurement**: Quantifies severity separately for **left** and **right** lungs.
* **Visualizations**: Generates histograms, segmented images, and infection overlays.

---

## 🧠 Methodology Highlights

1. **Data Input & Visualization**

   * Loads CT scans in `.nii` format using **NiBabel**.
   * Visualizes individual slices to analyze lung structures.

2. **Image Segmentation**

   * Uses **K-Means clustering** to reduce image colors and segment anatomical regions.
   * Refines segmentation to focus on lungs using **DBSCAN** clustering.

3. **GGO Detection**

   * Extracts Ground Glass Opacities — regions often linked to early infection patterns.

4. **Infection Quantification**

   * Calculates severity scores for left and right lungs individually.

---

## 🛠️ Dependencies

Install required Python libraries via `pip`:

```bash
pip install numpy nibabel matplotlib scikit-learn
```

**Libraries Used:**

* [NumPy](https://numpy.org/) — numerical operations
* [NiBabel](https://nipy.org/nibabel/) — `.nii` CT scan file reading
* [Matplotlib](https://matplotlib.org/) — visualization
* [Scikit-learn](https://scikit-learn.org/) — clustering (K-Means, DBSCAN)
* [itertools](https://docs.python.org/3/library/itertools.html) — combinatorial tools

---

## ▶️ Usage

```bash
python main.py
```

### Workflow Steps:

* Read `.nii` CT scan files.
* Perform **K-Means** segmentation to extract lung structures.
* Use **DBSCAN** clustering to isolate lungs.
* Detect **Ground Glass Opacities**.
* Output infection severity per lung.

---

## 📊 Example Output

* **Lung segmentation masks**
* **Highlighted Ground Glass Opacities**
* Infection severity scores:

  ```
  Left lung infection severity  : 0.27
  Right lung infection severity : 0.35
  ```

---

## 📂 Project Structure

```
CT_Scan_Analysis/
├── sub/
│   └── CTscan.py          # Core processing class
├── main.py                # Main execution script
└── README.md
└── Report.pdf
```