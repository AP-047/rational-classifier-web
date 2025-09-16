## Rational Function Digit Classifier

This project implements a **handwritten digit recognition** web application using rational functions (ratios of polynomials) trained on the MNIST dataset. Users can draw a digit (0–9) in the canvas and click **Predict** to see the model’s prediction in real time.

---

### Live Demo

[Visit: Website
](https://ap-047.github.io/rational-classifier-web/)
---

### Features

- **Pure JavaScript inference**: No backend required—PCA and classifier logic run entirely in the browser.  
- **Rational function classifiers**: Each digit is recognized by a ratio of two polynomial functions.  
- **Custom PCA transform**: Dimensionality reduction from 784 (28×28) to 25 components.  
- **Interactive canvas**: Draw with white ink on a black background, clear and predict with a click.  

---

### Model Performance

> **Note:** This model achieves approximately **80% accuracy** on the MNIST test set due to limited computational resources during training. As a result, some predictions may be incorrect.

---

### How to Use

1. Clone or download this repository.  
2. Serve the project folder over any static HTTP server (e.g., `python -m http.server 8000`).  
3. Open `http://localhost:8000` in your browser.  
4. Draw a digit in the black canvas and click **Predict**.  

---

### Future Improvements

- Unfortunately, I no longer have access to the model that achieved the highest accuracy (around 95%), and I currently do not have lab access at the university. With the limited resources available to me, I will focus on optimizing training and compute usage to push the current model’s accuracy beyond 80%.
- Experiment with deeper polynomial degrees or alternative classifiers.

---

### License
## Attribution & Credits
### Academic Project Background
This rational function classifier was developed as part of a special project (3rd semester) at the **Chair of Applied Mathematics, Faculty of Civil Engineering, Bauhaus-Universität Weimar** under the supervision of:
- **Prof. Dr. rer. nat. Björn Rüffer** (Project Supervisor)  
- **Dr. rer. nat. habil. Michael Schönlein** (Project Examiner)

**Team Approach:** Each member independently developed the rational function classifier using different frameworks (SageMath, SciPy, Gurobi), and the best-performing implementation was selected. The model used in this repository — developed by me using Gurobi.

### Usage & Citation
If you use this model or methodology in your work, please cite:
```plaintext
RClass—Classification by Rational Approximation (2025)
Developed at Bauhaus-Universität Weimar under supervision of Prof. Dr. Björn Rüffer
Original team: Omar Ghariani, Helen Dawit Weldemichael, Ajay Patil
GitHub implementation by: [Ajay Patil](https://github.com/AP-047/rclass)
```
