# PolyhedronNet: Representation Learning for Polyhedra with Surface-attributed Graph

This is the **official repository** for our **ICLR 2025 paper**:  
**"PolyhedronNet: Representation Learning for Polyhedra with Surface-attributed Graph"**

## 🔹 Overview

PolyhedronNet introduces an innovative representation learning framework for **3D geometries**, utilizing **surface-attributed graphs** to efficiently encode complex 3D structures. This approach extends polygon-based graph learning techniques into the polyhedral domain, enabling a wide range of applications in **computational geometry, urban modeling, and shape analysis**.

For **2D polygon/multipolygon encoding**, please refer to our related work:  
🔗 [**PolygonGNN Repository**](https://github.com/dyu62/PolyGNN)

## 📜 Paper

[**Read the Paper**](https://openreview.net/pdf?id=BpyHIrpUOL)

## 📂 Repository Structure

- `data/` – Contains `.pkl` files required for running experiments. Additionally, we provide code to generate polyhedron datasets for those interested in creating or modifying datasets.

## 🚀 Getting Started

### Dependencies

To install the required packages, run the command lines provided in the `install` file.

### Training

To start training, run the following command in the terminal:

```bash
python train.py
```
### Cite
```
@inproceedings{yupolyhedronnet,
  title={PolyhedronNet: Representation Learning for Polyhedra with Surface-attributed Graph},
  author={Yu, Dazhou and Zhang, Genpei and Zhao, Liang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
