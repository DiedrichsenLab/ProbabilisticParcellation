# Software Specification Document
## Cerebellar Atlas Explorer

### 1. Introduction

This document outlines the specifications for the development of a Cerebellar Atlas Explorer, a software tool designed to load and visualize a surface-mapped cerebellar atlas. The primary goal of this application is to provide users with an interactive interface for exploring the function, cortical connectivity and spatial pattern of cerebellar regions. The software will be built using Plotly Dash (free version) and will be available as both an embeddable version for websites and a downloadable Jupyter notebook.

### 2. Functional Requirements

#### 2.1. Loading and Displaying Atlas Data

- The application should allow users to load a surface-mapped cerebellar atlas file in a supported format (e.g., NIfTI, numpy array).

#### 2.2. Interactive Cerebellar Flatmap Visualization

- The loaded atlas data will be displayed on a cerebellar flatmap as a clickable visualization.
- Users should be able to interact with the flatmap by clicking on cerebellar regions.

#### 2.3. Region Highlighting

- When a user clicks on a cerebellar region, the selected region should be highlighted, and all other regions should be faded.
- Highlighting can be achieved by changing the color or opacity of the selected region (Code for this exists, see resources).

#### 2.4. Information Display

- Upon clicking on a cerebellar region, the application should display the following information:
    1. Probabilistic region map for the selected region.
    2. Cortical connectivity weights for the selected region.
    3. Profile matrix of all cerebellar regions with the row of the selected region highlighted (other rows faded).

#### 2.5. Granularity Selection

- The application should include a drop-down menu or sliding menu that allows users to select the granularity at which the atlas is displayed (e.g., 4 regions, 32 regions, or 68 regions).

### 3. Non-functional Requirements

#### 3.1. Performance

- The application should provide a responsive and smooth user experience, with minimal latency in loading and displaying atlas data.

#### 3.2. Usability

- The user interface should be intuitive and user-friendly, ensuring that users can easily navigate and interact with the cerebellar atlas.

#### 3.3. Compatibility

- The software should be compatible with modern web browsers, and it should be tested across various devices and screen sizes to ensure responsiveness.

#### 3.4. Portability

- The downloadable Jupyter notebook version should be self-contained and easy to install, with clear installation instructions provided.

### 4. Additional Features (Nice-to-Have)

- In addition to the core features, the following enhancements are considered nice-to-have:
    - Support for multiple file formats for atlas data.
    - Save and export functionality for generated visualizations.
    - Zoom and pan controls for the cerebellar flatmap.
    - Integration with external databases or resources for additional information on cerebellar regions.
    - Overlay of multiple atlasses to compare regions.

### 5. Development Environment

- The software will be developed using Plotly Dash, a Python web framework for creating interactive web applications.
- The development environment should include version control (e.g., Git) and proper documentation.

### 6. Deliverables

The project should deliver the following:

- Source code of the Cerebellar Atlas Explorer.
- Documentation, including installation instructions and user guides.
- Embeddable version for websites.
- Downloadable Jupyter notebook version.

### 7. Timeline

The project should be completed within the allocated FHS timeframe. A minimal first version of the atlas exists and should be extended to include the additional features, making this timeframe realistic.

### 8. Conclusion

The successful implementation of this tool will enable users to explore and analyze cerebellar atlas data interactively, enhancing their understanding of cerebellar function and connectivity.

### 9. Resources

#### 9.1. Existing Cerebellar Atlas Viewers

The development of this Cerebellar Atlas Explorer can benefit from the following existing viewers as starting points and references:

1. **Brainhack Cerebellar Atlas Explorer** - [Link](https://github.com/carobellum/functional_atlas_explorer)
   - This viewer provides a user-friendly interface for visualizing cerebellar atlas data and offers interactive features.

2. **SUIT Atlas Viewer** - [Link](https://www.diedrichsenlab.org/imaging/AtlasViewer/index.htm)
   - The SUIT atlas viewer includes surface and volume visualisation.

3. **NeuroSynth** - [Link](https://neurosynth.org)
   - Neurosynth is a platform for large-scale, automated synthesis of functional magnetic resonance imaging data.

These existing viewers will serve as valuable references for understanding best practices, user interface design, and feature implementations. We will draw inspiration from these viewers to develop the cerebellar atlas explorer.

#### 9.2. Code

In addition to the referenced viewers, the following resources may be helpful during the development process:

- [Cerebellar Atlas Repository](https://github.com/DiedrichsenLab/cerebellar_atlases/tree/develop/Nettekoven_2023)
  - This repository contains atlasses of the cerebellum. We want to display the Nettekoven_2023 atlas.
  
- [Probabilistic Parcellation Code](https://github.com/DiedrichsenLab/ProbabilisticParcellation/tree/main)
  - This repository includes code for visualisation of cerebellar atlasses on the flatmap.

- [Plotly Dash Documentation](https://dash.plotly.com/interactive-graphing)
  - We will refer to the Plotly Dash documentation for guidance on building interactive web applications.


