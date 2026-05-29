# Code Availability

Use this file as the source for the manuscript's `Code availability` section.

## Nature Portfolio expectations

Nature Portfolio requires that custom code central to the manuscript's main claims be made available to editors and reviewers on request, and that the published paper include a dedicated `Code availability` statement describing how the code can be accessed and reused. Nature also states that best practice is to deposit code in a DOI-minting repository such as Zenodo or Code Ocean and to cite that archived release in the reference list.

Source:

- https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards
- https://support.nature.com/en/support/solutions/articles/6000237619-software-and-code-sharing

## Repository scope

This repository currently contains the code for:

- YOLOv8-based apoptosis and necroptosis detection
- RMH heatmap inference
- optional ConvNeXt-based apoptosis versus pyroptosis reassignment
- batch processing and aggregation utilities
- YOLO training entry points

## Suggested manuscript text

```text
Code availability
The DeathScope source code used for training, inference, heatmap aggregation and pyroptosis reassignment is available at https://github.com/LSUHSC-CAIPP-Core-B/DeathScope. An archived release corresponding to the version used in this manuscript is available at [insert Zenodo or other DOI-minting archive DOI here]. Any updates to the software after publication will be maintained in the GitHub repository.
```

## Before submission

- replace the placeholder DOI with the archived release DOI
- tag the exact manuscript-matched release in Git
- confirm that any scripts used to generate manuscript figures are present or explicitly documented elsewhere
- add a final software license to the repository root
