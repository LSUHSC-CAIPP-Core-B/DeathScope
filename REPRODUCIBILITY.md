# Reproducibility Notes

This document summarizes the steps needed to make the public repository match the DeathScope manuscript as closely as possible.

## Repository expectations for publication

- The repository should correspond to a tagged release used by the manuscript.
- The README should describe the method, supported workflows, and limitations without relying on local lab knowledge.
- Data and model assets should be referenced by stable URLs or DOIs.
- The manuscript should include separate `Data availability` and `Code availability` sections.

## Recommended release contents

- source code for YOLO training and inference
- source code for the ConvNeXt pyroptosis classifier
- example dataset layout and configuration
- exact dependency pins
- citation metadata
- archival identifiers for data and model weights

## Items still requiring author decisions

- software license selection
- final archival home for raw and processed datasets
- final archival home for trained weights
- whether the manuscript document itself should be kept in the code repository or tracked separately

## Verification performed during repository preparation

- Python sources under `yolo/` and `train/` compile successfully
- import of `CellDetector` works from the checked-in environment after packaging-oriented import cleanup

## Suggested release process

1. Remove local-only artifacts from the public commit.
2. Upload data and model assets to an archival host with persistent identifiers.
3. Tag the manuscript-matched software release.
4. Update `README.md`, `CODE_AVAILABILITY.md`, and `DATA_AVAILABILITY.md` with the final links.
5. Cite the archival software and data records in the manuscript.
