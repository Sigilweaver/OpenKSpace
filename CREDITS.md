# Credits & acknowledgements

OpenKSpace is a clean-room implementation. No source code has been copied from the
projects below. They are cited because we studied their public documentation,
papers, file-format specifications, or behaviour for comparison.

## Citing this project

If you use OpenKSpace in published work, citing this repository is appreciated
but entirely optional. **What matters is citing the underlying algorithm papers.**
The authoritative references are listed in the *Algorithms & references* section
below — please credit those authors directly.

## Standards & specifications

- **ISMRMRD** — *ISMRM Raw Data format*. The HDF5 container layout, the
  `AcquisitionHeader` compound struct, the `EncodingCounters` indices, the
  acquisition-flag bit assignments, and the XML header schema are all defined
  by the ISMRMRD project (MIT licence). We implement a reader against that
  specification, using the public header file `ismrmrd.h` and the schema
  documentation only.
  <https://ismrmrd.github.io/>

- **mridata.org** — public repository of fully-sampled and prospectively
  undersampled raw k-space datasets. Used for validation only; data is not
  redistributed in this repository.

## Algorithms & references

When we add features beyond a plain IFFT+RSS we credit the original papers here,
not implementations.

- *Griswold MA et al.* "Generalized autocalibrating partially parallel
  acquisitions (GRAPPA)." **MRM** 47(6):1202–1210, 2002. — reference for the
  GRAPPA kernel calibration.

- *Pruessmann KP et al.* "SENSE: sensitivity encoding for fast MRI."
  **MRM** 42(5):952–962, 1999. — reference for image-domain unfolding.

- *Walsh DO, Gmitro AF, Marcellin MW.* "Adaptive reconstruction of
  phased array MR imagery." **MRM** 43(5):682–690, 2000. — reference
  for the iterative-eigenvector method used by our Walsh
  sensitivity-map estimator.

- *Uecker M et al.* "ESPIRiT — an eigenvalue approach to autocalibrating
  parallel MRI." **MRM** 71(3):990–1001, 2014. — reference for coil sensitivity
  maps.

- *Bernstein MA, King KF, Zhou XJ.* **Handbook of MRI Pulse Sequences**.
  Academic Press, 2004. — reference text for navigator phase correction
  (§13.5), readout oversampling handling, partial-Fourier homodyne.

- *Noll DC, Nishimura DG, Macovski A.* "Homodyne detection in magnetic
  resonance imaging." **IEEE Transactions on Medical Imaging** 10(2):
  154–163, 1991. — reference for the partial-Fourier homodyne
  reconstruction (ramp/step k-space weighting + low-frequency phase
  demodulation).

- *McGibney G, Smith MR, Nichols ST, Crawley A.* "Quantitative evaluation
  of several partial Fourier reconstruction algorithms used in MRI."
  **MRM** 30(1):51–59, 1993. — comparison of partial-Fourier techniques
  used when designing our weighting profile.

- *Lustig M, Donoho D, Pauly JM.* "Sparse MRI: the application of compressed
  sensing for rapid MR imaging." **MRM** 58(6):1182–1195, 2007. — reference
  for the compressed-sensing reconstruction with an L1-wavelet prior.

- *Beck A, Teboulle M.* "A fast iterative shrinkage-thresholding algorithm
  for linear inverse problems." **SIAM Journal on Imaging Sciences**
  2(1):183–202, 2009. — reference for the FISTA solver used by our CS
  strategy.

- *Hammernik K et al.* "Learning a variational network for reconstruction of
  accelerated MRI data." **MRM** 79(6):3055–3071, 2018. — origin of several
  fully-sampled knee datasets used in our validation corpus.

## Open-source projects consulted (not copied)

We have inspected these projects' *documentation and behaviour* to understand
expected outputs and edge cases. None of their source code has been
incorporated into OpenKSpace.

- **BART — Berkeley Advanced Reconstruction Toolbox** (BSD-2-Clause).
  Reference for expected reconstruction output.
  <https://mrirecon.github.io/bart/>

- **Gadgetron** (MIT). Reference for the standard preprocessing pipeline
  (noise pre-whitening → phase correction → recon).
  <https://github.com/gadgetron/gadgetron>

- **ismrmrd-python-tools / ismrmrdpy** (MIT). Reference for the ISMRMRD
  file layout and sample iteration patterns.
  <https://github.com/ismrmrd/ismrmrdpy>

## Rust ecosystem dependencies

See `Cargo.toml` for the exhaustive list; notable direct dependencies include
`hdf5-metno`, `rustfft`, `ndarray`, `num-complex`, `quick-xml`, `clap`,
`rayon`, and `tracing`. All are under permissive licences (MIT / Apache-2.0 /
BSD). We gratefully use them.

## Citation

If you use Sigil in research, please cite the underlying algorithm paper for
whichever reconstruction method you invoked (see above) and, optionally,
this repository.
